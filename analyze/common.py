"""ตัวช่วยกลางของโฟลเดอร์ analyze/ — ใช้ร่วมกันทั้ง 3 อัลกอริทึม

หน้าที่
  - โหลดชุดข้อมูลที่ผ่าน pre-processing แล้วจาก outputs/processed.npz
  - เทรน + วัดผลบน test set ที่คงการกระจายจริงไว้ (fraud ~0.17%)
  - ตัวชี้วัดหลักคือ AUPRC (PR-AUC) และ Recall ตามวัตถุประสงค์ใน Introduction.txt
    **ไม่ใช่** accuracy เพราะทายว่า "ไม่โกง" ทุกแถวก็ได้ accuracy 99.8%
  - เขียนผลออกเป็น CSV (utf-8-sig) / JSON / PNG ให้ compare_all.py มารวมต่อ
  - จด progress ลง logs/analyze_progress.log (append) เผื่อปิดเครื่องแล้วรันต่อ

กฎที่ยึดตาม AGENTS.md
  - ไม่มี data leakage: เทรนบน X_train เท่านั้น แล้ว predict ลง X_test
  - SMOTE ใช้ได้แต่กับ train (X_train_smote มาจาก step_smote แล้ว) ห้ามแตะ test
  - ส่ง random_state=RANDOM_STATE ให้ทุกโมเดลที่สุ่ม เพื่อให้รันซ้ำได้ผลเดิม
  - ค่าใน dict สรุปต้องเป็น Python native ก่อนเขียน JSON
  - log ภาษาไทยผ่าน logger กลาง (agent/logger.py) ไม่ print ภาษาไทยตรง ๆ
"""
from pathlib import Path
import json
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             precision_recall_curve, confusion_matrix)

# Paths — ต้องใส่ ROOT ลง sys.path ก่อน เพื่อ import logger กลางของโปรเจกต์ได้
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.logger import get_logger  # noqa: E402  (ต้องมาหลัง sys.path)

PROCESSED_NPZ = ROOT / "outputs" / "processed.npz"
ANALYZE_DIR = Path(__file__).resolve().parent
PROGRESS_LOG = ROOT / "logs" / "analyze_progress.log"

log = get_logger("analyze")

# hyperparameter กลางของขั้นวิเคราะห์ (ไม่ hardcode ซ้ำในแต่ละสคริปต์)
RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.5
# grid ของ threshold ที่เขียนลง CSV — เติมค่าปลายสูงเพราะโมเดลที่ถ่วงน้ำหนักคลาส
# ดันคะแนนไปกองใกล้ 1 จุดตัดที่ดีที่สุดจึงมักอยู่เหนือ 0.95
THRESHOLDS = ([round(0.05 * i, 2) for i in range(1, 20)]
              + [0.97, 0.99, 0.995, 0.999])
# เกณฑ์ธุรกิจ: ยอม false positive ได้ระดับหนึ่ง ถ้า precision ไม่ต่ำกว่าค่านี้
MIN_PRECISION = 0.50

# จานสีกลางของ analyze/ — ผ่านการตรวจ colorblind-safety แล้ว
# (categorical slot 1-3 สำหรับ "ตัวตน" ของอัลกอริทึม + status สำหรับชนิดความผิดพลาด)
# worst-pair CVD ΔE 9.2, normal-vision ΔE 24.0 บนพื้นขาว
PALETTE = {
    # ตัวตนของอัลกอริทึม — สีติดตัวโมเดล ไม่เปลี่ยนตามอันดับหรือตามกราฟ
    "lr": "#2a78d6",      # slot 1 blue
    "rf": "#eb6834",      # slot 2 orange
    "gb": "#1baf7a",      # slot 3 aqua
    # ชนิดของความผิดพลาด (status — ใช้คู่กับ label เสมอ ไม่ให้สีสื่อความหมายลำพัง)
    "fn": "#d03b3b",      # critical: พลาด fraud
    "fp": "#ec835a",      # serious: แจ้งเตือนผิด
    # ตัววัดภายในกราฟเดียว (ใช้ slot 1-3 ตามลำดับคงที่)
    "m1": "#2a78d6", "m2": "#eb6834", "m3": "#1baf7a",
    # หมึกและเส้นประกอบ
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#ffffff",
}

# ใช้ Leelawadee UI เพื่อให้ใส่หัวข้อภาษาไทยในกราฟได้ (ไม่งั้นได้กล่องสี่เหลี่ยม)
# ถ้าเครื่องไหนไม่มี matplotlib จะ fallback ไป Tahoma / DejaVu Sans เอง
plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                     "font.family": ["Leelawadee UI", "Tahoma", "DejaVu Sans"],
                     "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False})


def mark_progress(step, status, detail=""):
    """จด progress ลง logs/analyze_progress.log แบบ append (ไม่ทับของเดิม)

    ใช้สำหรับกรณีปิดเครื่องกลางทาง: เปิดไฟล์นี้ดูว่า step ไหน DONE แล้ว
    แล้วรันต่อเฉพาะ step ที่ยังไม่ DONE
    """
    PROGRESS_LOG.parent.mkdir(exist_ok=True)
    line = (f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {status:<7} | "
            f"{step}" + (f" | {detail}" if detail else ""))
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    log.info("progress: %s -> %s %s", step, status, detail)


def load_data():
    """โหลด train / test / train(SMOTE) ที่ step_save เขียนไว้

    คืน dict เพื่อให้สคริปต์เลือกใช้เฉพาะชุดที่ต้องการ
    """
    if not PROCESSED_NPZ.exists():
        raise FileNotFoundError(
            f"ไม่พบ {PROCESSED_NPZ} — ต้องรัน `python main.py` ให้จบก่อน")
    log.info("โหลดข้อมูลที่ pre-process แล้วจาก outputs/%s", PROCESSED_NPZ.name)
    d = np.load(PROCESSED_NPZ)
    data = {
        "X_train": d["X_train"], "y_train": d["y_train"],
        "X_test": d["X_test"], "y_test": d["y_test"],
        "X_smote": d["X_train_smote"], "y_smote": d["y_train_smote"],
        "features": [str(f) for f in d["features"]],
    }
    log.info("train=%s (fraud=%d) | test=%s (fraud=%d) | train+SMOTE=%s (fraud=%d)",
             data["X_train"].shape, int(data["y_train"].sum()),
             data["X_test"].shape, int(data["y_test"].sum()),
             data["X_smote"].shape, int(data["y_smote"].sum()))
    log.info("ฟีเจอร์ %d คอลัมน์: %s", len(data["features"]),
             ", ".join(data["features"]))
    return data


def _metrics_at(y_true, scores, threshold):
    """คำนวณตัวชี้วัดที่ threshold ที่กำหนด (ค่าทั้งหมด cast เป็น native แล้ว)"""
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 6),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def evaluate(name, model, X_train, y_train, X_test, y_test, train_label):
    """เทรนโมเดลแล้ววัดผลบน test set — คืน (dict สรุป, คะแนน, โมเดลที่ fit แล้ว)

    คืนโมเดลกลับไปด้วย เพื่อให้สคริปต์ดึง coefficient / feature importance
    ต่อได้โดยไม่ต้อง fit ซ้ำ

    dict ที่คืนมามีรูปแบบเดียวกันทุกอัลกอริทึม เพื่อให้ compare_all.py
    เอาไปรวมเป็นตารางเดียวได้โดยไม่ต้องรู้ว่าเป็นโมเดลอะไร
    """
    log.info("เทรน %s (train=%s, %d แถว)", name, train_label, len(y_train))
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_sec = round(time.time() - t0, 2)

    t1 = time.time()
    scores = model.predict_proba(X_test)[:, 1]
    pred_sec = round(time.time() - t1, 2)

    base = _metrics_at(y_test, scores, DEFAULT_THRESHOLD)
    sweep = [_metrics_at(y_test, scores, t) for t in THRESHOLDS]

    # หา threshold ที่ดีที่สุดจาก PR curve เต็มเส้น (ไม่ใช่จาก grid หยาบ ๆ)
    # เพราะจุดตัดที่เหมาะสมมักอยู่ในช่วงแคบ ๆ ใกล้ 1 ที่ grid มองไม่เห็น
    pr, rc, th = precision_recall_curve(y_test, scores)
    pr, rc = pr[:-1], rc[:-1]  # จุดสุดท้ายของ curve ไม่มี threshold คู่กัน
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_curve = np.nan_to_num(2 * pr * rc / (pr + rc))
    best_f1 = _metrics_at(y_test, scores, float(th[int(np.argmax(f1_curve))]))

    # จุดที่ได้ recall สูงสุด โดย precision ยังไม่ต่ำกว่าเกณฑ์ที่ยอมรับได้
    ok = np.flatnonzero(pr >= MIN_PRECISION)
    best_recall_ok = (_metrics_at(y_test, scores, float(th[ok[int(np.argmax(rc[ok]))]]))
                      if ok.size else None)

    summary = {
        "name": name,
        "train_set": train_label,
        "train_rows": int(len(y_train)),
        "pr_auc": round(float(average_precision_score(y_test, scores)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, scores)), 4),
        "at_threshold_0.5": base,
        "best_f1_threshold": best_f1,
        f"best_recall_with_precision_ge_{MIN_PRECISION}": best_recall_ok,
        "fit_seconds": fit_sec,
        "predict_seconds": pred_sec,
        "threshold_sweep": sweep,
    }
    log.info("%-32s AUPRC=%.4f ROC-AUC=%.4f | @0.5 recall=%.4f precision=%.4f "
             "f1=%.4f (TP=%d FN=%d FP=%d) | fit %.2fs",
             name, summary["pr_auc"], summary["roc_auc"], base["recall"],
             base["precision"], base["f1"], base["tp"], base["fn"], base["fp"],
             fit_sec)
    log.info("  จุดที่ F1 ดีที่สุดอยู่ที่ threshold=%.6f -> F1=%.4f "
             "(recall=%.4f precision=%.4f)",
             best_f1["threshold"], best_f1["f1"], best_f1["recall"],
             best_f1["precision"])
    return summary, scores, model


def write_results(out_dir, prefix, method, model_params, variants, y_test,
                  score_map, extra=None):
    """เขียน CSV / JSON / คะแนน .npz ของอัลกอริทึมหนึ่งตัว

    out_dir   : โฟลเดอร์ย่อยของอัลกอริทึมนั้น
    variants  : list ของ dict ที่ได้จาก evaluate()
    score_map : {ชื่อ variant: scores} เก็บไว้ให้ compare_all วาด PR curve ต่อ
    """
    out_dir = Path(out_dir)

    # ตารางเทียบ variant (แถวละ variant) — อ่านใน Excel ได้
    rows = []
    for v in variants:
        b = v["at_threshold_0.5"]
        rows.append({
            "Variant": v["name"], "Train_Set": v["train_set"],
            "PR_AUC": v["pr_auc"], "ROC_AUC": v["roc_auc"],
            "Precision": b["precision"], "Recall": b["recall"], "F1": b["f1"],
            "TP": b["tp"], "FP": b["fp"], "FN": b["fn"], "TN": b["tn"],
            "Best_F1": v["best_f1_threshold"]["f1"],
            "Best_F1_Threshold": v["best_f1_threshold"]["threshold"],
            "Fit_Seconds": v["fit_seconds"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{prefix}_results.csv", index=False, encoding="utf-8-sig")

    # ตาราง threshold sweep ของทุก variant (แถวละ threshold)
    sweep_rows = []
    for v in variants:
        for m in v["threshold_sweep"]:
            sweep_rows.append({
                "Variant": v["name"], "Threshold": m["threshold"],
                "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"],
                "TP": m["tp"], "FP": m["fp"], "FN": m["fn"], "TN": m["tn"]})
    pd.DataFrame(sweep_rows).to_csv(
        out_dir / f"{prefix}_thresholds.csv", index=False, encoding="utf-8-sig")

    best = max(variants, key=lambda v: v["pr_auc"])
    summary = {
        "method": method,
        "objective": "จำแนกธุรกรรมทุจริต ประเมินด้วย AUPRC (ตาม Introduction.txt)",
        "dataset": {
            "source": "outputs/processed.npz (ผลจาก pre-processing pipeline)",
            "train_rows": int(variants[0]["train_rows"]),
            "test_rows": int(len(y_test)),
            "test_fraud": int(np.sum(y_test)),
            "test_fraud_pct": round(100 * float(np.mean(y_test)), 4),
            "note": "test set ไม่ถูก resample คงการกระจายจริงไว้",
        },
        "model_params": model_params,
        "random_state": RANDOM_STATE,
        "primary_metric": "pr_auc (AUPRC)",
        "best_variant": {"name": best["name"], "pr_auc": best["pr_auc"]},
        "variants": variants,
    }
    if extra:
        summary.update(extra)
    with open(out_dir / f"{prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # เก็บคะแนนไว้ให้ compare_all.py วาด PR curve รวม ไม่ต้องเทรนซ้ำ
    np.savez_compressed(
        out_dir / f"{prefix}_test_scores.npz",
        y_test=np.asarray(y_test).astype(np.int8),
        **{k: np.asarray(v).astype(np.float32) for k, v in score_map.items()})

    log.info("บันทึก %s_results.csv / %s_thresholds.csv / %s_summary.json / "
             "%s_test_scores.npz", prefix, prefix, prefix, prefix)
    return df, summary


def plot_algorithm(out_dir, prefix, title, variants, y_test, score_map, color):
    """กราฟ 3 ช่อง: PR curve / ตัวชี้วัดเทียบ variant / confusion matrix ของตัวที่ดีที่สุด

    title ใช้ภาษาอังกฤษ เพราะ matplotlib ที่ติดตั้งไว้ไม่มีฟอนต์ไทย
    (ถ้าใส่ไทยจะได้กล่องสี่เหลี่ยม — ดู AGENTS.md)
    """
    out_dir = Path(out_dir)
    best = max(variants, key=lambda v: v["pr_auc"])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=150)

    # 1) PR curve ของทุก variant
    for i, v in enumerate(variants):
        pr, rc, _ = precision_recall_curve(y_test, score_map[v["name"]])
        axes[0].plot(rc, pr, linewidth=1.6,
                     color=color if i == 0 else PALETTE["muted"],
                     linestyle="-" if i == 0 else "--",
                     label=f"{v['name']} (AUPRC={v['pr_auc']:.3f})")
    axes[0].set_title("Precision-Recall Curve", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1.02)
    axes[0].legend(loc="lower left", fontsize=7.5, frameon=True)

    # 2) ตัวชี้วัดหลักเทียบ variant (ไม่แสดง accuracy โดยเจตนา)
    x = np.arange(len(variants)); w = 0.26
    for i, (k, lb) in enumerate([("recall", "Recall"), ("precision", "Precision")]):
        vals = [v["at_threshold_0.5"][k] for v in variants]
        axes[1].bar(x + (i - 1) * w, vals, w, label=lb,
                    color=[PALETTE["m1"], PALETTE["m2"]][i])
        for xi, val in zip(x + (i - 1) * w, vals):
            axes[1].text(xi, val, f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)
    aucs = [v["pr_auc"] for v in variants]
    axes[1].bar(x + w, aucs, w, label="AUPRC", color=PALETTE["m3"])
    for xi, val in zip(x + w, aucs):
        axes[1].text(xi, val, f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([v["name"] for v in variants], fontsize=8)
    axes[1].set_ylim(0, 1.15)
    axes[1].legend(ncol=3, frameon=False, fontsize=8)
    axes[1].set_title("Recall / Precision / AUPRC (threshold 0.5)",
                      fontsize=11, fontweight="bold")

    # 3) confusion matrix ของ variant ที่ AUPRC ดีที่สุด
    b = best["at_threshold_0.5"]
    cm = np.array([[b["tn"], b["fp"]], [b["fn"], b["tp"]]])
    shade = np.log1p(cm)
    axes[2].imshow(shade, cmap="Blues")
    for (i, j), val in np.ndenumerate(cm):
        axes[2].text(j, i, f"{val:,}", ha="center", va="center", fontsize=11,
                     fontweight="bold",
                     color="#ffffff" if shade[i, j] > shade.max() * 0.6 else "#1f2937")
    axes[2].set_xticks([0, 1]); axes[2].set_xticklabels(["pred 0", "pred 1"])
    axes[2].set_yticks([0, 1]); axes[2].set_yticklabels(["true 0", "true 1"])
    axes[2].set_title(f"Confusion Matrix - {best['name']}", fontsize=11,
                      fontweight="bold")
    for s in axes[2].spines.values():
        s.set_visible(False)

    for ax in axes[:2]:
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    img = out_dir / f"{prefix}_analysis.png"
    fig.savefig(img, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    log.info("บันทึกกราฟ %s", img.name)
    return img
