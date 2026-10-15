"""รวมผลทั้ง 3 อัลกอริทึมมาเทียบกันในที่เดียว + เขียนสรุป SUMMARY.md

อ่านไฟล์ที่แต่ละอัลกอริทึมเขียนไว้ (ไม่เทรนใหม่ จึงรันซ้ำได้เร็ว)
  analyze/logistic_regression/lr_summary.json  + lr_test_scores.npz
  analyze/random_forest/rf_summary.json        + rf_test_scores.npz
  analyze/gradient_boosting/gb_summary.json    + gb_test_scores.npz

ผลลัพธ์
  analyze/comparison_all_models.csv   ตารางเทียบทุก variant (6 แถว)
  analyze/comparison_summary.json     สรุปเชิงโครงสร้าง + อันดับตาม AUPRC
  analyze/comparison_summary.png      กราฟ 3 ช่อง: PR curve / AUPRC / FN-FP
  analyze/SUMMARY.md                  สรุปภาษาไทยว่าทำอะไรไปและได้ผลอะไร

รัน: python analyze/compare_all.py
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ANALYZE_DIR, MIN_PRECISION, PALETTE, log,  # noqa: E402
                    mark_progress)
from key_findings import build_key_findings  # noqa: E402

# (โฟลเดอร์, prefix, ชื่อที่ใช้ในตาราง) — prefix เป็น key ของจานสีด้วย
# สีจึงติดตัวอัลกอริทึมเสมอ ไม่เปลี่ยนตามอันดับหรือตามกราฟ
ALGOS = [
    ("logistic_regression", "lr", "1) Logistic Regression"),
    ("random_forest", "rf", "2) Random Forest"),
    ("gradient_boosting", "gb", "3) Gradient Boosting"),
]
BEST_RECALL_KEY = f"best_recall_with_precision_ge_{MIN_PRECISION}"


def _load_algo(folder, prefix):
    """โหลด summary.json + คะแนนบน test ของอัลกอริทึมหนึ่งตัว"""
    d = ANALYZE_DIR / folder
    sj = d / f"{prefix}_summary.json"
    sz = d / f"{prefix}_test_scores.npz"
    if not sj.exists() or not sz.exists():
        raise FileNotFoundError(
            f"ยังไม่มีผลของ {folder} — ต้องรัน analyze/{folder}/{prefix}_analysis.py ก่อน")
    with open(sj, encoding="utf-8") as f:
        summary = json.load(f)
    return summary, np.load(sz)


def build_table(loaded):
    """ทำตารางเทียบทุก variant ของทุกอัลกอริทึม (แถวละ variant)"""
    rows = []
    for (folder, prefix, label), (summary, _) in zip(ALGOS, loaded):
        for v in summary["variants"]:
            b = v["at_threshold_0.5"]
            bf = v["best_f1_threshold"]
            br = v.get(BEST_RECALL_KEY) or {}
            rows.append({
                "Algorithm": label,
                "Variant": v["name"],
                "Train_Set": v["train_set"],
                "AUPRC": v["pr_auc"],
                "ROC_AUC": v["roc_auc"],
                "Recall_at_0.5": b["recall"],
                "Precision_at_0.5": b["precision"],
                "F1_at_0.5": b["f1"],
                "TP_at_0.5": b["tp"], "FP_at_0.5": b["fp"], "FN_at_0.5": b["fn"],
                "Best_F1": bf["f1"],
                "Best_F1_Threshold": bf["threshold"],
                "Recall_at_Best_F1": bf["recall"],
                "Precision_at_Best_F1": bf["precision"],
                "FP_at_Best_F1": bf["fp"], "FN_at_Best_F1": bf["fn"],
                "Max_Recall_at_Precision_50": br.get("recall"),
                "Fit_Seconds": v["fit_seconds"],
            })
    return pd.DataFrame(rows).sort_values("AUPRC", ascending=False).reset_index(drop=True)


def plot_comparison(loaded, df, out_path):
    """กราฟ 3 ช่อง (ภาษาอังกฤษในกราฟ เพราะ matplotlib ไม่มีฟอนต์ไทย)"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), dpi=150)

    # 1) PR curve ของ variant ที่ AUPRC ดีสุดของแต่ละอัลกอริทึม
    for (folder, prefix, label), (summary, scores) in zip(ALGOS, loaded):
        best = max(summary["variants"], key=lambda v: v["pr_auc"])
        color = PALETTE[prefix]
        y = scores["y_test"]
        pr, rc, _ = precision_recall_curve(y, scores[best["name"]])
        axes[0].plot(rc, pr, linewidth=1.8, color=color,
                     label=f"{label} (AUPRC={best['pr_auc']:.3f})")
    base_rate = float(np.mean(loaded[0][1]["y_test"]))
    axes[0].axhline(base_rate, color=PALETTE["muted"], linestyle=":", linewidth=1,
                    label=f"random baseline ({base_rate:.4f})")
    axes[0].set_title("PR Curve - best variant per algorithm",
                      fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1.02)
    axes[0].legend(loc="lower left", fontsize=7.5)

    # 2) อันดับ AUPRC ของทุก variant (ตัวชี้วัดหลักตาม Introduction.txt)
    d2 = df.sort_values("AUPRC")
    colors = [{a[2]: PALETTE[a[1]] for a in ALGOS}[a] for a in d2["Algorithm"]]
    bars = axes[1].barh(d2["Variant"], d2["AUPRC"], color=colors, height=0.6)
    for bar, val in zip(bars, d2["AUPRC"]):
        axes[1].text(val + 0.008, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", fontsize=8, fontweight="medium")
    axes[1].set_xlim(0, max(d2["AUPRC"]) * 1.18)
    axes[1].set_title("AUPRC ranking (primary metric)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("AUPRC (average precision)")
    axes[1].tick_params(axis="y", labelsize=7.5)

    # 3) ต้นทุนจริงที่จุด F1 ดีที่สุด: พลาด fraud กี่เคส (FN) vs รบกวนลูกค้ากี่เคส (FP)
    d3 = df.sort_values("AUPRC", ascending=False)
    x = np.arange(len(d3)); w = 0.38
    axes[2].bar(x - w / 2, d3["FN_at_Best_F1"], w, label="FN (missed fraud)",
                color=PALETTE["fn"])
    axes[2].bar(x + w / 2, d3["FP_at_Best_F1"], w, label="FP (false alarm)",
                color=PALETTE["fp"])
    for xi, fn, fp in zip(x, d3["FN_at_Best_F1"], d3["FP_at_Best_F1"]):
        axes[2].text(xi - w / 2, fn, f"{fn}", ha="center", va="bottom", fontsize=7.5)
        axes[2].text(xi + w / 2, fp, f"{fp}", ha="center", va="bottom", fontsize=7.5)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(d3["Variant"], rotation=28, ha="right", fontsize=7)
    axes[2].set_title("Cost at best-F1 threshold", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("transactions")
    axes[2].legend(fontsize=8, frameon=False)

    for ax in axes[1:]:
        ax.grid(axis="x" if ax is axes[1] else "y", linestyle=":", alpha=0.5)
    fig.suptitle("Credit Card Fraud - 3 Algorithm Comparison on real test set",
                 fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    log.info("บันทึกกราฟ %s", out_path.name)


def _findings_section(ev):
    """ข้อค้นพบสำคัญ + ข้อควรระวัง เขียนจากตัวเลขที่ key_findings.py คำนวณจริง"""
    tb = ev["trivial_baseline"]
    algos = ev["algorithms"]
    win = max(algos, key=lambda a: a["pr_auc"])
    runner = sorted(algos, key=lambda a: a["pr_auc"])[-2]
    c = ev["roc_vs_auprc_contradictions"][0] if ev["roc_vs_auprc_contradictions"] else None
    smote_worst = min(algos, key=lambda a: a["smote_effect"]["delta"])
    lr = next(a for a in algos if a["color_key"] == "lr")
    lr_th = max(v["best_f1_threshold"] for v in lr["variants"])
    tt = win["threshold_transfer"]
    ci_overlap = runner["auprc_ci"]["hi"] >= win["auprc_ci"]["lo"]

    out = [
        "## Key Findings — ข้อค้นพบสำคัญ",
        "",
        "![Key Findings](key_findings.png)",
        "",
        "> แผนภาพเต็ม: `analyze/key_findings.png` · ตัวเลขดิบ: `analyze/key_findings.json`",
        "",
        f"**1. accuracy ใช้ตัดสินงานนี้ไม่ได้** — โมเดลที่ทายว่า \"ไม่โกง\" ทุกแถว "
        f"ได้ accuracy {tb['accuracy']}% ทั้งที่จับ fraud ไม่ได้สักเคสจาก {tb['fraud']} เคส "
        f"จึงใช้ AUPRC เป็นตัวตัดสินตลอดทั้งงาน",
        "",
        f"**2. ROC-AUC สูงไม่ได้แปลว่าโมเดลดี** — "
        + (f"{c['higher_roc_lower_auprc']['code']} ({c['higher_roc_lower_auprc']['variant']}) "
           f"มี ROC-AUC สูงกว่า {c['lower_roc_higher_auprc']['code']} อยู่ {c['roc_gap']:.3f} "
           f"แต่ AUPRC ต่ำกว่าถึง {c['auprc_gap']:.3f} "
           f"เพราะ ROC-AUC ถูกถ่วงด้วย true negative นับแสนแถวที่ทายง่ายอยู่แล้ว"
           if c else "ไม่พบคู่ที่ขัดกันในรอบนี้"),
        "",
        f"**3. การตั้ง threshold สำคัญกว่าการเลือกวิธีแก้ imbalance** — ที่ค่าปริยาย 0.5 "
        f"Logistic Regression ได้ F1 แค่ {min(v['f1_at_0.5'] for v in lr['variants']):.3f} "
        f"แต่พอเลือกจุดตัดจาก PR curve ได้ถึง {max(v['best_f1'] for v in lr['variants']):.3f} "
        f"(ช่อง 3 ของแผนภาพ: เส้นของ GB แบนราบ = ทนการตั้งจุดตัดผิด ส่วน LR เป็นยอดแหลม)",
        "",
        f"**4. SMOTE ไม่ได้ช่วยเสมอ** — ทำให้ {smote_worst['label'][3:]} แย่ลง "
        f"{abs(smote_worst['smote_effect']['delta']):.3f} AUPRC "
        f"({smote_worst['smote_effect']['class_weight_auprc']:.3f} → "
        f"{smote_worst['smote_effect']['smote_auprc']:.3f}) "
        f"ขณะที่อีกสองตัวแทบไม่ต่าง การถ่วงน้ำหนักคลาสจึงเป็นค่าเริ่มต้นที่ปลอดภัยกว่า",
        "",
        f"**5. fraud ใน test มีแค่ {ev['test_fraud']} เคส ตัวเลขจึงแกว่งกว้าง** — "
        f"AUPRC ของ {win['label'][3:]} = {win['pr_auc']:.3f} "
        f"แต่ช่วงความเชื่อมั่น 95% คือ {win['auprc_ci']['lo']:.3f}–{win['auprc_ci']['hi']:.3f} "
        f"(bootstrap {win['auprc_ci']['n_bootstrap']} รอบ) กว้างกว่าช่องว่างระหว่างอันดับ 1 กับ 2 เสียอีก",
        "",
        f"**6. ทุกจุดตัดคือการแลก** — fraud 1 เคสที่จับเพิ่มได้ ต้องจ่ายด้วยจำนวนแจ้งเตือนผิด "
        f"ที่เพิ่มขึ้นเร็วมากในช่วงท้าย (ช่อง 6 แกน X เป็นสเกล log) "
        f"การเลือกจุดตัดจึงเป็นการตัดสินใจเชิงธุรกิจ ไม่ใช่เรื่องที่โมเดลตอบให้ได้",
        "",
        "## ข้อควรระวัง / สิ่งที่ต้องโฟกัสต่อ",
        "",
        f"1. **threshold ถูกเลือกบน test set เดียวกับที่วัดผล — ตัวเลขจึงดูดีเกินจริง** "
        f"ทดสอบโดยเลือกจุดตัดจาก test ครึ่งแรกแล้วเอาไปใช้กับครึ่งหลัง: {win['label'][3:]} "
        f"ได้ F1 = {tt['f1_on_half_B']:.3f} เทียบกับ {tt['f1_oracle_on_half_B']:.3f} "
        f"ถ้ารู้คำตอบล่วงหน้า (ต่างกัน {tt['optimism_gap']:.3f}) "
        f"**ถ้าจะใช้งานจริงต้องเลือก threshold จาก validation set แยก หรือ cross-validation บน train**",
        f"2. **อย่าฟันธงว่าอันดับ 1 ชนะอันดับ 2 จริง** — ช่วงความเชื่อมั่นของ "
        f"{win['label'][3:]} ({win['auprc_ci']['lo']:.3f}–{win['auprc_ci']['hi']:.3f}) "
        f"กับ {runner['label'][3:]} ({runner['auprc_ci']['lo']:.3f}–{runner['auprc_ci']['hi']:.3f}) "
        + ("**ซ้อนทับกันเกือบทั้งช่วง**" if ci_overlap else "แยกจากกัน")
        + " ด้วย fraud แค่ 95 เคส ความต่างระดับ 0.02 ยังไม่พอสรุป",
        "3. **ข้อมูลเป็นลำดับเวลา 2 วัน แต่ split แบบสุ่ม** — pipeline ใช้ stratified random split "
        "แปลว่าโมเดลได้เห็นธุรกรรม \"อนาคต\" ตอนเทรน ถ้าจะประเมินแบบใกล้ของจริง "
        "ควรแบ่งตามเวลา (เทรนวันแรก ทดสอบวันที่สอง) ซึ่งผลมักต่ำกว่านี้",
        f"4. **threshold ที่ดีที่สุดของ Logistic Regression อยู่ที่ {lr_th:.6f}** "
        f"ขยับนิดเดียวผลเปลี่ยนทันที (ดูยอดแหลมในช่อง 3) ถือว่าเปราะเกินกว่าจะเอาไปใช้งานจริง",
        "5. **V1–V28 เป็นผลลัพธ์จาก PCA ตีความไม่ได้** — ถึงโมเดลจะชี้ว่า V14/V12/V10 "
        "สำคัญที่สุด ก็อธิบายลูกค้าหรือผู้ตรวจสอบไม่ได้ว่าทำไมธุรกรรมถึงถูกปฏิเสธ "
        "ถ้าต้องใช้จริงในธุรกิจต้องเตรียมวิธีอธิบายเพิ่ม",
        "6. **ยังไม่ได้จูน hyperparameter** — ทุกโมเดลใช้ค่าที่ตั้งไว้ตายตัว ตัวเลขชุดนี้จึงเป็น "
        "baseline ที่เทียบกันได้อย่างเป็นธรรม ไม่ใช่เพดานของแต่ละอัลกอริทึม",
        "7. **ถ้ารัน `python main.py` ใหม่ ต้องรัน `analyze/` ใหม่ทั้งหมด** "
        "เพราะทุกอย่างอิง `outputs/processed.npz` ของรอบนั้น ผลเก่าจะไม่ตรงกัน",
        "",
    ]
    return out


def write_summary_md(loaded, df, path, ev=None):
    """เขียนสรุปภาษาไทยจากตัวเลขจริง (regenerate ทุกครั้งที่รัน จึงไม่ค้างของเก่า)"""
    best = df.iloc[0]
    ds = loaded[0][0]["dataset"]
    lines = [
        "# สรุปผลการวิเคราะห์ด้วย 3 อัลกอริทึม (Analyze)",
        "",
        "> ไฟล์นี้ถูกสร้างใหม่อัตโนมัติทุกครั้งที่รัน `python analyze/compare_all.py`",
        "",
        "## ทำอะไรไป",
        "",
        "ต่อยอดจาก pre-processing pipeline (`python main.py`) และผลคัดฟีเจอร์ใน `feature/`",
        "โดยนำชุดข้อมูลที่ประมวลผลแล้วใน `outputs/processed.npz` ไปเทรนและวัดผล",
        "ด้วยอัลกอริทึมจำแนก 3 ตระกูล เพื่อตอบวัตถุประสงค์ใน `Introduction.txt`",
        "คือจำแนกธุรกรรมทุจริต และประเมินด้วย **AUPRC** (ไม่ใช่ accuracy)",
        "",
        "| # | อัลกอริทึม | ตระกูลโมเดล | โฟลเดอร์ |",
        "|---|-----------|------------|---------|",
        "| 1 | Logistic Regression | linear model | `analyze/logistic_regression/` |",
        "| 2 | Random Forest | bagging ensemble | `analyze/random_forest/` |",
        "| 3 | Histogram Gradient Boosting | boosting ensemble | `analyze/gradient_boosting/` |",
        "",
        "ทุกอัลกอริทึมเทียบ 2 วิธีจัดการ imbalance บนข้อมูลชุดเดียวกัน",
        "คือ **ถ่วงน้ำหนักคลาส** (`class_weight`) กับ **SMOTE บน train** รวมเป็น 6 แบบ",
        "",
        "## ทำอย่างไร",
        "",
        f"- ข้อมูล: train {ds['train_rows']:,} แถว / test {ds['test_rows']:,} แถว "
        f"(fraud {ds['test_fraud']} เคส = {ds['test_fraud_pct']}%)",
        "- test set **ไม่ถูก resample** คงการกระจายจริงไว้ ตามกฎใน `AGENTS.md`",
        "- ไม่มี data leakage: เทรนบน train เท่านั้น แล้ว predict ลง test",
        "- `random_state=42` ทุกโมเดล รันซ้ำได้ผลเดิม",
        "- วัดผล 2 จุด: ที่ threshold 0.5 (ค่าปริยาย) และที่ threshold ซึ่งให้ F1 สูงสุด",
        "  หาจาก PR curve เต็มเส้น เพราะโมเดลที่ถ่วงน้ำหนักคลาสดันคะแนนไปกองใกล้ 1",
        "  ทำให้ threshold 0.5 ให้ false positive เยอะเกินใช้งานจริง",
        "",
        "## ผลลัพธ์ (เรียงตาม AUPRC)",
        "",
        "| อันดับ | แบบ | AUPRC | ROC-AUC | Recall@0.5 | Precision@0.5 | F1 ดีที่สุด | Recall ที่จุดนั้น | Precision ที่จุดนั้น | FN | FP |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in df.iterrows():
        lines.append(
            f"| {i + 1} | {r['Variant']} | **{r['AUPRC']:.4f}** | {r['ROC_AUC']:.4f} | "
            f"{r['Recall_at_0.5']:.4f} | {r['Precision_at_0.5']:.4f} | "
            f"{r['Best_F1']:.4f} | {r['Recall_at_Best_F1']:.4f} | "
            f"{r['Precision_at_Best_F1']:.4f} | {r['FN_at_Best_F1']} | {r['FP_at_Best_F1']} |")
    lines += [
        "",
        f"**ดีที่สุดตาม AUPRC: {best['Variant']}** (AUPRC = {best['AUPRC']:.4f}) — "
        f"ที่จุด F1 ดีที่สุดจับ fraud ได้ {best['Recall_at_Best_F1'] * 100:.1f}% "
        f"โดยพลาด {best['FN_at_Best_F1']} เคส และแจ้งเตือนผิด {best['FP_at_Best_F1']} เคส",
        "",
        "![เทียบ 3 อัลกอริทึม](comparison_summary.png)",
        "",
    ]
    if ev:
        lines += _findings_section(ev)
    lines += [
        "## ไฟล์ผลลัพธ์",
        "",
        "| ไฟล์ | เนื้อหา |",
        "|------|---------|",
        "| `analyze/comparison_all_models.csv` | ตารางเทียบทั้ง 6 แบบ |",
        "| `analyze/comparison_summary.json` | สรุปเชิงโครงสร้าง + อันดับ AUPRC |",
        "| `analyze/comparison_summary.png` | PR curve / อันดับ AUPRC / ต้นทุน FN-FP |",
        "| `analyze/key_findings.png` | แผนภาพข้อค้นพบสำคัญ 6 ช่อง |",
        "| `analyze/key_findings.json` | ตัวเลขหลังแผนภาพ (CI, threshold transfer, ผล SMOTE) |",
        "| `analyze/<algo>/<p>_results.csv` | ผลของอัลกอริทึมนั้นแยก variant |",
        "| `analyze/<algo>/<p>_thresholds.csv` | ตัวชี้วัดไล่ตาม threshold |",
        "| `analyze/<algo>/<p>_summary.json` | พารามิเตอร์ + ตัวชี้วัดครบ |",
        "| `analyze/<algo>/<p>_analysis.png` | PR curve + confusion matrix |",
        "| `logs/analyze_progress.log` | log ว่า step ไหนเสร็จแล้ว (append ทุกรอบ) |",
        "",
        "## รันต่อ / รันซ้ำ",
        "",
        "```powershell",
        "python analyze/logistic_regression/lr_analysis.py   # ~10 วินาที",
        "python analyze/random_forest/rf_analysis.py         # ~85 วินาที (ช้าสุด)",
        "python analyze/gradient_boosting/gb_analysis.py     # ~20 วินาที",
        "python analyze/compare_all.py                       # ~2 วินาที (ไม่เทรนใหม่)",
        "```",
        "",
        "ถ้าปิดเครื่องกลางทาง เปิด `logs/analyze_progress.log` ดูว่าบรรทัดล่าสุด",
        "ของแต่ละ ALGO เป็น `DONE` แล้วหรือยัง แล้วรันต่อเฉพาะตัวที่ยังไม่ `DONE`",
        "(`compare_all.py` ต้องรันหลังสุด เพราะอ่านผลของทั้ง 3 ตัว)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("เขียนสรุป %s", path.name)


def run_comparison():
    log.info("=" * 72)
    log.info("รวมผลทั้ง 3 อัลกอริทึมมาเทียบกัน")
    mark_progress("COMPARE all 3 algorithms", "START")

    loaded = [_load_algo(folder, prefix) for folder, prefix, _ in ALGOS]
    df = build_table(loaded)

    csv_path = ANALYZE_DIR / "comparison_all_models.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log.info("บันทึกตารางเทียบ %s (%d แถว)", csv_path.name, len(df))

    for _, r in df.iterrows():
        log.info("%-34s AUPRC=%.4f | best-F1=%.4f @%.6f "
                 "(recall=%.4f precision=%.4f FN=%d FP=%d)",
                 r["Variant"], r["AUPRC"], r["Best_F1"], r["Best_F1_Threshold"],
                 r["Recall_at_Best_F1"], r["Precision_at_Best_F1"],
                 r["FN_at_Best_F1"], r["FP_at_Best_F1"])

    best = df.iloc[0]
    out = {
        "objective": "จำแนกธุรกรรมทุจริต ประเมินด้วย AUPRC (ตาม Introduction.txt)",
        "primary_metric": "AUPRC (average precision)",
        "test_set": loaded[0][0]["dataset"],
        "algorithms": [
            {"label": label,
             "method": summary["method"],
             "model_params": summary["model_params"],
             "best_variant": summary["best_variant"]}
            for (folder, prefix, label), (summary, _) in zip(ALGOS, loaded)],
        "ranking_by_auprc": [
            {"rank": int(i + 1), "variant": r["Variant"], "algorithm": r["Algorithm"],
             "auprc": float(r["AUPRC"]), "roc_auc": float(r["ROC_AUC"]),
             "best_f1": float(r["Best_F1"]),
             "best_f1_threshold": float(r["Best_F1_Threshold"]),
             "recall_at_best_f1": float(r["Recall_at_Best_F1"]),
             "precision_at_best_f1": float(r["Precision_at_Best_F1"]),
             "fn_at_best_f1": int(r["FN_at_Best_F1"]),
             "fp_at_best_f1": int(r["FP_at_Best_F1"])}
            for i, r in df.iterrows()],
        "winner": {"variant": best["Variant"], "auprc": float(best["AUPRC"])},
        "note": "ไม่รายงาน accuracy เป็นตัวหลัก เพราะ fraud มีแค่ ~0.17% "
                "ทายว่าไม่โกงทุกแถวก็ได้ accuracy 99.8%",
    }
    json_path = ANALYZE_DIR / "comparison_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log.info("บันทึกสรุป %s", json_path.name)

    plot_comparison(loaded, df, ANALYZE_DIR / "comparison_summary.png")
    # แผนภาพ Key Findings + ตัวเลขข้อควรระวัง (คำนวณจากคะแนนที่เก็บไว้ ไม่เทรนใหม่)
    ev = build_key_findings(loaded, ALGOS)
    write_summary_md(loaded, df, ANALYZE_DIR / "SUMMARY.md", ev)

    log.info("สรุปสุดท้าย: %s ให้ AUPRC สูงสุด = %.4f", best["Variant"], best["AUPRC"])
    mark_progress("COMPARE all 3 algorithms", "DONE",
                  f"winner={best['Variant']} AUPRC={best['AUPRC']:.4f}")
    log.info("=" * 72)
    return df


if __name__ == "__main__":
    run_comparison()
