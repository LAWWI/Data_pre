"""Key Findings — แผนภาพสรุปข้อค้นพบสำคัญ + ตัวเลขที่ใช้เตือนข้อควรระวัง

อ่านผลที่ 3 อัลกอริทึมเขียนไว้ (ไม่เทรนใหม่) แล้วคำนวณหลักฐานประกอบข้อค้นพบ
  1. accuracy ใช้ตัดสินงานนี้ไม่ได้ — ทายว่า "ไม่โกง" ทุกแถวก็ได้ ~99.83%
  2. ROC-AUC สูงไม่ได้แปลว่าดี — LR มี ROC-AUC สูงกว่า RF แต่ AUPRC ต่ำกว่ามาก
  3. การเลือก threshold สำคัญกว่าการเลือกวิธีแก้ imbalance
  4. SMOTE ไม่ได้ช่วยเสมอ — ช่วย RF เล็กน้อย แต่ทำให้ GB แย่ลงชัดเจน
  5. fraud ใน test มีแค่ 95 เคส → ช่วงความเชื่อมั่นกว้าง อย่าฟันธงจากทศนิยมตัวที่ 3
  6. ทุกจุดตัดคือการแลก: จับ fraud เพิ่ม 1 เคส ต้องยอมแจ้งเตือนผิดเพิ่มกี่เคส

ผลลัพธ์
  analyze/key_findings.png    แผนภาพ 6 ช่อง (ข้อค้นพบละช่อง)
  analyze/key_findings.json   ตัวเลขทั้งหมดที่ใช้ในแผนภาพ + ข้อควรระวัง

รัน: python analyze/key_findings.py   (หรือถูกเรียกอัตโนมัติจาก compare_all.py)
"""
from pathlib import Path
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import average_precision_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ANALYZE_DIR, PALETTE, RANDOM_STATE, log, mark_progress  # noqa: E402

BOOTSTRAP_N = 500       # จำนวนรอบ bootstrap หาช่วงความเชื่อมั่นของ AUPRC
GRID = 200              # ความละเอียดของเส้น F1 เทียบ threshold
CI_LEVEL = 95


# ---------------------------------------------------------------- คำนวณหลักฐาน

def _best_variant(summary):
    return max(summary["variants"], key=lambda v: v["pr_auc"])


def trivial_baseline(y_test):
    """โมเดลโง่ที่สุด: ทายว่าไม่โกงทุกแถว — ใช้โชว์ว่า accuracy หลอกแค่ไหน"""
    n = int(len(y_test))
    fraud = int(np.sum(y_test))
    return {"rows": n, "fraud": fraud,
            "accuracy": round(100 * (n - fraud) / n, 3),
            "recall": 0.0, "fraud_caught": 0}


def bootstrap_auprc(y_test, scores, n=BOOTSTRAP_N, seed=RANDOM_STATE):
    """สุ่มซ้ำแบบ bootstrap บน test เพื่อหาช่วงความเชื่อมั่นของ AUPRC

    ทำเพราะ fraud ใน test มีแค่ 95 เคส — ตัวเลขทศนิยมตัวท้ายแกว่งง่ายมาก
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_test))
    vals = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if y_test[s].sum() == 0:      # ไม่มี fraud เลยในรอบนั้น คำนวณไม่ได้
            continue
        vals.append(average_precision_score(y_test[s], scores[s]))
    lo, hi = np.percentile(vals, [(100 - CI_LEVEL) / 2, 100 - (100 - CI_LEVEL) / 2])
    return {"mean": round(float(np.mean(vals)), 4),
            "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "n_bootstrap": int(len(vals))}


def threshold_transfer(y_test, scores, seed=RANDOM_STATE):
    """เช็กความซื่อสัตย์: เลือก threshold จากครึ่งหนึ่งของ test แล้วไปใช้กับอีกครึ่ง

    ค่า "F1 ดีที่สุด" ที่รายงานไว้ เลือกจุดตัดบน test ชุดเดียวกับที่วัดผล
    ตัวเลขจึงดูดีเกินจริงเล็กน้อย การทดสอบนี้บอกว่าเกินจริงไปเท่าไหร่
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y_test))
    a, b = idx[:len(idx) // 2], idx[len(idx) // 2:]

    grid = np.unique(np.quantile(scores, np.linspace(0.90, 1.0, 400)))
    f1_a = [f1_score(y_test[a], (scores[a] >= t).astype(int), zero_division=0)
            for t in grid]
    t_a = float(grid[int(np.argmax(f1_a))])

    f1_b_transfer = f1_score(y_test[b], (scores[b] >= t_a).astype(int), zero_division=0)
    f1_b_oracle = max(f1_score(y_test[b], (scores[b] >= t).astype(int), zero_division=0)
                      for t in grid)
    return {"threshold_from_half_A": round(t_a, 6),
            "f1_on_half_B": round(float(f1_b_transfer), 4),
            "f1_oracle_on_half_B": round(float(f1_b_oracle), 4),
            "optimism_gap": round(float(f1_b_oracle - f1_b_transfer), 4)}


def f1_curve(y_test, scores, grid_n=GRID):
    """เส้น F1 เทียบ threshold ใช้ดูว่าโมเดลทนต่อการตั้งจุดตัดผิดแค่ไหน"""
    ts = np.linspace(0.001, 0.999, grid_n)
    return ts, np.array([f1_score(y_test, (scores >= t).astype(int), zero_division=0)
                         for t in ts])


def cost_curve(y_test, scores):
    """ไล่ threshold แล้วดูว่า 'จับ fraud ได้กี่เคส' แลกกับ 'เตือนผิดกี่เคส'"""
    order = np.argsort(-scores)
    y_sorted = y_test[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    return fp, tp


def collect(loaded, algos):
    """รวมหลักฐานทุกข้อจากผลที่โหลดมา (คืน dict ที่พร้อมเขียน JSON)"""
    y_test = np.asarray(loaded[0][1]["y_test"]).astype(int)
    out = {"test_rows": int(len(y_test)), "test_fraud": int(y_test.sum()),
           "trivial_baseline": trivial_baseline(y_test), "algorithms": []}

    for (folder, prefix, label), (summary, scores_npz) in zip(algos, loaded):
        best = _best_variant(summary)
        s = np.asarray(scores_npz[best["name"]]).astype(float)
        log.info("คำนวณหลักฐานของ %s (variant=%s)", label, best["name"])
        entry = {
            "label": label, "color_key": prefix, "best_variant": best["name"],
            "pr_auc": best["pr_auc"], "roc_auc": best["roc_auc"],
            "auprc_ci": bootstrap_auprc(y_test, s),
            "threshold_transfer": threshold_transfer(y_test, s),
            "variants": [{"name": v["name"], "train_set": v["train_set"],
                          "pr_auc": v["pr_auc"], "roc_auc": v["roc_auc"],
                          "f1_at_0.5": v["at_threshold_0.5"]["f1"],
                          "best_f1": v["best_f1_threshold"]["f1"],
                          "best_f1_threshold": v["best_f1_threshold"]["threshold"]}
                         for v in summary["variants"]],
        }
        cw = next(v for v in entry["variants"] if "SMOTE" not in v["train_set"])
        sm = next(v for v in entry["variants"] if "SMOTE" in v["train_set"])
        entry["smote_effect"] = {
            "class_weight_auprc": cw["pr_auc"], "smote_auprc": sm["pr_auc"],
            "delta": round(sm["pr_auc"] - cw["pr_auc"], 4)}
        out["algorithms"].append(entry)

    # ข้อค้นพบที่ 2: หาคู่ที่ ROC-AUC กับ AUPRC ให้คำตอบขัดกัน (ดูทุก variant)
    points = [{"label": a["label"], "code": a["color_key"].upper(), "variant": v["name"],
               "roc_auc": v["roc_auc"], "pr_auc": v["pr_auc"]}
              for a in out["algorithms"] for v in a["variants"]]
    pairs = [{"higher_roc_lower_auprc": x, "lower_roc_higher_auprc": y,
              "roc_gap": round(x["roc_auc"] - y["roc_auc"], 4),
              "auprc_gap": round(y["pr_auc"] - x["pr_auc"], 4)}
             for x in points for y in points
             if x["roc_auc"] > y["roc_auc"] and x["pr_auc"] < y["pr_auc"]]
    # เรียงให้คู่ที่ "ขัดกันแรงที่สุด" ขึ้นก่อน เพราะเป็นตัวอย่างที่อธิบายได้ชัดสุด
    pairs.sort(key=lambda p: p["auprc_gap"], reverse=True)
    out["roc_vs_auprc_contradictions"] = pairs[:5]
    return out


# ------------------------------------------------------------------- แผนภาพ

def _tile(ax, title, lines):
    """ช่องตัวเลขล้วน (stat tile) สำหรับเรื่องที่เป็น 'ตัวเลขเดียว' ไม่ใช่กราฟ"""
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=PALETTE["ink"],
                 loc="left", pad=10)
    ax.axis("off")
    y = 0.90
    for big, small, color in lines:
        ax.text(0.02, y, big, fontsize=26, fontweight="bold", color=color,
                va="top", transform=ax.transAxes)
        ax.text(0.02, y - 0.17, small, fontsize=9, color=PALETTE["ink2"],
                va="top", transform=ax.transAxes, wrap=True)
        y -= 0.37
        ax.plot([0.02, 0.98], [y + 0.08, y + 0.08], color=PALETTE["grid"],
                linewidth=1, transform=ax.transAxes, clip_on=False)


def build_figure(ev, loaded, algos, out_path):
    """วาดแผนภาพ 6 ช่อง — หนึ่งช่องคือหนึ่งข้อค้นพบ หัวข้อบอกข้อสรุปตรง ๆ"""
    y_test = np.asarray(loaded[0][1]["y_test"]).astype(int)
    colors = {a["label"]: PALETTE[a["color_key"]] for a in ev["algorithms"]}

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.6), dpi=150)
    fig.patch.set_facecolor(PALETTE["surface"])

    # ── 1. accuracy หลอก — เป็นเรื่องของตัวเลข ไม่ใช่กราฟ จึงใช้ stat tile
    tb = ev["trivial_baseline"]
    winner = max(ev["algorithms"], key=lambda a: a["pr_auc"])
    _tile(axes[0][0], "1. accuracy ใช้ตัดสินงานนี้ไม่ได้", [
        (f"{tb['accuracy']}%",
         f"accuracy ของโมเดลที่ทายว่า \"ไม่โกง\" ทุกแถว\n"
         f"จับ fraud ได้ 0 จาก {tb['fraud']} เคส — ใช้งานจริงไม่ได้เลย",
         PALETTE["fn"]),
        (f"{winner['pr_auc']:.3f}",
         f"AUPRC ของโมเดลที่ดีที่สุด ({winner['label'][3:]})\n"
         f"สูงกว่าเส้นสุ่ม {tb['fraud'] / tb['rows'] * 100:.2f}% อยู่ราว "
         f"{winner['pr_auc'] / (tb['fraud'] / tb['rows']):.0f} เท่า",
         PALETTE["gb"]),
    ])

    # ── 2. ROC-AUC หลอก: scatter ROC-AUC เทียบ AUPRC (รูปทรงแยก variant)
    ax = axes[0][1]
    for a in ev["algorithms"]:
        code = a["color_key"].upper()
        for v in a["variants"]:
            smote = "SMOTE" in v["train_set"]
            ax.scatter(v["roc_auc"], v["pr_auc"], s=170,
                       marker="^" if smote else "o",
                       color=colors[a["label"]], edgecolor=PALETTE["surface"],
                       linewidth=2, zorder=3)
            ax.annotate(f"{code}{' +SMOTE' if smote else ''}",
                        (v["roc_auc"], v["pr_auc"]), xytext=(0, -18),
                        textcoords="offset points", ha="center", fontsize=8.5,
                        color=PALETTE["ink2"])
    ax.set_xlabel("ROC-AUC  (ยิ่งขวายิ่ง 'ดู' ดี)", fontsize=9.5,
                  color=PALETTE["ink2"])
    if ev["roc_vs_auprc_contradictions"]:
        c = ev["roc_vs_auprc_contradictions"][0]
        hi, lo = c["higher_roc_lower_auprc"], c["lower_roc_higher_auprc"]
        ax.annotate("", xy=(lo["roc_auc"], lo["pr_auc"]),
                    xytext=(hi["roc_auc"], hi["pr_auc"]),
                    arrowprops=dict(arrowstyle="->", color=PALETTE["ink2"],
                                    linewidth=1.3, linestyle="--",
                                    shrinkA=12, shrinkB=12))
        note = (f"{hi['code']} ชนะ {lo['code']} ที่ ROC-AUC {c['roc_gap']:+.3f}\n"
                f"แต่แพ้ที่ AUPRC ถึง {c['auprc_gap']:.3f}")
        ax.text(0.03, 0.06, note, transform=ax.transAxes, fontsize=9,
                color=PALETTE["ink"], va="bottom",
                bbox=dict(boxstyle="round,pad=0.45", facecolor="#f6f5f2",
                          edgecolor=PALETTE["grid"]))
    ax.set_ylabel("AUPRC  (ตัวชี้วัดที่ใช้ตัดสินจริง)", fontsize=9.5,
                  color=PALETTE["ink2"])
    ax.set_title("2. ROC-AUC สูงไม่ได้แปลว่าโมเดลดี", fontsize=11.5,
                 fontweight="bold", loc="left", pad=10)
    ax.scatter([], [], marker="o", color=PALETTE["muted"], label="class_weight")
    ax.scatter([], [], marker="^", color=PALETTE["muted"], label="SMOTE")
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    # ── 3. ความทนต่อ threshold: เส้น F1 เทียบจุดตัด (สีคือตัวตนของอัลกอริทึม)
    ax = axes[0][2]
    for (folder, prefix, label), (summary, npz) in zip(algos, loaded):
        best = _best_variant(summary)
        ts, f1s = f1_curve(y_test, np.asarray(npz[best["name"]]).astype(float))
        ax.plot(ts, f1s, linewidth=2, color=PALETTE[prefix], label=label[3:])
        peak = int(np.argmax(f1s))
        ax.scatter([ts[peak]], [f1s[peak]], s=55, color=PALETTE[prefix],
                   edgecolor=PALETTE["surface"], linewidth=2, zorder=4)
    ax.axvline(0.5, color=PALETTE["muted"], linestyle="--", linewidth=1.2)
    ax.text(0.5, 0.02, " ค่าปริยาย 0.5", fontsize=8.5, color=PALETTE["ink2"],
            rotation=90, va="bottom")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("threshold ที่ใช้ตัดสินว่าโกง", fontsize=9.5, color=PALETTE["ink2"])
    ax.set_ylabel("F1", fontsize=9.5, color=PALETTE["ink2"])
    ax.set_title("3. ตั้งจุดตัดผิด โมเดลดีก็พัง", fontsize=11.5,
                 fontweight="bold", loc="left", pad=10)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")

    # ── 4. ผลของ SMOTE ต่อ AUPRC (แท่งทึบ = class_weight, แท่งลาย = SMOTE)
    ax = axes[1][0]
    x = np.arange(len(ev["algorithms"])); w = 0.36
    for i, a in enumerate(ev["algorithms"]):
        c = colors[a["label"]]
        se = a["smote_effect"]
        ax.bar(x[i] - w / 2 - 0.01, se["class_weight_auprc"], w, color=c,
               edgecolor=PALETTE["surface"], linewidth=2)
        ax.bar(x[i] + w / 2 + 0.01, se["smote_auprc"], w, color=c, alpha=0.45,
               hatch="///", edgecolor=PALETTE["surface"], linewidth=2)
        ax.text(x[i] - w / 2, se["class_weight_auprc"] + 0.012,
                f"{se['class_weight_auprc']:.3f}", ha="center", fontsize=8.5)
        ax.text(x[i] + w / 2, se["smote_auprc"] + 0.012,
                f"{se['smote_auprc']:.3f}", ha="center", fontsize=8.5)
        # delta ที่เล็กกว่า 0.005 ถือว่าเท่ากัน (อยู่ในช่วงที่ข้อมูลแยกไม่ออก)
        dcolor = (PALETTE["muted"] if abs(se["delta"]) < 0.005
                  else PALETTE["fn"] if se["delta"] < 0 else "#006300")
        dtext = "เท่าเดิม" if abs(se["delta"]) < 0.005 else f"{se['delta']:+.3f}"
        ax.text(x[i], max(se["class_weight_auprc"], se["smote_auprc"]) + 0.075,
                dtext, ha="center", fontsize=10, fontweight="bold", color=dcolor)
    ax.set_xticks(x)
    ax.set_xticklabels([a["label"][3:] for a in ev["algorithms"]], fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("AUPRC", fontsize=9.5, color=PALETTE["ink2"])
    ax.set_title("4. SMOTE ไม่ได้ช่วยเสมอ", fontsize=11.5, fontweight="bold",
                 loc="left", pad=10)
    ax.legend(handles=[
        Patch(facecolor=PALETTE["muted"], label="ถ่วงน้ำหนักคลาส"),
        Patch(facecolor=PALETTE["muted"], alpha=0.45, hatch="///", label="SMOTE")],
        fontsize=8.5, frameon=False, loc="upper right", ncol=2)

    # ── 5. ช่วงความเชื่อมั่นของ AUPRC (fraud ใน test มีแค่ 95 เคส)
    ax = axes[1][1]
    order = sorted(ev["algorithms"], key=lambda a: a["pr_auc"])
    for i, a in enumerate(order):
        ci = a["auprc_ci"]
        ax.plot([ci["lo"], ci["hi"]], [i, i], linewidth=3.5,
                color=colors[a["label"]], alpha=0.42, solid_capstyle="round")
        ax.scatter([a["pr_auc"]], [i], s=120, color=colors[a["label"]],
                   edgecolor=PALETTE["surface"], linewidth=2, zorder=3)
        ax.text(ci["hi"] + 0.012, i, f"{a['pr_auc']:.3f}  [{ci['lo']:.3f}-{ci['hi']:.3f}]",
                va="center", fontsize=8.5, color=PALETTE["ink2"])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([a["label"][3:] for a in order], fontsize=9)
    ax.set_xlim(0.4, 1.12)
    ax.set_xlabel(f"AUPRC พร้อมช่วงความเชื่อมั่น {CI_LEVEL}% "
                  f"(bootstrap {BOOTSTRAP_N} รอบ)", fontsize=9.5,
                  color=PALETTE["ink2"])
    ax.set_title(f"5. fraud ใน test มีแค่ {ev['test_fraud']} เคส → ตัวเลขแกว่งกว้าง",
                 fontsize=11.5, fontweight="bold", loc="left", pad=10)

    # ── 6. ราคาของการจับเพิ่ม: เตือนผิดกี่เคส ต่อ fraud ที่จับได้เพิ่ม
    ax = axes[1][2]
    for (folder, prefix, label), (summary, npz) in zip(algos, loaded):
        best = _best_variant(summary)
        fp, tp = cost_curve(y_test, np.asarray(npz[best["name"]]).astype(float))
        ax.step(fp, tp, where="post", linewidth=2, color=PALETTE[prefix],
                label=label[3:])
    ax.axhline(ev["test_fraud"], color=PALETTE["muted"], linestyle=":", linewidth=1.2)
    ax.text(1.2, ev["test_fraud"] - 3, f"fraud ทั้งหมด {ev['test_fraud']} เคส",
            fontsize=8.5, color=PALETTE["ink2"], va="top")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(0, 2000); ax.set_ylim(0, ev["test_fraud"] * 1.12)
    ax.set_xlabel("จำนวนที่แจ้งเตือนผิด (FP) — สเกล log", fontsize=9.5,
                  color=PALETTE["ink2"])
    ax.set_ylabel("จับ fraud ได้ (TP)", fontsize=9.5, color=PALETTE["ink2"])
    ax.set_title("6. จับเพิ่มอีกเคส ต้องยอมเตือนผิดเพิ่มเท่าไหร่", fontsize=11.5,
                 fontweight="bold", loc="left", pad=10)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")

    for row in axes:
        for a in row:
            a.set_facecolor(PALETTE["surface"])
            a.grid(axis="both", linestyle=":", linewidth=0.8,
                   color=PALETTE["grid"], alpha=0.9)
            a.set_axisbelow(True)
            a.tick_params(colors=PALETTE["ink2"], labelsize=8.5)
            for sp in ("left", "bottom"):
                a.spines[sp].set_color(PALETTE["axis"])
    axes[0][0].grid(False)

    fig.suptitle("Key Findings — วิเคราะห์การตรวจจับธุรกรรมทุจริตด้วย 3 อัลกอริทึม",
                 fontsize=15, fontweight="bold", y=1.0, color=PALETTE["ink"])
    fig.text(0.5, 0.965, f"วัดบน test set ที่ไม่ถูก resample "
                         f"({ev['test_rows']:,} แถว · fraud {ev['test_fraud']} เคส "
                         f"= {100 * ev['test_fraud'] / ev['test_rows']:.3f}%)",
             ha="center", fontsize=10, color=PALETTE["ink2"])
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out_path, bbox_inches="tight", facecolor=PALETTE["surface"])
    plt.close(fig)
    log.info("บันทึกแผนภาพ %s", Path(out_path).name)


def build_key_findings(loaded, algos):
    """คำนวณหลักฐาน + วาดแผนภาพ + เขียน JSON (ถูกเรียกจาก compare_all.py ด้วย)"""
    log.info("สร้าง Key Findings จากผลที่มีอยู่ (ไม่เทรนใหม่)")
    ev = collect(loaded, algos)
    build_figure(ev, loaded, algos, ANALYZE_DIR / "key_findings.png")
    with open(ANALYZE_DIR / "key_findings.json", "w", encoding="utf-8") as f:
        json.dump(ev, f, indent=2, ensure_ascii=False)
    log.info("บันทึก key_findings.json")
    return ev


if __name__ == "__main__":
    from compare_all import ALGOS, _load_algo

    mark_progress("KEY FINDINGS", "START")
    _loaded = [_load_algo(folder, prefix) for folder, prefix, _ in ALGOS]
    _ev = build_key_findings(_loaded, ALGOS)
    mark_progress("KEY FINDINGS", "DONE",
                  f"auprc_ci_width={_ev['algorithms'][0]['auprc_ci']['hi'] - _ev['algorithms'][0]['auprc_ci']['lo']:.3f}")
