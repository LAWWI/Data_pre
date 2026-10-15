"""ALGO 3/3 — Histogram-based Gradient Boosting (boosting ensemble)

ทำไมเลือกตัวนี้: boosting สร้างต้นไม้ต่อ ๆ กันเพื่อแก้ความผิดพลาดของต้นก่อนหน้า
มักให้ AUPRC สูงสุดในงาน fraud ที่ข้อมูล imbalance จัด
ใช้ HistGradientBoostingClassifier ของ scikit-learn (ไม่ต้องลง xgboost เพิ่ม)
เทียบ 2 วิธีจัดการ imbalance บนข้อมูลชุดเดียวกัน
  1. class_weight='balanced' — ถ่วงน้ำหนักคลาสใน loss
  2. SMOTE train             — ใช้ X_train_smote ที่ step_smote สังเคราะห์ไว้

วัดผลบน test set ที่คงการกระจายจริงไว้ (fraud ~0.17%) ตัวชี้วัดหลักคือ AUPRC

รัน: python analyze/gradient_boosting/gb_analysis.py
"""
from pathlib import Path
import sys

from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (RANDOM_STATE, PALETTE, evaluate, load_data, log,  # noqa: E402
                    mark_progress, plot_algorithm, write_results)

OUT_DIR = Path(__file__).resolve().parent
PREFIX = "gb"
MAX_ITER = 300
LEARNING_RATE = 0.1
MAX_LEAF_NODES = 31
# ปิด early stopping เพื่อให้ผลเท่ากันทุกครั้ง (early stopping แบ่ง validation เอง)
EARLY_STOPPING = False


def run_gb_analysis():
    log.info("=" * 72)
    log.info("ALGO 3/3 Histogram Gradient Boosting — เริ่มวิเคราะห์")
    mark_progress("ALGO 3 gradient_boosting", "START")

    d = load_data()
    base = dict(max_iter=MAX_ITER, learning_rate=LEARNING_RATE,
                max_leaf_nodes=MAX_LEAF_NODES, early_stopping=EARLY_STOPPING,
                random_state=RANDOM_STATE)

    v1, s1, _ = evaluate(
        "GB class_weight=balanced",
        HistGradientBoostingClassifier(class_weight="balanced", **base),
        d["X_train"], d["y_train"], d["X_test"], d["y_test"], "train (ของจริง)")
    v2, s2, _ = evaluate(
        "GB + SMOTE train",
        HistGradientBoostingClassifier(**base),
        d["X_smote"], d["y_smote"], d["X_test"], d["y_test"], "train + SMOTE")

    variants = [v1, v2]
    score_map = {v1["name"]: s1, v2["name"]: s2}

    write_results(
        OUT_DIR, PREFIX,
        method="Histogram-based Gradient Boosting (boosting ensemble)",
        model_params={"max_iter": MAX_ITER, "learning_rate": LEARNING_RATE,
                      "max_leaf_nodes": MAX_LEAF_NODES,
                      "early_stopping": EARLY_STOPPING,
                      "class_weight": "balanced (variant 1) / None (variant 2)"},
        variants=variants, y_test=d["y_test"], score_map=score_map,
        extra={"interpretation": "boosting แก้ความผิดพลาดของต้นไม้ก่อนหน้าทีละรอบ "
                                 "จึงไวต่อ fraud ที่หายาก แต่ถ้าเพิ่ม max_iter มากไป "
                                 "อาจ overfit ข้อมูลสังเคราะห์จาก SMOTE"})

    plot_algorithm(OUT_DIR, PREFIX, "ALGO 3: Histogram Gradient Boosting",
                   variants, d["y_test"], score_map, PALETTE["gb"])

    best = max(variants, key=lambda v: v["pr_auc"])
    log.info("สรุป ALGO 3: variant ที่ AUPRC ดีสุดคือ %s (AUPRC=%.4f)",
             best["name"], best["pr_auc"])
    mark_progress("ALGO 3 gradient_boosting", "DONE",
                  f"best={best['name']} AUPRC={best['pr_auc']}")
    log.info("=" * 72)
    return variants


if __name__ == "__main__":
    run_gb_analysis()
