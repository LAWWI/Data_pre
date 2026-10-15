"""ALGO 2/3 — Random Forest (bagging ensemble)

ทำไมเลือกตัวนี้: เป็น non-linear ensemble ที่จับ interaction ระหว่างฟีเจอร์ได้
และทนต่อ imbalance ได้ดีกว่าโมเดลเชิงเส้นเมื่อถ่วงน้ำหนักคลาส
เทียบ 2 วิธีจัดการ imbalance บนข้อมูลชุดเดียวกัน
  1. class_weight='balanced_subsample' — ถ่วงน้ำหนักคลาสในแต่ละ bootstrap
  2. SMOTE train                        — ใช้ X_train_smote ที่ step_smote สังเคราะห์ไว้

วัดผลบน test set ที่คงการกระจายจริงไว้ (fraud ~0.17%) ตัวชี้วัดหลักคือ AUPRC

รัน: python analyze/random_forest/rf_analysis.py
"""
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (RANDOM_STATE, PALETTE, evaluate, load_data, log,  # noqa: E402
                    mark_progress, plot_algorithm, write_results)

OUT_DIR = Path(__file__).resolve().parent
PREFIX = "rf"
N_ESTIMATORS = 200
MAX_DEPTH = 16
MIN_SAMPLES_LEAF = 2


def run_rf_analysis():
    log.info("=" * 72)
    log.info("ALGO 2/3 Random Forest — เริ่มวิเคราะห์")
    mark_progress("ALGO 2 random_forest", "START")

    d = load_data()
    base = dict(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                random_state=RANDOM_STATE, n_jobs=-1)

    v1, s1, m1 = evaluate(
        "RF class_weight=balanced_subsample",
        RandomForestClassifier(class_weight="balanced_subsample", **base),
        d["X_train"], d["y_train"], d["X_test"], d["y_test"], "train (ของจริง)")
    v2, s2, _ = evaluate(
        "RF + SMOTE train",
        RandomForestClassifier(**base),
        d["X_smote"], d["y_smote"], d["X_test"], d["y_test"], "train + SMOTE")

    variants = [v1, v2]
    score_map = {v1["name"]: s1, v2["name"]: s2}

    # feature importance ของ variant แรก เอาไว้เทียบกับผลใน feature/random_forest
    imp = m1.feature_importances_
    order = np.argsort(imp)[::-1][:10]
    top_imp = [{"rank": int(r + 1), "feature": d["features"][i],
                "importance_pct": round(float(imp[i] * 100), 2)}
               for r, i in enumerate(order)]
    log.info("ฟีเจอร์สำคัญสุด 10 ตัว: %s",
             ", ".join(f"{t['feature']}({t['importance_pct']}%)" for t in top_imp))

    write_results(
        OUT_DIR, PREFIX,
        method="Random Forest (bagging ensemble of decision trees)",
        model_params={"n_estimators": N_ESTIMATORS, "max_depth": MAX_DEPTH,
                      "min_samples_leaf": MIN_SAMPLES_LEAF,
                      "class_weight": "balanced_subsample (variant 1) / None (variant 2)"},
        variants=variants, y_test=d["y_test"], score_map=score_map,
        extra={"top_10_importance": top_imp,
               "cross_check": "เทียบอันดับฟีเจอร์กับ feature/random_forest/"
                              "rf_importance_results.csv ได้ (คนละ hyperparameter "
                              "แต่อันดับควรใกล้กัน)",
               "interpretation": "จับ interaction ได้ ให้ precision สูงกว่าโมเดลเชิงเส้น "
                                 "ที่ recall ระดับเดียวกัน แต่เทรนนานกว่า"})

    plot_algorithm(OUT_DIR, PREFIX, "ALGO 2: Random Forest",
                   variants, d["y_test"], score_map, PALETTE["rf"])

    best = max(variants, key=lambda v: v["pr_auc"])
    log.info("สรุป ALGO 2: variant ที่ AUPRC ดีสุดคือ %s (AUPRC=%.4f)",
             best["name"], best["pr_auc"])
    mark_progress("ALGO 2 random_forest", "DONE",
                  f"best={best['name']} AUPRC={best['pr_auc']}")
    log.info("=" * 72)
    return variants


if __name__ == "__main__":
    run_rf_analysis()
