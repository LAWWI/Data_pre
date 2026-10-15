"""ALGO 1/3 — Logistic Regression (linear model)

ทำไมเลือกตัวนี้: เป็น baseline เชิงเส้นที่ตีความได้ว่าฟีเจอร์ไหนดันให้เป็น fraud
เทียบ 2 วิธีจัดการ imbalance บนข้อมูลชุดเดียวกัน
  1. class_weight='balanced'  — ถ่วงน้ำหนักคลาสในฟังก์ชัน loss (ไม่สร้างข้อมูลใหม่)
  2. SMOTE train              — ใช้ X_train_smote ที่ step_smote สังเคราะห์ไว้

วัดผลบน test set ที่คงการกระจายจริงไว้ (fraud ~0.17%) ตัวชี้วัดหลักคือ AUPRC

รัน: python analyze/logistic_regression/lr_analysis.py
"""
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (RANDOM_STATE, PALETTE, evaluate, load_data, log,  # noqa: E402
                    mark_progress, plot_algorithm, write_results)

OUT_DIR = Path(__file__).resolve().parent
PREFIX = "lr"
MAX_ITER = 2000


def run_lr_analysis():
    log.info("=" * 72)
    log.info("ALGO 1/3 Logistic Regression — เริ่มวิเคราะห์")
    mark_progress("ALGO 1 logistic_regression", "START")

    d = load_data()
    base = dict(max_iter=MAX_ITER, random_state=RANDOM_STATE)

    v1, s1, m1 = evaluate(
        "LR class_weight=balanced",
        LogisticRegression(class_weight="balanced", **base),
        d["X_train"], d["y_train"], d["X_test"], d["y_test"], "train (ของจริง)")
    v2, s2, _ = evaluate(
        "LR + SMOTE train",
        LogisticRegression(**base),
        d["X_smote"], d["y_smote"], d["X_test"], d["y_test"], "train + SMOTE")

    variants = [v1, v2]
    score_map = {v1["name"]: s1, v2["name"]: s2}

    # ฟีเจอร์ที่มีน้ำหนักมากสุดของ variant แรก (ใช้ตีความทิศทางของสัญญาณ)
    coefs = m1.coef_[0]
    order = np.argsort(np.abs(coefs))[::-1][:10]
    top_coefs = [{"feature": d["features"][i],
                  "coefficient": round(float(coefs[i]), 4),
                  "direction": "Positive" if coefs[i] > 0 else "Negative"}
                 for i in order]
    log.info("ฟีเจอร์ที่มีน้ำหนักสูงสุด 10 ตัว: %s",
             ", ".join(f"{c['feature']}({c['coefficient']:+.3f})" for c in top_coefs))

    write_results(
        OUT_DIR, PREFIX,
        method="Logistic Regression (linear, L2 penalty)",
        model_params={"max_iter": MAX_ITER, "penalty": "l2", "solver": "lbfgs",
                      "class_weight": "balanced (variant 1) / None (variant 2)"},
        variants=variants, y_test=d["y_test"], score_map=score_map,
        extra={"top_10_coefficients": top_coefs,
               "interpretation": "โมเดลเชิงเส้น ตีความง่าย แต่จับ interaction "
                                 "ระหว่างฟีเจอร์ไม่ได้ จึงมักแพ้ ensemble ในงาน fraud"})

    plot_algorithm(OUT_DIR, PREFIX, "ALGO 1: Logistic Regression",
                   variants, d["y_test"], score_map, PALETTE["lr"])

    best = max(variants, key=lambda v: v["pr_auc"])
    log.info("สรุป ALGO 1: variant ที่ AUPRC ดีสุดคือ %s (AUPRC=%.4f)",
             best["name"], best["pr_auc"])
    mark_progress("ALGO 1 logistic_regression", "DONE",
                  f"best={best['name']} AUPRC={best['pr_auc']}")
    log.info("=" * 72)
    return variants


if __name__ == "__main__":
    run_lr_analysis()
