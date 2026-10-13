"""เปรียบเทียบวิธีจัดการ imbalance: ไม่ทำอะไร vs class_weight=balanced vs SMOTE
ใช้ LogisticRegression เป็น baseline เพื่อยืนยันว่า pre-processing ใช้งานได้
"""
import time, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix)
from .config import RANDOM_STATE
from .logger import get_logger

log = get_logger("modeling")


def _eval(name, model, Xtr, ytr, Xte, yte):
    t0 = time.time()
    model.fit(Xtr, ytr)
    p = model.predict(Xte)
    s = model.predict_proba(Xte)[:, 1]
    tn, fp, fn, tp = confusion_matrix(yte, p).ravel()
    m = {
        "name": name,
        "precision": round(precision_score(yte, p, zero_division=0), 4),
        "recall": round(recall_score(yte, p, zero_division=0), 4),
        "f1": round(f1_score(yte, p, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(yte, s), 4),
        "pr_auc": round(average_precision_score(yte, s), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "fit_sec": round(time.time() - t0, 1),
    }
    log.info("%-22s recall=%.4f precision=%.4f f1=%.4f PR-AUC=%.4f "
             "(TP=%d FN=%d FP=%d) ใช้เวลา %.1fs",
             name, m["recall"], m["precision"], m["f1"], m["pr_auc"],
             tp, fn, fp, m["fit_sec"])
    return m


def compare(Xtr, ytr, Xte, yte, Xb, yb):
    log.info("STEP 9 เทรน baseline 3 แบบเพื่อเทียบผล")
    base = dict(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
    res = [
        _eval("1) ไม่แก้ imbalance", LogisticRegression(**base), Xtr, ytr, Xte, yte),
        _eval("2) class_weight=balanced",
              LogisticRegression(class_weight="balanced", **base), Xtr, ytr, Xte, yte),
        _eval("3) SMOTE", LogisticRegression(**base), Xb, yb, Xte, yte),
    ]
    best = max(res, key=lambda r: r["pr_auc"])
    log.info("สรุป: วิธีที่ให้ PR-AUC สูงสุดคือ %s (PR-AUC=%.4f)",
             best["name"], best["pr_auc"])
    return res, best["name"]
