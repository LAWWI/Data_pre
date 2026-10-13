"""สร้างกราฟประกอบรายงาน (คืนค่าเป็น base64 PNG สำหรับฝังใน HTML)"""
import base64, io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .logger import get_logger

log = get_logger("charts")
C = {"bg": "#ffffff", "a": "#2563eb", "b": "#dc2626", "g": "#6b7280"}
plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def class_balance(before, after):
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4))
    for i, (d, t) in enumerate([(before, "Train: before SMOTE"),
                                (after, "Train: after SMOTE")]):
        ax[i].bar(["Normal (0)", "Fraud (1)"], [d.get(0, 0), d.get(1, 0)],
                  color=[C["g"], C["b"]], width=.55)
        ax[i].set_title(t)
        for j, v in enumerate([d.get(0, 0), d.get(1, 0)]):
            ax[i].text(j, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
        ax[i].set_ylim(0, max(d.values()) * 1.18)
        ax[i].set_ylabel("rows" if i == 0 else "")
    return _b64(fig)


def amount_time(raw_amount, log_amount, scaled_amount, time_raw):
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.1))
    ax[0].hist(raw_amount, bins=60, color=C["a"])
    ax[0].set_title("Amount (raw)"); ax[0].set_yscale("log")
    ax[1].hist(log_amount, bins=60, color=C["a"])
    ax[1].set_title("log1p(Amount)")
    ax[2].hist(scaled_amount, bins=60, color=C["b"])
    ax[2].set_title("Amount_scaled (Robust)")
    for a in ax: a.set_ylabel("")
    fig.tight_layout()
    img1 = _b64(fig)

    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    hours = (time_raw / 3600) % 24
    ax.hist(hours, bins=48, color=C["a"])
    ax.set_title("จำนวนธุรกรรมตามชั่วโมงในวัน (ที่มาของ Hour_sin/Hour_cos)")
    ax.set_xlabel("hour of day"); ax.set_xlim(0, 24)
    return img1, _b64(fig)


def metrics_chart(res):
    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    labels = [r["name"] for r in res]
    x = np.arange(len(labels)); w = .26
    for i, (k, c, lb) in enumerate([("recall", C["b"], "Recall"),
                                    ("precision", C["a"], "Precision"),
                                    ("pr_auc", C["g"], "PR-AUC")]):
        vals = [r[k] for r in res]
        ax.bar(x + (i - 1) * w, vals, w, label=lb, color=c)
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.12); ax.legend(ncol=3, frameon=False, fontsize=8)
    ax.set_title("ผลเทียบวิธีจัดการ imbalance (บน test set จริง)")
    return _b64(fig)
