"""ขั้นตอน pre-processing ของ dataset creditcard
ลำดับ: load -> inspect -> clean -> feature/scale -> split -> balance(SMOTE)
"""
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE

from .config import (DATA_CSV, OUT_DIR, TARGET, RANDOM_STATE, TEST_SIZE,
                     SMOTE_K, SMOTE_RATIO)
from .logger import get_logger

log = get_logger("preprocess")


def step_load():
    log.info("STEP 1 โหลดข้อมูลจาก %s", DATA_CSV.name)
    df = pd.read_csv(DATA_CSV)
    log.info("โหลดสำเร็จ shape=%s | memory=%.1f MB",
             df.shape, df.memory_usage(deep=True).sum() / 1e6)
    return df


def step_inspect(df):
    log.info("STEP 2 ตรวจสอบคุณภาพข้อมูล")
    info = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "missing_total": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).value_counts().to_dict(),
        "class_counts": {int(k): int(v) for k, v in df[TARGET].value_counts().items()},
    }
    info["fraud_pct"] = round(100 * info["class_counts"].get(1, 0) / info["rows"], 4)
    info["imbalance_ratio"] = round(
        info["class_counts"].get(0, 0) / max(info["class_counts"].get(1, 1), 1), 1)
    log.info("missing=%d | duplicate rows=%d", info["missing_total"], info["duplicates"])
    log.info("Class 0=%d, Class 1=%d (fraud %.4f%%) -> อัตราส่วน 1:%s",
             info["class_counts"].get(0, 0), info["class_counts"].get(1, 0),
             info["fraud_pct"], info["imbalance_ratio"])
    for c in ("Time", "Amount"):
        s = df[c]
        log.info("%s: min=%.2f max=%.2f mean=%.2f std=%.2f skew=%.2f",
                 c, s.min(), s.max(), s.mean(), s.std(), s.skew())
    info["amount_skew"] = round(float(df["Amount"].skew()), 3)
    info["time_range_h"] = round(float(df["Time"].max() / 3600), 1)
    return info


def step_clean(df):
    log.info("STEP 3 ทำความสะอาดข้อมูล")
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped = n0 - len(df)
    log.info("ลบแถวซ้ำ %d แถว (%.3f%%) -> เหลือ %d แถว",
             dropped, 100 * dropped / n0, len(df))
    neg = int((df["Amount"] < 0).sum())
    zero = int((df["Amount"] == 0).sum())
    log.info("Amount ติดลบ=%d, Amount=0 จำนวน=%d (คงไว้ ไม่ใช่ค่าผิดปกติเชิง logic)",
             neg, zero)
    return df, {"dup_dropped": dropped, "rows_after": int(len(df)),
                "amount_zero": zero, "amount_negative": neg}


def step_features(df):
    """แยกฟีเจอร์เวลาออกมาเป็นรายชั่วโมงแบบ cyclic ก่อน scale"""
    log.info("STEP 4 สร้างฟีเจอร์จาก Time")
    hour = (df["Time"] / 3600.0) % 24
    df["Hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * hour / 24)
    log.info("เพิ่ม Hour_sin/Hour_cos (cyclic encoding ของชั่วโมงในวัน)")
    return df, {"added": ["Hour_sin", "Hour_cos"]}


def step_split(df):
    log.info("STEP 5 แบ่ง train/test แบบ stratified (test=%.0f%%)", TEST_SIZE * 100)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    log.info("train=%s (fraud=%d) | test=%s (fraud=%d)",
             Xtr.shape, int(ytr.sum()), Xte.shape, int(yte.sum()))
    return Xtr, Xte, ytr, yte, {
        "train_rows": int(len(Xtr)), "test_rows": int(len(Xte)),
        "train_fraud": int(ytr.sum()), "test_fraud": int(yte.sum())}


def step_scale(Xtr, Xte):
    """fit scaler บน train เท่านั้น เพื่อไม่ให้เกิด data leakage
    - Amount: log1p แก้ skew แล้วใช้ RobustScaler (ทนค่า outlier)
    - Time  : StandardScaler
    """
    log.info("STEP 6 ปรับ scale คอลัมน์ Time และ Amount")
    Xtr, Xte = Xtr.copy(), Xte.copy()

    Xtr["Amount_log"] = np.log1p(Xtr["Amount"])
    Xte["Amount_log"] = np.log1p(Xte["Amount"])
    log.info("Amount: ทำ log1p ลด skew %.2f -> %.2f",
             Xtr["Amount"].skew(), Xtr["Amount_log"].skew())

    rs = RobustScaler().fit(Xtr[["Amount_log"]])
    ss = StandardScaler().fit(Xtr[["Time"]])
    for X in (Xtr, Xte):
        X["Amount_scaled"] = rs.transform(X[["Amount_log"]])
        X["Time_scaled"] = ss.transform(X[["Time"]])
    log.info("Amount -> RobustScaler (center=%.4f, scale=%.4f)",
             rs.center_[0], rs.scale_[0])
    log.info("Time   -> StandardScaler (mean=%.2f, std=%.2f)",
             ss.mean_[0], float(np.sqrt(ss.var_[0])))

    drop = ["Time", "Amount", "Amount_log"]
    Xtr = Xtr.drop(columns=drop)
    Xte = Xte.drop(columns=drop)
    log.info("ตัดคอลัมน์ดิบออก %s | V1-V28 ไม่ scale ซ้ำ (เป็นผล PCA อยู่แล้ว)", drop)
    log.info("ฟีเจอร์สุดท้าย %d คอลัมน์", Xtr.shape[1])

    joblib.dump({"amount_robust": rs, "time_standard": ss},
                OUT_DIR / "scalers.joblib")
    log.info("บันทึก scaler -> outputs/scalers.joblib")
    return Xtr, Xte, {
        "features": int(Xtr.shape[1]),
        "amount_center": round(float(rs.center_[0]), 4),
        "amount_scale": round(float(rs.scale_[0]), 4),
        "time_mean": round(float(ss.mean_[0]), 2),
        "time_std": round(float(np.sqrt(ss.var_[0])), 2),
        "dropped": drop, "feature_names": list(Xtr.columns)}


def step_smote(Xtr, ytr):
    """ทำ SMOTE บน train เท่านั้น (test ต้องคงการกระจายจริงไว้)"""
    log.info("STEP 7 แก้ class imbalance ด้วย SMOTE (k=%d, ratio=%.2f)",
             SMOTE_K, SMOTE_RATIO)
    before = ytr.value_counts().to_dict()
    sm = SMOTE(sampling_strategy=SMOTE_RATIO, k_neighbors=SMOTE_K,
               random_state=RANDOM_STATE)
    Xb, yb = sm.fit_resample(Xtr, ytr)
    after = pd.Series(yb).value_counts().to_dict()
    log.info("ก่อน SMOTE: 0=%d, 1=%d | หลัง SMOTE: 0=%d, 1=%d (สังเคราะห์เพิ่ม %d แถว)",
             before.get(0, 0), before.get(1, 0), after.get(0, 0), after.get(1, 0),
             after.get(1, 0) - before.get(1, 0))
    return Xb, yb, {
        "before": {int(k): int(v) for k, v in before.items()},
        "after": {int(k): int(v) for k, v in after.items()},
        "synthetic": int(after.get(1, 0) - before.get(1, 0)),
        "k_neighbors": SMOTE_K, "sampling_strategy": SMOTE_RATIO}


def step_save(Xtr, ytr, Xte, yte, Xb, yb):
    log.info("STEP 8 บันทึกชุดข้อมูลที่ประมวลผลแล้ว")
    np.savez_compressed(
        OUT_DIR / "processed.npz",
        X_train=Xtr.values.astype(np.float32), y_train=ytr.values,
        X_test=Xte.values.astype(np.float32), y_test=yte.values,
        X_train_smote=Xb.values.astype(np.float32), y_train_smote=np.asarray(yb),
        features=np.array(list(Xtr.columns)))
    size = (OUT_DIR / "processed.npz").stat().st_size / 1e6
    log.info("บันทึก outputs/processed.npz (%.1f MB)", size)
    Xte.assign(Class=yte.values).head(2000).to_csv(
        OUT_DIR / "sample_test_2000.csv", index=False, encoding="utf-8-sig")
    log.info("บันทึกตัวอย่าง outputs/sample_test_2000.csv (2000 แถว)")
    return {"npz_mb": round(size, 1)}
