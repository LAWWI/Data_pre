"""Agent: pre-processing pipeline สำหรับ dataset ตรวจจับการโกงบัตรเครดิต

ลำดับงานของ agent
  1 load -> 2 inspect -> 3 clean -> 4 feature(time) -> 5 split
  6 scale(Time, Amount) -> 7 SMOTE / balanced -> 8 save
  9 ตรวจสอบด้วยโมเดล -> 10 ออกรายงาน PDF + log

รัน: python main.py
"""
import time, numpy as np
from agent import preprocess as P, modeling, charts, report
from agent.logger import get_logger, RUN_ID

log = get_logger("main")


def main():
    t0 = time.time()
    log.info("=" * 72)
    log.info("เริ่ม pipeline (run id = %s)", RUN_ID)

    df = P.step_load()
    ins = P.step_inspect(df)
    df, cl = P.step_clean(df)

    raw_amount = df["Amount"].values.copy()
    raw_time = df["Time"].values.copy()

    df, _ = P.step_features(df)
    Xtr, Xte, ytr, yte, sp = P.step_split(df)
    Xtr, Xte, sc = P.step_scale(Xtr, Xte)
    Xb, yb, sm = P.step_smote(Xtr, ytr)
    sv = P.step_save(Xtr, ytr, Xte, yte, Xb, yb)

    res, best = modeling.compare(Xtr, ytr, Xte, yte, Xb, yb)

    log.info("สร้างกราฟประกอบรายงาน")
    img_amount, img_hour = charts.amount_time(
        raw_amount, np.log1p(raw_amount), Xtr["Amount_scaled"].values, raw_time)
    imgs = {
        "amount": img_amount,
        "hour": img_hour,
        "balance": charts.class_balance(sm["before"], sm["after"]),
        "metrics": charts.metrics_chart(res),
    }

    S = {"run_id": RUN_ID, "inspect": ins, "clean": cl, "split": sp, "scale": sc,
         "smote": sm, "save": sv, "models": res, "best": best,
         "elapsed": round(time.time() - t0, 1)}
    report.generate(S, imgs)

    log.info("เสร็จสิ้นทั้ง pipeline ใช้เวลา %.1f วินาที", time.time() - t0)
    log.info("=" * 72)


if __name__ == "__main__":
    main()
