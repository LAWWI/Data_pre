"""ตัว log กลาง: เขียนทั้งไฟล์และ console"""
import logging, sys, io
from datetime import datetime
from .config import LOG_DIR

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

def get_logger(name="agent"):
    lg = logging.getLogger(name)
    if lg.handlers:
        return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_DIR / f"run_{RUN_ID}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    ah = logging.FileHandler(LOG_DIR / "agent.log", encoding="utf-8")
    ah.setFormatter(fmt)
    try:  # Windows console เป็น cp1252 เขียนภาษาไทยไม่ได้ ต้องบังคับ UTF-8
        stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        stream = sys.stdout
    sh = logging.StreamHandler(stream)
    sh.setFormatter(fmt)
    for h in (fh, ah, sh):
        lg.addHandler(h)
    return lg
