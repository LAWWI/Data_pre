"""ค่าคอนฟิกกลางของ agent"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "creditcard.csv"
LOG_DIR = ROOT / "logs"
OUT_DIR = ROOT / "outputs"
for d in (LOG_DIR, OUT_DIR):
    d.mkdir(exist_ok=True)

TARGET = "Class"
RANDOM_STATE = 42
TEST_SIZE = 0.2
SMOTE_K = 5
# สัดส่วนที่ต้องการหลัง SMOTE (0.5 = minority เป็นครึ่งหนึ่งของ majority)
SMOTE_RATIO = 0.5
