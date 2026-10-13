# Data_pre — Agent ทำ Pre-processing ชุดข้อมูล Credit Card Fraud

Pipeline อัตโนมัติที่ทำ data pre-processing ชุดข้อมูล `creditcard.csv`
(Credit Card Fraud Detection — 284,807 แถว × 31 คอลัมน์) ตั้งแต่โหลดไฟล์
จนถึงออก **รายงาน PDF พร้อมกราฟ** และ **log ทีละขั้นตอน** ด้วยคำสั่งเดียว

จุดเน้นของโปรเจกต์คือ *ความถูกต้องของ pre-processing* — ไม่มี data leakage,
SMOTE ทำบน train เท่านั้น, และวัดผลด้วย Recall / PR-AUC ไม่ใช่ accuracy
(ข้อมูลนี้ fraud แค่ 0.17% ทายว่า "ไม่โกง" ทุกแถวก็ได้ accuracy 99.8%)

---

## เริ่มใช้งาน

### สิ่งที่ต้องมี
- Python 3.10+ (พัฒนาและทดสอบบน 3.13 / Windows)
- `creditcard.csv` วางไว้ที่ root ของโปรเจกต์
  (ไม่ได้ commit มาด้วยเพราะขนาด ~150 MB — ดาวน์โหลดจาก
  [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud))
- Chrome หรือ Edge สำหรับแปลง HTML → PDF *(ไม่มีก็ได้ จะได้แค่ HTML)*

### รัน
```powershell
pip install -r requirements.txt
python main.py
```
ใช้เวลาประมาณ 20 วินาที – 3 นาที ขึ้นกับเครื่อง

---

## ลำดับงานของ agent

| # | ขั้นตอน | รายละเอียด | ไฟล์ |
|---|---------|-----------|------|
| 1 | load | อ่าน CSV เข้า DataFrame | `agent/preprocess.py` |
| 2 | inspect | missing / duplicate / imbalance ratio / skew ของ `Amount` / ช่วง `Time` | `agent/preprocess.py` |
| 3 | clean | ลบแถวซ้ำ 1,081 แถว (เหลือ 283,726) — คง `Amount == 0` ไว้ | `agent/preprocess.py` |
| 4 | feature | `Time` → `Hour_sin`, `Hour_cos` (cyclic encoding 24 ชม.) | `agent/preprocess.py` |
| 5 | split | stratified 80/20 → train 226,980 / test 56,746 แถว | `agent/preprocess.py` |
| 6 | scale | `Amount`: log1p + RobustScaler · `Time`: StandardScaler — **fit บน train เท่านั้น** | `agent/preprocess.py` |
| 7 | balance | SMOTE บน train (`sampling_strategy=0.5`) เทียบกับ `class_weight='balanced'` | `agent/preprocess.py`, `agent/modeling.py` |
| 8 | save | `processed.npz`, `scalers.joblib`, sample CSV | `agent/preprocess.py` |
| 9 | verify | เทรน LogisticRegression 3 แบบเทียบผล | `agent/modeling.py` |
| 10 | report | HTML → PDF ด้วย Chrome headless | `agent/report.py`, `agent/charts.py` |

`V1`–`V28` **ไม่ถูก scale ซ้ำ** เพราะผ่าน PCA + standardize มาแล้วจากต้นทาง

---

## ผลที่ได้จากการตรวจสอบ (STEP 9)

วัดบน test set ที่คงการกระจายจริงไว้ (fraud 95 จาก 56,746 แถว)

| วิธีจัดการ imbalance | Precision | Recall | F1 | PR-AUC | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| ไม่แก้ imbalance | 0.851 | 0.600 | 0.704 | **0.696** | 57 | 10 | 38 |
| `class_weight='balanced'` | 0.056 | **0.874** | 0.105 | 0.681 | 83 | 1,407 | 12 |
| SMOTE (ratio 0.5) | 0.108 | 0.863 | 0.191 | 0.666 | 82 | 680 | 13 |

สรุป: การแก้ imbalance ดัน **Recall** ขึ้นจาก 0.60 → 0.87 (จับ fraud ได้เพิ่ม ~26 เคส)
แต่แลกมากับ false positive ที่พุ่งขึ้นมาก และ PR-AUC ไม่ได้ดีขึ้น
→ การเลือกวิธีขึ้นกับว่าต้นทางยอมรับ false positive ได้แค่ไหน ไม่ใช่ดูตัวเลขเดียว

---

## โครงสร้างโปรเจกต์

```
Data_pre/
├── main.py              orchestrator — เรียก step ตามลำดับ เก็บผลลง dict S
├── requirements.txt
├── AGENTS.md            คู่มือสำหรับ AI agent ที่มาแก้โปรเจกต์นี้ (กฎห้ามละเมิด)
├── creditcard.csv        ← ต้องวางเอง (ไม่ commit)
├── agent/
│   ├── config.py        path + hyperparameter ทั้งหมด
│   ├── logger.py        logger กลาง (ไฟล์ + console UTF-8) พร้อม RUN_ID
│   ├── preprocess.py    STEP 1–8
│   ├── modeling.py      STEP 9
│   ├── charts.py        กราฟ matplotlib → base64 PNG
│   └── report.py        STEP 10 — HTML → PDF
├── outputs/             (สร้างอัตโนมัติ)
└── logs/                (สร้างอัตโนมัติ)
```

### ไฟล์ผลลัพธ์
| ไฟล์ | เนื้อหา |
|------|---------|
| `outputs/report.pdf` / `report.html` | รายงานทุกขั้นตอน พร้อมกราฟ 4 ชุด |
| `outputs/processed.npz` | X/y ของ train, test, train(SMOTE) — ~73 MB |
| `outputs/scalers.joblib` | scaler ที่ fit แล้ว สำหรับ transform ข้อมูลใหม่ |
| `outputs/summary.json` | ตัวเลขสรุปทุกขั้นตอน (ใช้อ่านแทนการเปิด CSV) |
| `outputs/sample_test_2000.csv` | ตัวอย่าง test set 2,000 แถว (UTF-8 BOM เปิดใน Excel ได้) |
| `logs/agent.log` | log สะสมทุกรอบ |
| `logs/run_<timestamp>.log` | log แยกต่อรอบ |

---

## ปรับแต่ง

แก้ `agent/config.py` — ไม่ hardcode ค่าในฟังก์ชัน

| ตัวแปร | ค่าเริ่มต้น | ความหมาย |
|--------|------------|----------|
| `RANDOM_STATE` | 42 | seed ของทุกขั้นที่สุ่ม (reproducible) |
| `TEST_SIZE` | 0.2 | สัดส่วน test set |
| `SMOTE_K` | 5 | จำนวน neighbor ของ SMOTE |
| `SMOTE_RATIO` | 0.5 | สัดส่วนหลัง SMOTE (0.5 = minority เป็นครึ่งของ majority) |

---

## ข้อควรรู้

- **รันซ้ำจะทับไฟล์ใน `outputs/` ทั้งหมด** ถ้าต้องเก็บผลเก่าให้ copy ออกก่อน
- ถ้าหา Chrome/Edge ไม่เจอ จะ warn แล้วข้าม PDF — HTML ยังออกปกติ ไม่ถือว่า fail
- ไม่มี test suite การยืนยันว่าแก้ถูกคือรัน `python main.py` ให้จบ แล้วเช็กว่า
  `logs/run_<timestamp>.log` ผ่านครบ STEP 1–10 และ `outputs/report.pdf` ถูกสร้างใหม่
- จะแก้โค้ดโปรเจกต์นี้ **อ่าน `AGENTS.md` ก่อน** — มีกฎเรื่อง data leakage,
  ขอบเขตของ SMOTE และแบบแผนของโค้ดที่ต้องทำตาม
