# Data_pre — Agent ทำ Pre-processing ชุดข้อมูล Credit Card Fraud

Pipeline อัตโนมัติที่ทำ data pre-processing ชุดข้อมูล `creditcard.csv`
(Credit Card Fraud Detection — 284,807 แถว × 31 คอลัมน์) ตั้งแต่โหลดไฟล์
จนถึงออก **รายงาน PDF พร้อมกราฟ** และ **log ทีละขั้นตอน** ด้วยคำสั่งเดียว

จุดเน้นของโปรเจกต์คือ *ความถูกต้องของ pre-processing* — ไม่มี data leakage,
SMOTE ทำบน train เท่านั้น, และวัดผลด้วย Recall / PR-AUC ไม่ใช่ accuracy
(ข้อมูลนี้ fraud แค่ 0.17% ทายว่า "ไม่โกง" ทุกแถวก็ได้ accuracy 99.8%)

---

## User Stories

ความต้องการที่โปรเจกต์นี้ตอบ เขียนในมุมของผู้ใช้แต่ละแบบ

### US-1 — รัน pre-processing ทั้งชุดด้วยคำสั่งเดียว
> **ในฐานะ** นักศึกษา/นักวิเคราะห์ข้อมูล
> **ฉันต้องการ** รัน pre-processing ทั้ง pipeline ด้วย `python main.py` คำสั่งเดียว
> **เพื่อ** ไม่ต้องไล่รัน notebook ทีละ cell และได้ผลเหมือนกันทุกครั้ง

เงื่อนไขว่าสำเร็จ
- [x] คำสั่งเดียวทำครบ STEP 1–10 จบใน ~20 วินาที
- [x] ตั้ง `random_state=RANDOM_STATE` ทุกขั้นที่สุ่ม → รันซ้ำได้ผลเดิม
- [x] โฟลเดอร์ `outputs/`, `logs/` ถูกสร้างเองถ้ายังไม่มี

### US-2 — ได้รายงานที่ส่งอาจารย์/หัวหน้าได้เลย
> **ในฐานะ** คนที่ต้องส่งงาน
> **ฉันต้องการ** ไฟล์รายงานที่อ่านรู้เรื่องพร้อมกราฟ ไม่ใช่แค่ข้อความใน console
> **เพื่อ** ส่งงานได้ทันทีโดยไม่ต้องมานั่งเรียบเรียงใหม่

เงื่อนไขว่าสำเร็จ
- [x] ได้ `outputs/report.pdf` (และ `report.html`) ภาษาไทย พร้อมกราฟ 4 ชุด
- [x] รายงานแสดงตัวเลขจริงทุก step ไม่ใช่ค่าที่เขียนตายไว้
- [x] ถ้าเครื่องไม่มี Chrome/Edge ยังได้ HTML ไม่ fail ทั้ง pipeline

### US-3 — เชื่อถือตัวเลขได้ว่าไม่มี data leakage
> **ในฐานะ** คนตรวจงาน / คนที่จะเอาผลไปใช้ต่อ
> **ฉันต้องการ** มั่นใจว่า scaler fit บน train เท่านั้น และ SMOTE ไม่แตะ test set
> **เพื่อ** ให้ตัวเลขที่รายงานเป็นผลจริง ไม่ใช่ผลที่สูงเกินจริงจาก leakage

เงื่อนไขว่าสำเร็จ
- [x] `step_scale` fit บน train แล้ว transform ลง test (ดูค่า `amount_center`, `time_mean` ใน `summary.json`)
- [x] test set คงการกระจายจริงไว้ fraud 95 / 56,746 แถว (0.17%)
- [x] `V1`–`V28` ไม่ถูก scale ซ้ำ
- [x] กฎทั้งหมดเขียนไว้เป็นลายลักษณ์อักษรใน `AGENTS.md`

### US-4 — เห็นว่าการแก้ imbalance แลกอะไรไปกับอะไร
> **ในฐานะ** คนเลือกวิธีทำโมเดลขั้นถัดไป
> **ฉันต้องการ** เทียบ "ไม่แก้ imbalance / class_weight / SMOTE" บนข้อมูลชุดเดียวกัน
> **เพื่อ** ตัดสินใจจาก trade-off จริง ไม่ใช่เดาว่า SMOTE ดีกว่าเสมอ

เงื่อนไขว่าสำเร็จ
- [x] ตารางเทียบ Precision / Recall / F1 / PR-AUC / TP-FP-FN ทั้ง 3 วิธี
- [x] รายงานด้วย Recall และ PR-AUC เป็นตัวหลัก **ไม่ใช่** accuracy
- [x] มีกราฟเทียบจำนวนคลาสก่อน–หลัง SMOTE

### US-5 — นำผลที่ประมวลผลแล้วไปใช้ต่อได้
> **ในฐานะ** คนที่จะเทรนโมเดลจริงต่อจากขั้นนี้
> **ฉันต้องการ** ไฟล์ X/y ที่พร้อมใช้ และ scaler ที่ fit แล้ว
> **เพื่อ** ไม่ต้องรัน pre-processing ซ้ำ และ transform ข้อมูลใหม่ได้ด้วยค่าเดิม

เงื่อนไขว่าสำเร็จ
- [x] `outputs/processed.npz` มี X/y ของ train, test และ train(SMOTE)
- [x] `outputs/scalers.joblib` โหลดกลับมา `transform()` ข้อมูลใหม่ได้
- [x] `outputs/summary.json` อ่านตัวเลขทุกขั้นได้โดยไม่ต้องเปิด CSV 150 MB

### US-6 — ย้อนดูได้ว่าแต่ละรอบเกิดอะไรขึ้น
> **ในฐานะ** คนที่ต้อง debug เมื่อผลเปลี่ยน
> **ฉันต้องการ** log แยกต่อรอบ พร้อม run id
> **เพื่อ** เทียบได้ว่าการรันรอบไหนให้ผลอะไร และพลาดที่ step ใด

เงื่อนไขว่าสำเร็จ
- [x] `logs/run_<timestamp>.log` แยกต่อรอบ และ `logs/agent.log` สะสมทุกรอบ
- [x] log ทุกขั้นขึ้นหัวด้วย `STEP n` เป็นภาษาไทย
- [x] `run_id` ปรากฏทั้งใน log, `summary.json` และหัวรายงาน

### US-7 — แก้โปรเจกต์ต่อได้โดยไม่ทำของเดิมพัง
> **ในฐานะ** คน (หรือ AI agent) ที่มารับงานต่อ
> **ฉันต้องการ** รู้ว่าโครงสร้างโค้ดเป็นแบบไหน และอะไรห้ามแตะ
> **เพื่อ** เพิ่ม step/โมเดล/กราฟ ได้โดยไม่ทำผลวิเคราะห์เพี้ยน

เงื่อนไขว่าสำเร็จ
- [x] `AGENTS.md` ระบุกฎห้ามละเมิด 7 ข้อ และวิธีเพิ่มของใหม่
- [x] ทุก step คืนค่าเป็น `(ผลลัพธ์, dict สรุป)` รูปแบบเดียวกันหมด
- [x] hyperparameter รวมอยู่ที่ `agent/config.py` ที่เดียว

---

## เริ่มใช้งาน

### สิ่งที่ต้องมี
- Python 3.10+ (พัฒนาและทดสอบบน 3.13 / Windows)
- [Git LFS](https://git-lfs.com/) — `creditcard.csv` (~150 MB) และ `outputs/processed.npz`
  (~73 MB) commit มาด้วยแล้วผ่าน LFS ติดตั้ง LFS ก่อน clone แล้วไฟล์จะมาครบเอง
  ถ้า clone มาแล้วได้เป็นไฟล์ pointer ข้อความสั้นๆ ให้รัน `git lfs pull`
  ต้นฉบับของ dataset มาจาก
  [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
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
├── .gitattributes       กฎ Git LFS สำหรับไฟล์ข้อมูลขนาดใหญ่
├── creditcard.csv       ข้อมูลดิบ 284,807 แถว (commit ผ่าน Git LFS)
├── agent/
│   ├── config.py        path + hyperparameter ทั้งหมด
│   ├── logger.py        logger กลาง (ไฟล์ + console UTF-8) พร้อม RUN_ID
│   ├── preprocess.py    STEP 1–8
│   ├── modeling.py      STEP 9
│   ├── charts.py        กราฟ matplotlib → base64 PNG
│   └── report.py        STEP 10 — HTML → PDF
├── outputs/             ผลลัพธ์ของรอบล่าสุด (commit มาด้วย)
└── logs/                (สร้างอัตโนมัติ ไม่ commit)
```

### ไฟล์ผลลัพธ์
| ไฟล์ | เนื้อหา |
|------|---------|
| `outputs/report.pdf` / `report.html` | รายงานทุกขั้นตอน พร้อมกราฟ 4 ชุด |
| `outputs/processed.npz` | X/y ของ train, test, train(SMOTE) — ~73 MB |
| `outputs/scalers.joblib` | scaler ที่ fit แล้ว สำหรับ transform ข้อมูลใหม่ |
| `outputs/summary.json` | ตัวเลขสรุปทุกขั้นตอน (ใช้อ่านแทนการเปิด CSV) |
| `outputs/sample_test_2000.csv` | ตัวอย่าง test set 2,000 แถว (UTF-8 BOM เปิดใน Excel ได้) |
| `logs/agent.log` | log สะสมทุกรอบ *(ไม่ commit — เกิดใหม่เมื่อรัน)* |
| `logs/run_<timestamp>.log` | log แยกต่อรอบ *(ไม่ commit — เกิดใหม่เมื่อรัน)* |

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
