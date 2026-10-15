# RUN_LOG — บันทึกการทำงานของขั้น Analyze (3 อัลกอริทึม)

ไฟล์นี้เป็น **บันทึกไว้อ่านเอง** ว่าทำอะไรไปแล้ว เหลืออะไร และจะรันต่อยังไง
ถ้าปิดเครื่องแล้วกลับมาทำต่อ ให้อ่านไฟล์นี้คู่กับ `logs/analyze_progress.log`

- สรุปผลเชิงตัวเลข (สร้างใหม่อัตโนมัติทุกครั้งที่รัน) → `analyze/SUMMARY.md`
- log ต่อรอบของเครื่อง → `logs/run_<timestamp>.log`, `logs/agent.log`
- log ว่า step ไหนเสร็จแล้ว (append ไม่ทับ) → `logs/analyze_progress.log`
  (อยู่ใน `logs/` ซึ่ง `.gitignore` ไม่ commit — เป็นบันทึกของเครื่องนี้เท่านั้น)

---

## สิ่งที่ต้องมีก่อน (prerequisite)

ขั้น Analyze **ไม่อ่าน `creditcard.csv` ตรง ๆ** แต่อ่าน `outputs/processed.npz`
ที่ได้จาก pre-processing pipeline ดังนั้นถ้าไฟล์นั้นหาย ต้องรัน pipeline ก่อน

```powershell
python main.py            # STEP 1-10 สร้าง outputs/processed.npz
```

ตรวจว่าพร้อม: `outputs/processed.npz` มีอยู่ และมี key
`X_train (226980, 32)`, `X_test (56746, 32)`, `X_train_smote (339903, 32)`

---

## ลำดับการรัน (รันตามนี้ จากโฟลเดอร์ราก Data_pre)

| ลำดับ | คำสั่ง | เวลาที่ใช้ | ผลที่ได้ |
|------|--------|-----------|---------|
| 1 | `python analyze/logistic_regression/lr_analysis.py` | ~10 วินาที | `lr_*` ในโฟลเดอร์เดียวกัน |
| 2 | `python analyze/random_forest/rf_analysis.py` | ~85 วินาที (ช้าสุด) | `rf_*` |
| 3 | `python analyze/gradient_boosting/gb_analysis.py` | ~20 วินาที | `gb_*` |
| 4 | `python analyze/compare_all.py` | ~40 วินาที | `comparison_*` + `key_findings.*` + `SUMMARY.md` |

(เวลาที่วัดจริงบนเครื่องนี้เมื่อ 15 ต.ค. 2026 — รวมทั้งหมดไม่ถึง 3 นาที)

ข้อ 4 เรียก `analyze/key_findings.py` ให้เองอยู่แล้ว (ส่วนที่ใช้เวลาคือ bootstrap
หาช่วงความเชื่อมั่น 500 รอบ × 3 โมเดล) ถ้าจะแก้เฉพาะแผนภาพ Key Findings
รัน `python analyze/key_findings.py` เดี่ยว ๆ ได้ ไม่ต้องรัน compare_all ใหม่

ข้อ 1-3 เป็นอิสระต่อกัน รันสลับลำดับได้ แต่ **ข้อ 4 ต้องรันหลังสุด**
เพราะอ่านไฟล์ `*_summary.json` และ `*_test_scores.npz` ของทั้ง 3 ตัว
(ข้อ 4 ไม่เทรนใหม่ จึงรันซ้ำได้เร็วเมื่อแก้แค่กราฟหรือข้อความสรุป)

### ถ้าปิดเครื่องกลางทางแล้วกลับมาทำต่อ

```powershell
Get-Content logs/analyze_progress.log -Tail 20
```

แต่ละ ALGO จะมี 2 บรรทัดคือ `START` แล้ว `DONE`
- เห็น `DONE` ของ ALGO ไหนแล้ว → ข้ามตัวนั้นไปได้เลย (ไฟล์ผลลัพธ์ครบแล้ว)
- เห็นแค่ `START` ไม่มี `DONE` → รันตัวนั้นใหม่ทั้งตัว (เทรนไม่ได้ทำ checkpoint ระหว่างทาง)

---

## ทำอะไรไปแล้วบ้าง (ไล่ตามลำดับที่ทำจริง)

1. **อ่านโจทย์ก่อนลงมือ** — `AGENTS.md` (กฎห้ามละเมิด 7 ข้อ), `Introduction.txt`
   (วัตถุประสงค์: จำแนกธุรกรรมทุจริต + ประเมินด้วย AUPRC), `README.md`,
   และผลคัดฟีเจอร์ใน `feature/` ทั้ง 3 วิธี (correlation / lasso / random forest)
   → ได้ข้อสรุปว่าตัวชี้วัดหลักต้องเป็น **AUPRC และ Recall ไม่ใช่ accuracy**

2. **เลือก 3 อัลกอริทึมให้ครอบ 3 ตระกูลโมเดล** (ไม่ใช่ 3 ตัวที่คล้ายกัน)
   - Logistic Regression → linear
   - Random Forest → bagging ensemble
   - Histogram Gradient Boosting → boosting ensemble
   เลือก `HistGradientBoostingClassifier` ของ scikit-learn แทน XGBoost/LightGBM
   เพราะเครื่องนี้ยังไม่ได้ลง 2 ตัวนั้น และ `requirements.txt` ไม่มี

3. **แยกโค้ดที่ใช้ซ้ำไว้ที่ `analyze/common.py`** — โหลดข้อมูล, เทรน+วัดผล,
   เขียน CSV/JSON/PNG, จด progress log ทั้ง 3 สคริปต์เรียกใช้ตัวเดียวกัน
   จึงได้ตัวเลขที่เทียบกันได้ตรง ๆ และ `compare_all.py` รวมผลได้โดยไม่ต้องรู้ว่าโมเดลอะไร

4. **วางโครง `analyze/` ให้เหมือน `feature/`** — โฟลเดอร์ย่อยต่ออัลกอริทึม
   แต่ละตัวออก `*_results.csv`, `*_thresholds.csv`, `*_summary.json`, `*_analysis.png`
   แล้วมี `compare_all.py` ที่ระดับบนรวมผล (แบบเดียวกับ `feature/compare_all.py`)

5. **รันครบทั้ง 4 ขั้น** — ผลตัวเลขอยู่ใน `analyze/SUMMARY.md`

6. **ทำส่วน Data Visualization + Key Findings** (`analyze/key_findings.py`)
   เลือกรูปแบบกราฟตามหน้าที่ของข้อมูลในแต่ละช่อง ไม่ใช่เลือกเพราะสวย
   - เรื่องที่เป็น "ตัวเลขเดียว" (accuracy หลอก) → ใช้ stat tile ไม่ใช่กราฟแท่ง
   - เทียบตัวชี้วัด 2 ตัวที่ขัดกัน (ROC-AUC vs AUPRC) → scatter + ลูกศรชี้คู่ที่ขัดกัน
   - ความทนต่อ threshold → เส้น F1 เทียบจุดตัด
   - ผลของ SMOTE → แท่งคู่ ทึบ = ถ่วงน้ำหนักคลาส / ลาย = SMOTE
   - ความไม่แน่นอน → จุด + ช่วงความเชื่อมั่นจาก bootstrap
   - การแลกได้แลกเสีย → เส้นขั้นบันได TP เทียบ FP (แกน log)

   เพิ่มการคำนวณ "หลักฐาน" ที่ไม่เคยมีมาก่อน 3 อย่าง คือ bootstrap CI ของ AUPRC,
   การทดสอบว่า threshold ที่เลือกไว้ย้ายไปใช้กับข้อมูลอีกครึ่งแล้วแย่ลงเท่าไหร่
   และการไล่หาคู่ที่ ROC-AUC กับ AUPRC ให้คำตอบขัดกัน — ทั้งหมดไหลเข้า
   หัวข้อ "Key Findings" และ "ข้อควรระวัง" ใน `SUMMARY.md` อัตโนมัติ

### ปัญหาที่เจอระหว่างทำ และแก้ไปแล้ว

| ปัญหา | สาเหตุ | ที่แก้ |
|-------|--------|-------|
| F1 ที่ threshold 0.5 ต่ำมาก (0.10) ทั้งที่ AUPRC ดี | โมเดลที่ถ่วงน้ำหนักคลาสดันคะแนนไปกองใกล้ 1 จุดตัด 0.5 จึงเตือนผิดเป็นพันเคส | หา threshold ที่ให้ F1 สูงสุดจาก PR curve เต็มเส้น (`common.evaluate`) แล้วรายงานทั้ง 2 จุด |
| grid threshold 0.05-0.95 หาจุดที่ดีที่สุดไม่เจอ | จุดที่ดีที่สุดอยู่เหนือ 0.95 (เช่น 0.99996) | เติม 0.97/0.99/0.995/0.999 ลง grid และเก็บ threshold ละเอียด 6 ตำแหน่ง |
| `LogisticRegression(n_jobs=-1)` ขึ้น FutureWarning | sklearn 1.8 เลิกใช้ `n_jobs` ของ LR แล้ว | ถอด `n_jobs` ออกจาก LR (ของ RandomForest ยังใช้ได้) |
| หัวข้อกราฟภาษาไทยเคยขึ้นเป็นกล่องสี่เหลี่ยม | matplotlib ไม่ได้ตั้ง font ที่มีอักษรไทย | ตั้ง `font.family = ["Leelawadee UI", "Tahoma", "DejaVu Sans"]` ใน `common.py` (เครื่องนี้มี Leelawadee UI อยู่แล้ว) |
| สีเดิมที่ใช้ (`#6366f1/#10b981/#f59e0b`) ไม่ได้ตรวจเรื่องตาบอดสี | เลือกจากความสวยอย่างเดียว | เปลี่ยนเป็นจานสีที่ตรวจผ่านแล้ว (`#2a78d6/#eb6834/#1baf7a` + status) และ **ล็อกสีไว้กับอัลกอริทึม** ไม่เปลี่ยนตามอันดับหรือตามกราฟ |

---

## กฎที่ยึดตาม `AGENTS.md` (ตรวจแล้วทุกข้อ)

- ไม่มี data leakage: เทรนบน `X_train` / `X_train_smote` เท่านั้น แล้ว predict ลง `X_test`
  ไม่มีการ fit อะไรใหม่บน test
- SMOTE ใช้แต่ของ train ที่ `step_smote` ทำไว้แล้ว — **ไม่แตะ test** (fraud คงที่ 95/56,746)
- ไม่ scale `V1`-`V28` ซ้ำ (ใช้ข้อมูลจาก `processed.npz` ตามที่ pipeline ทำไว้)
- ไม่ impute อะไร (dataset ไม่มี missing value)
- รายงาน Recall / AUPRC เป็นตัวหลัก **ไม่รายงาน accuracy เลย** แม้ในกราฟ
- `random_state=42` ทุกโมเดลที่สุ่ม → รันซ้ำได้ผลเดิม
- log ภาษาไทยผ่าน logger กลาง (`agent/logger.py`) ไม่ `print()` ภาษาไทยตรง ๆ
- CSV เขียนด้วย `encoding="utf-8-sig"` เปิดใน Excel ได้
- หัวข้อกราฟภาษาไทยใช้ได้แล้ว เพราะตั้ง font family เป็น Leelawadee UI ไว้ใน
  `common.py` (ตามที่ `AGENTS.md` กำหนดว่าต้องตั้ง font ก่อนใส่ภาษาไทยในกราฟ)
  ถ้าย้ายไปเครื่องที่ไม่มีฟอนต์นี้ จะ fallback เป็น Tahoma แล้ว DejaVu Sans ตามลำดับ
- ไฟล์ที่สร้างใหม่ไม่มีตัวไหนเกิน 50 MB จึงไม่ต้อง track ด้วย Git LFS
  (`*_test_scores.npz` ประมาณ 0.2 MB ต่อไฟล์)

---

## ถ้าจะทำต่อ (ยังไม่ได้ทำ)

- เพิ่มอัลกอริทึมที่ 4: เขียนโฟลเดอร์ใหม่แบบเดียวกัน แล้วเพิ่ม 1 แถวใน
  `ALGOS` ของ `analyze/compare_all.py` — ตารางและกราฟวน list เองไม่ต้องแก้อย่างอื่น
- ลองเทรนด้วยฟีเจอร์เฉพาะ 8 ตัวที่ `feature/random_forest` บอกว่าให้พลังตัดสินใจ 80%
  เทียบว่า AUPRC ตกไหม (ถ้าไม่ตก ก็ลดต้นทุนการประมวลผลได้)
- ฝังผลขั้น Analyze ลงรายงาน PDF ของ pipeline (`agent/report.py`) ให้เป็นรายงานฉบับเดียว
- ปรับ hyperparameter ด้วย cross-validation บน train (ตอนนี้ตั้งค่าคงที่ไว้)
