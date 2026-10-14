# AGENTS.md — คู่มือสำหรับ AI agent ที่มาแก้โปรเจกต์นี้

โปรเจกต์นี้คือ pipeline ทำ data pre-processing ของ dataset `creditcard.csv`
(Credit Card Fraud Detection, 284,807 แถว × 31 คอลัมน์) แล้วออกรายงาน PDF + log อัตโนมัติ

## คำสั่งที่ใช้

```powershell
pip install -r requirements.txt
python main.py                 # รันทั้ง pipeline (STEP 1-10) ใช้เวลา ~1-3 นาที
```

ไม่มี test suite / linter ในโปรเจกต์นี้ การยืนยันว่าแก้ถูกคือ **รัน `python main.py` ให้จบ**
แล้วเช็ก `logs/run_<timestamp>.log` ว่าผ่านครบ STEP 1-10 และ `outputs/report.pdf` ถูกสร้างใหม่

## โครงสร้าง

| ไฟล์ | หน้าที่ |
|------|---------|
| `main.py` | orchestrator เรียก step ตามลำดับ เก็บผลลง dict `S` แล้วส่งให้ `report.generate` |
| `agent/config.py` | path + hyperparameter ทั้งหมด (`RANDOM_STATE`, `TEST_SIZE`, `SMOTE_K`, `SMOTE_RATIO`) |
| `agent/logger.py` | logger กลาง เขียนทั้งไฟล์และ console พร้อม `RUN_ID` |
| `agent/preprocess.py` | STEP 1-8: load, inspect, clean, features, split, scale, smote, save |
| `agent/modeling.py` | STEP 9: เทรน LogisticRegression 3 แบบเทียบผล |
| `agent/charts.py` | กราฟ matplotlib → base64 PNG ฝังใน HTML |
| `agent/report.py` | STEP 10: สร้าง HTML → PDF ด้วย Chrome headless |
| `outputs/`, `logs/` | ผลลัพธ์ทั้งหมด (สร้างอัตโนมัติโดย `config.py`) — `outputs/` commit ไว้ด้วย, `logs/` ไม่ commit |

`creditcard.csv` ขนาด ~150 MB commit ไว้แล้วผ่าน Git LFS (กฎอยู่ใน `.gitattributes`)
แต่ **ห้ามอ่านทั้งไฟล์เข้า context** ถ้าต้องดูข้อมูลให้ใช้ `outputs/sample_test_2000.csv`
หรืออ่าน `outputs/summary.json` — ไฟล์ใหม่ที่ใหญ่กว่า 50 MB ให้ track ด้วย LFS ด้วยเสมอ

## กฎที่ห้ามละเมิด (ไม่อย่างนั้นผลวิเคราะห์ผิด)

1. **ห้ามเกิด data leakage** — scaler ทุกตัวต้อง `fit` บน train set เท่านั้น แล้ว `transform` ลง test
   (ดู `step_scale`) ห้าม fit บนข้อมูลรวมหรือบน test
2. **SMOTE ทำบน train เท่านั้น** — test set ต้องคงการกระจายจริง (fraud ~0.17%) ไว้เสมอ
   ห้าม resample/oversample test
3. **ห้าม scale `V1`–`V28` ซ้ำ** — คอลัมน์เหล่านี้ผ่าน PCA + standardize มาแล้ว
   ที่ต้อง scale มีแค่ `Time` และ `Amount`
4. **ห้าม impute** — dataset นี้ไม่มี missing value (ยืนยันใน `step_inspect`)
   ถ้าโค้ดเริ่มต้องการ impute แปลว่าทำอะไรพลาดก่อนหน้านั้น
5. **ตัวชี้วัดคือ Recall / PR-AUC ไม่ใช่ accuracy** — ทายว่า "ไม่โกง" ทุกแถวก็ได้ accuracy 99.8%
   ห้ามเปลี่ยนไปรายงาน accuracy เป็นตัวหลัก
6. **ทุกอย่างต้อง reproducible** — ส่ง `random_state=RANDOM_STATE` ให้ทุกฟังก์ชันที่สุ่ม
7. `Amount == 0` และค่าติดลบ **คงไว้** ไม่ใช่ outlier เชิง logic (ดู `step_clean`)

## แบบแผนของโค้ดที่ต้องทำตาม

- **ทุก step คืนค่า `(ผลลัพธ์, dict สรุปตัวเลข)`** — dict นั้นไหลไปที่ `S` ใน `main.py`
  แล้วถูกใช้ทั้งใน `summary.json` และตารางในรายงาน
  เพิ่ม step ใหม่ต้องคืน dict ด้วย ไม่อย่างนั้นรายงานจะไม่มีตัวเลขให้แสดง
- **ค่าใน dict สรุปต้องเป็น Python native** (`int`, `float`, `str`, `list`, `dict`)
  ไม่ใช่ `np.int64` / `np.float32` เพราะ `json.dumps` จะพัง — cast ด้วย `int()` / `float()` / `round()`
- **log ทุกขั้นตอนด้วย `log.info` เป็นภาษาไทย** ขึ้นหัวด้วย `STEP n` สำหรับ step หลัก
  ใช้ lazy formatting (`log.info("... %s", x)`) ไม่ใช้ f-string ใน log call
- **docstring และ comment เขียนภาษาไทย** ศัพท์เทคนิค (SMOTE, data leakage, skew) คงภาษาอังกฤษ
- hyperparameter ใหม่ไปไว้ใน `agent/config.py` ไม่ hardcode ในฟังก์ชัน
- import logger ด้วย `log = get_logger("<ชื่อโมดูล>")` ที่ระดับโมดูล

## ข้อจำกัดของเครื่อง (Windows)

- console เป็น cp1252 เขียนภาษาไทยตรง ๆ ไม่ได้ — `logger.py` wrap `sys.stdout` เป็น UTF-8 ไว้แล้ว
  อย่าเอาออก และอย่า `print()` ภาษาไทยตรง ๆ ให้ใช้ logger
- ไม่มี pandoc / wkhtmltopdf — **PDF สร้างด้วย Chrome/Edge headless** (`report.to_pdf`)
  ถ้าหา browser ไม่เจอจะ warn แล้วข้าม PDF (HTML ยังออก) ซึ่งถือว่าไม่ fail
- CSV ที่จะเปิดใน Excel ต้องเขียนด้วย `encoding="utf-8-sig"` (UTF-8 BOM) ไม่อย่างนั้นภาษาไทยเพี้ยน
- ฟอนต์ใน HTML ใช้ `"Leelawadee UI","Tahoma"` สำหรับภาษาไทย
  กราฟ matplotlib ถ้าใส่ title ภาษาไทยต้องตั้ง font family ที่รองรับ ไม่อย่างนั้นได้กล่องสี่เหลี่ยม

## เพิ่มของใหม่ยังไง

- **เพิ่ม step pre-processing**: เขียน `step_xxx` ใน `agent/preprocess.py` คืน `(ผล, dict)`
  → เรียกใน `main.py` ตามลำดับ → ใส่ key ลง `S` → เพิ่มแถวในตาราง `t_steps` ของ `report.py`
- **เพิ่มโมเดล**: เพิ่ม `_eval(...)` อีกบรรทัดใน `modeling.compare` — ตาราง `t_m`
  และกราฟ `metrics_chart` วน list `res` อยู่แล้ว ไม่ต้องแก้
- **เพิ่มกราฟ**: ฟังก์ชันใหม่ใน `charts.py` คืนค่าผ่าน `_b64(fig)` → ใส่ใน dict `imgs`
  → อ้างด้วย `<img src="{...}">` ใน `build_html`
- **เพิ่มหัวข้อในรายงาน**: แก้ template string ใน `report.build_html`
  ทุก placeholder ต้องมีค่าส่งใน `.format(...)` ท้ายฟังก์ชัน ไม่อย่างนั้น `KeyError`

## เตือน

- `main.py` เก็บ `raw_amount` / `raw_time` ไว้ **ก่อน** `step_features`/`step_scale`
  เพราะกราฟต้องใช้ค่าดิบ ถ้าย้ายลำดับบรรทัดนี้กราฟจะผิด
- `logs/agent.log` เป็น log สะสมทุกรอบ (append) ส่วน `run_<RUN_ID>.log` แยกต่อรอบ — อย่าลบทิ้ง
- รันซ้ำจะ **ทับ** ไฟล์ใน `outputs/` ทั้งหมด ถ้าต้องเก็บผลเก่าให้ copy ออกก่อน
