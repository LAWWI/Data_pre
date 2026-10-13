"""สร้างรายงาน HTML แล้วแปลงเป็น PDF ด้วย Chrome headless"""
import json, subprocess, shutil, time
from datetime import datetime
from pathlib import Path
from .config import OUT_DIR
from .logger import get_logger, RUN_ID

log = get_logger("report")

CHROME = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
          r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
          r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]

CSS = """
*{box-sizing:border-box}
body{font-family:"Leelawadee UI","Tahoma","Segoe UI",sans-serif;font-size:10.5pt;
 color:#111827;line-height:1.55;margin:0}
h1{font-size:20pt;margin:0 0 2px;color:#0f172a}
h2{font-size:13pt;margin:18px 0 6px;padding-bottom:4px;
 border-bottom:2px solid #2563eb;color:#1e3a8a}
h3{font-size:11pt;margin:12px 0 4px;color:#1e3a8a}
.sub{color:#6b7280;font-size:9pt;margin-bottom:14px}
table{border-collapse:collapse;width:100%;font-size:9pt;margin:6px 0 10px}
th,td{border:1px solid #e5e7eb;padding:4px 7px;text-align:left;vertical-align:top}
th{background:#eff6ff;color:#1e3a8a;font-weight:600}
tr.best td{background:#ecfdf5;font-weight:600}
code{background:#f3f4f6;padding:1px 4px;border-radius:3px;font-size:9pt;
 font-family:Consolas,monospace}
.cards{display:flex;gap:8px;margin:8px 0 12px}
.card{flex:1;border:1px solid #e5e7eb;border-left:3px solid #2563eb;
 border-radius:4px;padding:7px 9px;background:#fafafa}
.card .k{font-size:8pt;color:#6b7280}
.card .v{font-size:13pt;font-weight:700;color:#1e3a8a}
img{width:100%;margin:6px 0 10px;border:1px solid #e5e7eb;border-radius:4px}
.note{background:#fffbeb;border-left:3px solid #f59e0b;padding:7px 10px;
 font-size:9.2pt;margin:8px 0;border-radius:3px}
ul,ol{margin:4px 0 8px;padding-left:20px}
li{margin:2px 0}
.pb{page-break-before:always}
@page{size:A4;margin:14mm 13mm}
"""


def _tbl(header, rows, best_name=None):
    h = "<tr>" + "".join("<th>%s</th>" % c for c in header) + "</tr>"
    b = ""
    for r in rows:
        cls = ' class="best"' if best_name and r[0] == best_name else ""
        b += "<tr%s>" % cls + "".join("<td>%s</td>" % c for c in r) + "</tr>"
    return "<table>%s%s</table>" % (h, b)


def build_html(S, imgs):
    ins, cl, sc, sp, sm = (S["inspect"], S["clean"], S["scale"],
                           S["split"], S["smote"])
    res, best = S["models"], S["best"]
    cc = ins["class_counts"]
    n0, n1 = cc.get(0, 0), cc.get(1, 0)

    t_check = _tbl(
        ["รายการตรวจสอบ", "ค่าที่พบ", "ข้อสรุปว่าต้องทำอะไร"],
        [["จำนวนแถว / คอลัมน์", "{:,} / {}".format(ins["rows"], ins["cols"]), "—"],
         ["ค่า missing ทั้งตาราง", str(ins["missing_total"]), "ไม่ต้อง impute"],
         ["แถวซ้ำ (duplicate)", "{:,}".format(ins["duplicates"]),
          "ต้องลบ ป้องกันข้อมูลเดียวกันโผล่ทั้ง train และ test"],
         ["ชนิดข้อมูล", "ตัวเลขทั้งหมด", "ไม่ต้อง encode ตัวแปรเชิงกลุ่ม"],
         ["V1–V28", "ผ่าน PCA + standardize มาแล้ว", "ไม่ scale ซ้ำ"],
         ["Time", "ช่วง 0–{:,.0f} ชม. (หน่วยวินาที)".format(ins["time_range_h"]),
          "สเกลใหญ่กว่าฟีเจอร์อื่นมาก ต้อง scale"],
         ["Amount", "เบ้ขวาจัด skew = {}".format(ins["amount_skew"]),
          "ต้อง log1p แล้วตามด้วย robust scale"],
         ["Class", "0 = {:,} / 1 = {:,}".format(n0, n1),
          "imbalance รุนแรง ต้องทำ SMOTE หรือใช้ balanced mode"]])

    t_steps = _tbl(
        ["#", "ขั้นตอน", "รายละเอียดที่ทำ"],
        [["1", "โหลดข้อมูล", "อ่าน creditcard.csv → {:,} × {}".format(ins["rows"], ins["cols"])],
         ["2", "ตรวจคุณภาพข้อมูล",
          "นับ missing / duplicate / การกระจาย Class / สถิติของ Time และ Amount"],
         ["3", "ทำความสะอาด",
          "ลบแถวซ้ำ {:,} แถว → เหลือ {:,} แถว (Amount = 0 มี {} แถว คงไว้ "
          "เพราะเป็นค่าที่เกิดได้จริง)".format(cl["dup_dropped"], cl["rows_after"],
                                               cl["amount_zero"])],
         ["4", "สร้างฟีเจอร์เวลา",
          "แปลง Time เป็นชั่วโมงในวัน แล้วเข้ารหัสแบบ cyclic เป็น Hour_sin, Hour_cos"],
         ["5", "แบ่ง train/test",
          "stratified 80/20 → train {:,} แถว (โกง {}) | test {:,} แถว (โกง {})".format(
              sp["train_rows"], sp["train_fraud"], sp["test_rows"], sp["test_fraud"])],
         ["6", "ปรับ scale Time, Amount",
          "fit scaler บน train เท่านั้น แล้ว transform ลง test"],
         ["7", "แก้ class imbalance",
          "SMOTE บน train สังเคราะห์เพิ่ม {:,} แถว และเทียบกับโหมด "
          "class_weight='balanced'".format(sm["synthetic"])],
         ["8", "บันทึกผลลัพธ์",
          "processed.npz ({} MB), scalers.joblib, sample CSV, summary.json".format(
              S["save"]["npz_mb"])],
         ["9", "ตรวจสอบด้วยโมเดล", "เทรน LogisticRegression 3 แบบ เทียบผลบน test จริง"],
         ["10", "ออกรายงาน", "สร้าง report.html → report.pdf และเขียน log ทุกขั้นตอน"]])

    t_bal = _tbl(
        ["ชุดข้อมูล", "Class 0", "Class 1", "สัดส่วนโกง"],
        [["ก่อน SMOTE (train)", "{:,}".format(sm["before"].get(0, 0)),
          "{:,}".format(sm["before"].get(1, 0)),
          "{:.3f}%".format(100 * sm["before"].get(1, 0) / sum(sm["before"].values()))],
         ["หลัง SMOTE (train)", "{:,}".format(sm["after"].get(0, 0)),
          "{:,}".format(sm["after"].get(1, 0)),
          "{:.3f}%".format(100 * sm["after"].get(1, 0) / sum(sm["after"].values()))],
         ["Test set (ไม่แตะ)", "{:,}".format(sp["test_rows"] - sp["test_fraud"]),
          "{:,}".format(sp["test_fraud"]),
          "{:.3f}%".format(100 * sp["test_fraud"] / sp["test_rows"])]])

    t_m = _tbl(
        ["วิธี", "Recall", "Precision", "F1", "ROC-AUC", "PR-AUC", "จับโกงได้", "False alarm"],
        [[r["name"], r["recall"], r["precision"], r["f1"], r["roc_auc"], r["pr_auc"],
          "{}/{}".format(r["tp"], r["tp"] + r["fn"]), r["fp"]] for r in res],
        best_name=best)

    t_files = _tbl(
        ["ไฟล์", "เนื้อหา"],
        [["outputs/processed.npz ({} MB)".format(S["save"]["npz_mb"]),
          "X/y ของ train, test และ train ที่ผ่าน SMOTE พร้อมชื่อฟีเจอร์"],
         ["outputs/scalers.joblib",
          "RobustScaler (Amount) + StandardScaler (Time) ที่ fit แล้ว ใช้ transform ข้อมูลใหม่"],
         ["outputs/sample_test_2000.csv", "ตัวอย่าง test set 2,000 แถว (UTF-8 BOM เปิดใน Excel ได้)"],
         ["outputs/summary.json", "ตัวเลขสรุปทุกขั้นตอนแบบเครื่องอ่านได้"],
         ["logs/run_{}.log".format(RUN_ID), "log ของรอบรันนี้ แยกทีละขั้นตอน"],
         ["logs/agent.log", "log รวมสะสมทุกรอบรัน"],
         ["outputs/report.html / report.pdf", "รายงานฉบับนี้"]])

    extra = ", ".join("<code>%s</code>" % f for f in sc["feature_names"]
                      if not f.startswith("V"))
    dropped = ", ".join("<code>%s</code>" % c for c in sc["dropped"])

    return """<title>รายงาน Pre-processing Credit Card Fraud</title>
<style>{css}</style>
<h1>รายงานการทำ Data Pre-processing</h1>
<div class="sub">Dataset: <b>creditcard.csv</b> (Credit Card Fraud Detection) ·
run id <code>{run}</code> · สร้างเมื่อ {now}</div>

<div class="cards">
 <div class="card"><div class="k">แถวข้อมูลเริ่มต้น</div><div class="v">{rows:,}</div></div>
 <div class="card"><div class="k">ฟีเจอร์หลัง process</div><div class="v">{nfeat}</div></div>
 <div class="card"><div class="k">ธุรกรรมโกง</div><div class="v">{fpct}%</div></div>
 <div class="card"><div class="k">อัตราส่วน 0 : 1</div><div class="v">1:{ratio:,.0f}</div></div>
</div>

<h2>1. ข้อมูลต้องทำอะไร — ผลการตรวจสอบ CSV</h2>
{t_check}
<div class="note"><b>ทำไม imbalance สำคัญ:</b> ถ้าโมเดลทายว่า “ไม่โกง” ทุกแถว
จะได้ accuracy สูงถึง {acc:.3f}% แต่จับการโกงไม่ได้แม้แต่รายการเดียว
จึงต้องวัดผลด้วย Recall / PR-AUC และต้อง rebalance ชุดเทรน</div>

<h2>2. สิ่งที่ทำไปทั้งหมด (ตามลำดับ)</h2>
{t_steps}

<h2 class="pb">3. การปรับ scale คอลัมน์ Time และ Amount</h2>
<h3>Amount — log1p แล้วตามด้วย RobustScaler</h3>
<ul>
<li>Amount เบ้ขวาจัด (skew {skew}) มีรายการยอดสูงผิดปกติจำนวนมาก
 จึงใช้ <code>log1p</code> ลดความเบ้ก่อน</li>
<li>ตามด้วย <b>RobustScaler</b> ที่อิง median และ IQR ทนค่า outlier ได้ดีกว่า StandardScaler</li>
<li>ค่าที่ fit ได้จาก train: center = {a_c}, scale (IQR) = {a_s}</li>
</ul>
<h3>Time — StandardScaler</h3>
<ul>
<li>ค่าดิบเป็นวินาทีนับจากธุรกรรมแรก (0 ถึง {trange:,.0f} ชม.) สเกลใหญ่กว่า V1–V28 หลายพันเท่า
 จึงบีบให้ mean = 0, std = 1</li>
<li>ค่าที่ fit ได้จาก train: mean = {t_m:,}, std = {t_s:,}</li>
<li>เพิ่ม Hour_sin / Hour_cos เพื่อเก็บพฤติกรรมตามช่วงเวลา
 และทำให้ 23:00 กับ 00:00 อยู่ใกล้กันในเชิงตัวเลข</li>
</ul>
<div class="note"><b>ป้องกัน data leakage:</b> scaler ทั้งสองตัว <code>fit</code>
จาก train set เท่านั้น แล้วจึง <code>transform</code> ลง test set
ไม่ให้สถิติของ test รั่วเข้าสู่กระบวนการเทรน</div>
<p>คอลัมน์ที่ตัดออกหลังแปลง: {dropped}<br>
ฟีเจอร์ที่เพิ่มเข้ามา: {extra} รวมกับ V1–V28 เดิม = <b>{nfeat} ฟีเจอร์</b></p>
<img src="{img_amount}">
<img src="{img_hour}">

<h2 class="pb">4. การแก้ปัญหา class imbalance</h2>
<p>ใช้ <b>SMOTE</b> (k_neighbors = {k}, sampling_strategy = {ratio_sm})
<b>บนชุด train เท่านั้น</b> ส่วน test set คงการกระจายจริงไว้
เพื่อให้ตัวเลขที่วัดได้ตรงกับการใช้งานจริง</p>
{t_bal}
<img src="{img_balance}">
<div class="note">ทำทั้งสองแนวทางไว้ให้เลือกใช้: (ก) ชุดข้อมูล SMOTE ที่สมดุลแล้ว
และ (ข) ใช้ข้อมูลเดิมพร้อมบอกโมเดลว่า <code>class_weight='balanced'</code>
ซึ่งไม่สร้างข้อมูลสังเคราะห์ จึงเทรนเร็วกว่าและไม่เสี่ยง overfit จุดที่สังเคราะห์ขึ้น</div>

<h2>5. ผลตรวจสอบด้วยโมเดล (LogisticRegression)</h2>
{t_m_tbl}
<img src="{img_metrics}">
<p>แถวที่ไฮไลต์คือวิธีที่ได้ PR-AUC สูงสุด (<b>{best}</b>)
แต่ต้องอ่านตารางนี้สองชั้น:</p>
<ul>
<li><b>PR-AUC ใกล้กันทั้งสามวิธี</b> ({pr_lo}–{pr_hi}) เพราะเป็นค่าที่วัด “การเรียงลำดับความเสี่ยง”
 ซึ่ง rebalance ไม่ได้เปลี่ยนมากนักสำหรับ LogisticRegression</li>
<li>สิ่งที่ rebalance เปลี่ยนจริงคือ <b>จุดตัดสินใจ (threshold ภายใน)</b>:
 recall ขยับจาก {r_base} เป็น {r_bal} (balanced) และ {r_smote} (SMOTE)
 คือจับการโกงได้เพิ่มจาก {tp_base} เป็น {tp_bal} จาก {n_fraud} รายการ</li>
<li>ราคาที่จ่ายคือ false alarm เพิ่มจาก {fp_base} เป็น {fp_bal} รายการ
 (precision ลดลง) — ในงานจับโกงมักยอมแลก เพราะ false negative แพงกว่ามาก</li>
</ul>
<div class="note"><b>ข้อเสนอแนะต่อ:</b> ใช้ข้อมูลที่ pre-process แล้วไปต่อกับ
XGBoost / LightGBM โดยตั้ง <code>scale_pos_weight</code> (ทำหน้าที่เดียวกับ balanced mode)
แล้วจูน threshold จาก precision–recall curve ให้ตรงกับต้นทุนธุรกิจ
จะได้ผลดีกว่าการเลือกเพียงวิธี rebalance อย่างเดียว</div>

<h2>6. ไฟล์ผลลัพธ์และ log</h2>
{t_files}
<p class="sub">คำสั่งรันซ้ำทั้ง pipeline: <code>python main.py</code> ·
เวลารวม {elapsed} วินาที</p>
""".format(css=CSS, run=RUN_ID, now=datetime.now().strftime("%d/%m/%Y %H:%M"),
           rows=ins["rows"], nfeat=sc["features"], fpct=ins["fraud_pct"],
           ratio=ins["imbalance_ratio"], acc=100 - ins["fraud_pct"],
           t_check=t_check, t_steps=t_steps, t_bal=t_bal, t_m_tbl=t_m,
           t_files=t_files, skew=ins["amount_skew"], a_c=sc["amount_center"],
           a_s=sc["amount_scale"], trange=ins["time_range_h"],
           t_m=sc["time_mean"], t_s=sc["time_std"], dropped=dropped, extra=extra,
           k=sm["k_neighbors"], ratio_sm=sm["sampling_strategy"], best=best,
           img_amount=imgs["amount"], img_hour=imgs["hour"],
           img_balance=imgs["balance"], img_metrics=imgs["metrics"],
           elapsed=S["elapsed"],
           pr_lo=min(r["pr_auc"] for r in res), pr_hi=max(r["pr_auc"] for r in res),
           r_base=res[0]["recall"], r_bal=res[1]["recall"], r_smote=res[2]["recall"],
           tp_base=res[0]["tp"], tp_bal=res[1]["tp"], fp_base=res[0]["fp"],
           fp_bal=res[1]["fp"], n_fraud=sp["test_fraud"])


def to_pdf(html_path, pdf_path):
    exe = next((c for c in CHROME if Path(c).exists()), None) or shutil.which("chrome")
    if not exe:
        log.warning("ไม่พบ Chrome/Edge — ข้ามการสร้าง PDF (ยังมีไฟล์ HTML ให้เปิดดู)")
        return False
    cmd = [exe, "--headless=new", "--disable-gpu", "--no-sandbox",
           "--no-pdf-header-footer", "--print-to-pdf=%s" % pdf_path,
           html_path.as_uri()]
    subprocess.run(cmd, capture_output=True, timeout=240)
    for _ in range(30):
        if pdf_path.exists() and pdf_path.stat().st_size > 1000:
            log.info("สร้าง PDF สำเร็จ -> outputs/%s (%.0f KB)",
                     pdf_path.name, pdf_path.stat().st_size / 1024)
            return True
        time.sleep(0.5)
    log.error("สร้าง PDF ไม่สำเร็จ")
    return False


def generate(S, imgs):
    log.info("STEP 10 สร้างรายงาน HTML และ PDF")
    (OUT_DIR / "summary.json").write_text(
        json.dumps(S, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("บันทึก outputs/summary.json")
    html = OUT_DIR / "report.html"
    html.write_text("<!doctype html><meta charset='utf-8'>" + build_html(S, imgs),
                    encoding="utf-8")
    log.info("เขียน outputs/report.html (%.0f KB)", html.stat().st_size / 1024)
    to_pdf(html, OUT_DIR / "report.pdf")
    return html
