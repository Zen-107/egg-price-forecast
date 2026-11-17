# Data

## `raw/` — ข้อมูลต้นฉบับ (ไม่แก้ไข)

| ไฟล์ | ที่มา | เนื้อหา |
|---|---|---|
| `average_prices.parquet` | U.S. Bureau of Labor Statistics, Average Price Data (ชุดรวมบน Kaggle: `sadiqguru/us-bls-average-prices`) | ราคาขายปลีกเฉลี่ยรายเดือนของสินค้าทุกรายการ ทุกพื้นที่ 356,549 แถว (1980-01 → 2026-09) ใช้เฉพาะ series `APU0000708111` = *Eggs, grade A, large, per doz.* / *U.S. city average* |
| `average_prices.csv` | เหมือนข้างบน | ข้อมูลเดียวกับ parquet แต่ใหญ่ 52 MB จึง **ไม่ได้อัปขึ้น GitHub** (โค้ดใช้ parquet) |
| `series.csv`, `items.csv`, `areas.csv`, `footnotes.csv` (+ `.parquet`) | BLS / Kaggle | ตารางอ้างอิง: รหัส series, ชื่อสินค้า, พื้นที่, หมายเหตุ (`X` = ไม่มีข้อมูลช่วง government shutdown 2025) |
| `schema.json` | Kaggle | ชนิดข้อมูลของแต่ละคอลัมน์ |
| `nass_layers.csv` | USDA NASS Quick Stats (Survey › Poultry › Chickens › Inventory, National, Monthly) | จำนวนแม่ไก่ไข่รายเดือน ใช้ชุด `CHICKENS, LAYERS, TABLE - INVENTORY, AVG, MEASURED IN HEAD`, Domain `TOTAL` (2007-12 → 2025-12) |
| `hpai_flocks.csv` | USDA APHIS, *Confirmed HPAI Detections in Commercial and Backyard Flocks* (ExportToCsv) | ฟาร์มที่ยืนยันพบไข้หวัดนก 1 แถว = 1 ฟาร์ม (2,307 แถว, 2022-02-08 → 2026-11-10) |

## `processed/` — ข้อมูลหลังทำความสะอาด (สร้างโดย `src/preprocess.py` และ `src/analysis.py`)

| ไฟล์ | คำอธิบาย |
|---|---|
| `egg_price_monthly.csv` | ราคาไข่ (USD/โหล) รายเดือนครบทุกเดือน 561 เดือน เดือน 2025-10 ที่ว่างเติมด้วย linear interpolation |
| `layers_monthly.csv` | จำนวนแม่ไก่ไข่ (ตัว) รายเดือน |
| `hpai_monthly.csv` | ไก่ไข่ (ฟาร์มประเภท Table Egg) ที่ติดไข้หวัดนกต่อเดือน (ล้านตัว) ก่อน 2016-07 = ว่าง (ไม่มีข้อมูล), 2016-07 → 2022-01 = 0 (สมมติฐาน) |
| `features.csv` | ตาราง feature + target ที่ใช้เทรนโมเดล (log-return, ความผันผวน, ฤดูกาล, แม่ไก่ lag 1 เดือน, ไข้หวัดนก) |
| `data_issues.csv` | สรุปปัญหาข้อมูลที่พบและวิธีจัดการ |

## ปัญหาข้อมูลหลักและวิธีจัดการ

- ราคาเดือน 2025-10 ว่าง (BLS หยุดเก็บช่วง government shutdown) → `asfreq('MS')` แล้ว interpolate
- ราคามีเทรนด์ระยะยาว (เงินเฟ้อ) → ทาย % การเปลี่ยนแปลง (log-return) แทนระดับราคา
- NASS ประกาศตัวเลขช้า ~1 เดือน → ใช้ข้อมูลแม่ไก่ช้ากว่าราคา 1 เดือน เพื่อไม่ให้ข้อมูลอนาคตรั่ว
- ไฟล์ไข้หวัดนกเป็นรายฟาร์มหลายประเภทนก → เลือกเฉพาะฟาร์มไก่ไข่แล้วรวมรายเดือน
