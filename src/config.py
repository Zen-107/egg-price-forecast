"""ค่าคงที่และ path ที่ใช้ร่วมกันทุกสคริปต์ (แก้การทดลองที่นี่ที่เดียว)"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / 'data' / 'raw'
PROCESSED_DIR = ROOT / 'data' / 'processed'
FIG_DIR = ROOT / 'results' / 'figures'
OUT_DIR = ROOT / 'results' / 'output'

# ข้อมูลดิบ
PRICES_FILE = 'average_prices.parquet'      # BLS Average Price Data (มี .csv ด้วย แต่ใหญ่ 50 เท่า)
LAYERS_FILE = 'nass_layers.csv'             # USDA NASS: จำนวนแม่ไก่ไข่รายเดือน
HPAI_FILE = 'hpai_flocks.csv'               # USDA APHIS: ฟาร์มที่ติดไข้หวัดนก (1 แถว = 1 ฟาร์ม)
ITEM_PREFIX = 'Eggs, grade A, large'        # ไข่ Grade A ขนาดใหญ่ ต่อโหล
AREA = 'U.S. city average'
LAYERS_ITEM = 'CHICKENS, LAYERS, TABLE - INVENTORY, AVG, MEASURED IN HEAD'

# การทดลอง
TEST_START = '2019-01-01'     # การทดลอง A: walk-forward ตั้งแต่ 2019 → ล่าสุด
SPIKE = 0.10                  # "ราคาพุ่ง" = เดือนหน้าขึ้นเกิน 10%
WARN = 0.05                   # โมเดล "เตือน" ถ้าทายว่าขึ้นเกิน 5%
VOL_Q = 0.75                  # "ช่วงผันผวน" = vol_6 > percentile 75 ของช่วง train
SEED = 42
TEST_B_START = '2022-06-01'   # การทดลอง B: เริ่มหลังข้อมูลไข้หวัดนกเริ่ม 4 เดือน
SELECT_END = '2024-05-01'     # ช่วงคัดเลือกตัวแปร: TEST_B_START → SELECT_END | ช่วงประเมิน: หลังจากนั้น
MIN_GAIN = 0.02               # เก็บกลุ่มตัวแปรถ้า MASE ช่วงคัดเลือกดีขึ้นอย่างน้อยเท่านี้
LAYERS_LAG = 1                # NASS ประกาศช้า ~1 เดือน → ใช้ข้อมูลแม่ไก่ช้ากว่าราคา 1 เดือน
