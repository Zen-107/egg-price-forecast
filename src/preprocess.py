"""ขั้น 2-3: โหลดข้อมูลดิบจาก data/raw ทำความสะอาด แล้วบันทึกลง data/processed

ผลลัพธ์:
  egg_price_monthly.csv  ราคาไข่รายเดือน (เติมเดือนที่ว่างด้วย linear interpolation)
  layers_monthly.csv     จำนวนแม่ไก่ไข่รายเดือน (USDA NASS)
  hpai_monthly.csv       ไก่ไข่ที่ติดไข้หวัดนกต่อเดือน (ล้านตัว, USDA APHIS)
  data_issues.csv        สรุปปัญหาข้อมูลและวิธีจัดการ
"""
import numpy as np
import pandas as pd

import config as C

MONTHS = {m: k for k, m in enumerate(
    ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'], 1)}


def load_eggs():
    """ราคาไข่ Grade A large ค่าเฉลี่ยทั้งประเทศ จากไฟล์ BLS ทั้งชุด"""
    path = C.RAW_DIR / C.PRICES_FILE
    if not path.exists():
        path = C.RAW_DIR / 'average_prices.csv'
    raw = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path, low_memory=False)
    eggs = raw[raw['item_name'].str.startswith(C.ITEM_PREFIX) & (raw['area_name'] == C.AREA)].copy()
    eggs['date'] = pd.to_datetime(eggs['date'])
    return raw, eggs.sort_values('date').reset_index(drop=True)


def clean_price(eggs):
    """บังคับให้มีครบทุกเดือน แล้ว interpolate เดือนที่ราคาว่าง (เช่น 2025-10 ช่วง government shutdown)"""
    assert not eggs['date'].duplicated().any(), 'พบวันที่ซ้ำในข้อมูลราคา'
    price = eggs.set_index('date')['price_usd'].asfreq('MS')
    imputed = price.index[price.isna()]
    price = price.interpolate(method='linear')
    price.name = 'price_usd'
    return price, imputed


def load_layers():
    raw = pd.read_csv(C.RAW_DIR / C.LAYERS_FILE)
    lay = raw[(raw['Data Item'] == C.LAYERS_ITEM) & (raw['Domain'] == 'TOTAL')].copy()
    lay['date'] = pd.to_datetime(dict(year=lay['Year'], month=lay['Period'].map(MONTHS), day=1))
    lay['value'] = pd.to_numeric(lay['Value'].astype(str).str.replace(',', ''), errors='coerce')
    layers = lay.set_index('date')['value'].sort_index()
    layers.name = 'layers_head'
    return raw, layers


def load_hpai(index):
    """ฟาร์มที่ติดเชื้อรายฟาร์ม → จำนวนไก่ไข่ที่ติดเชื้อต่อเดือน (ล้านตัว)

    ไฟล์ APHIS เริ่มที่การระบาดรอบ 2022 (2022-02-08):
      2016-07 → 2022-01 ถือว่า 0 (สมมติฐาน) | ก่อน 2016-07 ไม่รู้ → NaN
    """
    raw = pd.read_csv(C.RAW_DIR / C.HPAI_FILE)
    hp = raw.copy()
    hp['date'] = pd.to_datetime(hp['Confirmed Diagnosis'], errors='coerce')
    hp['birds'] = pd.to_numeric(hp['Birds Affected (Table)'].astype(str).str.replace(',', ''), errors='coerce')
    egg = hp[hp['Production'].str.contains('Table Egg', case=False, na=False)]
    monthly = egg.groupby(egg['date'].dt.to_period('M'))['birds'].sum()
    monthly.index = monthly.index.to_timestamp()
    hpai = monthly.reindex(index).fillna(0.0) / 1e6
    hpai[hpai.index < '2016-07-01'] = np.nan
    hpai.name = 'egg_birds_lost_million'
    return raw, hp, egg, hpai


def run():
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw, eggs = load_eggs()
    price, imputed = clean_price(eggs)
    layers_raw, layers = load_layers()
    hpai_raw, hp, egg, hpai = load_hpai(price.index)

    price.to_frame().to_csv(C.PROCESSED_DIR / 'egg_price_monthly.csv', index_label='date')
    layers.to_frame().to_csv(C.PROCESSED_DIR / 'layers_monthly.csv', index_label='date')
    hpai.to_frame().to_csv(C.PROCESSED_DIR / 'hpai_monthly.csv', index_label='date')

    issues = pd.DataFrame([
        ['Missing price month(s) (BLS did not collect during government shutdown)', len(imputed), 'asfreq + linear interpolation'],
        ['Only ~1 row per month', len(price), 'Small models (Ridge, shallow GBM) + walk-forward'],
        ['Long-run price trend (inflation)', '1980-2026', 'Target = next-month log-return'],
        ['NASS layers only cover part of the period', len(layers), 'Separate experiment B'],
        ['NASS published ~1 month late', '-', f'Lag layers by {C.LAYERS_LAG} month'],
        ['NASS file mixes several series', len(layers_raw), 'Keep LAYERS, TABLE + Domain TOTAL'],
        ['APHIS HPAI file starts 2022-02', '-', 'Before 2016-07 = NaN, 2016-07..2022-01 = 0'],
        ['HPAI is per farm, many bird types', len(hpai_raw), 'Keep Table Egg farms, sum per month'],
    ], columns=['issue', 'size', 'handling'])
    issues.to_csv(C.PROCESSED_DIR / 'data_issues.csv', index=False)

    print(f'[preprocess] BLS file: {len(raw):,} rows -> egg price {len(price)} months '
          f'({price.index[0]:%Y-%m} -> {price.index[-1]:%Y-%m}), imputed {[f"{d:%Y-%m}" for d in imputed]}')
    print(f'[preprocess] NASS layers: {len(layers)} months ({layers.index[0]:%Y-%m} -> {layers.index[-1]:%Y-%m})')
    print(f'[preprocess] APHIS HPAI: {len(hpai_raw):,} farms, {len(egg)} table-egg farms, '
          f'{egg.birds.sum() / 1e6:.1f}M birds')
    print(f'[preprocess] saved to {C.PROCESSED_DIR.relative_to(C.ROOT)}')
    return price, layers, hpai


if __name__ == '__main__':
    run()
