"""ขั้น 4-9: EDA, สร้าง features, โมเดล (walk-forward) และวิเคราะห์ความผิดพลาด

อ่านข้อมูลจาก data/processed แล้วบันทึก
  กราฟ   → results/figures/*.png
  ตาราง  → results/output/*.csv และ summary.txt
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C

plt.rcParams.update({'figure.figsize': (12, 4.5), 'axes.grid': True, 'grid.alpha': 0.3})

FEATS = ['ret_0', 'ret_1', 'ret_2', 'vol_3', 'dev_12', 'month_sin', 'month_cos']
GROUPS = {'layers': ['lay_chg_1', 'lay_chg_3', 'lay_dev_12'], 'HPAI': ['hpai_0', 'hpai_1', 'hpai_3m']}
MODELS = {
    'Ridge': lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    'GBM': lambda: GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05,
                                             subsample=0.8, random_state=C.SEED),
}


# ---------------------------------------------------------------- data
def load_processed():
    rd = lambda f: pd.read_csv(C.PROCESSED_DIR / f, index_col='date', parse_dates=True).iloc[:, 0]
    price = rd('egg_price_monthly.csv').asfreq('MS')
    layers = rd('layers_monthly.csv')
    hpai = rd('hpai_monthly.csv').reindex(price.index)
    return price, layers, hpai


def save(fig, name):
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / name, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- EDA (R3)
def eda(price, layers, hpai):
    ret = np.log(price).diff()
    big = ret.abs() > C.SPIKE
    notes = []

    # 4.1 trend
    fig, ax = plt.subplots()
    ax.plot(price.index, price, color='#c0392b', lw=1.6, label='Avg price, eggs grade A large (USD/dozen)')
    ax.scatter(price.index[big], price[big], s=14, color='black', zorder=3, label=f'|change| > {C.SPIKE:.0%}')
    ax.axvline(pd.Timestamp(C.TEST_START), color='grey', ls='--', lw=1)
    ax.set_title('U.S. average retail egg price'); ax.set_ylabel('USD per dozen'); ax.legend(loc='upper left')
    save(fig, '01_price_history.png')
    dec = price.groupby((price.index.year // 10) * 10).mean()
    notes.append(f'Decade mean price grew {dec.iloc[-1] / dec.iloc[0]:.1f}x since the 1980s -> predict log-return, not level')

    # 4.2 distribution of monthly changes
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(ret.dropna() * 100, bins=60, color='#2c7fb8'); axes[0].set_title('Monthly price change (%)')
    share = big.groupby(ret.index.year // 5 * 5).mean() * 100
    axes[1].bar(share.index.astype(str), share.values, color='#e6550d')
    axes[1].set_title(f'Share of months with |change| > {C.SPIKE:.0%}, per 5-year block')
    axes[1].tick_params(axis='x', rotation=45)
    save(fig, '02_monthly_change.png')
    notes.append(f'Std of monthly change = {ret.std() * 100:.1f}%, {big.sum()} months with |change| > {C.SPIKE:.0%}')

    # 4.3 volatility clustering
    nxt = big.shift(-1)
    valid = ret.notna() & nxt.notna() & ret.shift(-1).notna()
    p_big, p_quiet = nxt[valid & big].astype(float).mean(), nxt[valid & ~big].astype(float).mean()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(['after a calm month', f'after |change| > {C.SPIKE:.0%}'], [p_quiet * 100, p_big * 100],
           color=['#9ecae1', '#e6550d'])
    ax.set_ylabel('% of next months also > 10%'); ax.set_title('Volatility clustering')
    save(fig, '03_volatility_clustering.png')
    notes.append(f'Volatility clusters: P(big next | big now) = {p_big:.0%} vs {p_quiet:.0%} after calm month')

    # 4.4 seasonality
    season = ret.groupby(ret.index.month).agg(['mean', 'std', 'count'])
    se = season['std'] / np.sqrt(season['count'])
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(season.index, season['mean'] * 100, yerr=1.96 * se * 100, capsize=3,
           color=np.where(season['mean'] > 0, '#e6550d', '#3182bd'))
    ax.set_xticks(range(1, 13)); ax.set_title('Average monthly price change by calendar month (95% CI)')
    save(fig, '04_seasonality.png')

    # 4.5 autocorrelation
    lags = range(1, 13)
    acf = [ret.autocorr(lag=k) for k in lags]
    band = 1.96 / np.sqrt(ret.notna().sum())
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(lags, acf, color='#756bb1'); ax.axhline(band, ls='--', c='k'); ax.axhline(-band, ls='--', c='k')
    ax.set_xticks(list(lags)); ax.set_title('Autocorrelation of monthly price change'); ax.set_xlabel('lag (months)')
    save(fig, '05_autocorrelation.png')
    notes.append('Significant ACF lags: ' + str([k for k, a in zip(lags, acf) if abs(a) > band]))

    # 4.6 layers vs price
    lay_ret = np.log(layers).diff().reindex(ret.index)
    xc = {k: ret.corr(lay_ret.shift(k)) for k in range(0, 5)}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(layers.index, layers / 1e6, color='#31a354'); axes[0].set_ylabel('Table-egg layers (million)')
    ax2 = axes[0].twinx(); ax2.plot(price.loc[layers.index[0]:], color='#c0392b', lw=1.2)
    ax2.set_ylabel('Egg price (USD/dozen)'); axes[0].set_title('Layers (green) vs egg price (red)')
    axes[1].bar(list(xc), list(xc.values()), color='#31a354'); axes[1].axhline(0, c='k', lw=0.8)
    axes[1].set_title('corr(layer change t-k, price change t)'); axes[1].set_xlabel('k (months)')
    save(fig, '06_layers_vs_price.png')

    # 4.7 HPAI vs price
    xh = np.log1p(hpai)
    win = ret.index >= '2022-02-01'
    xc_h = {k: ret[win].corr(xh.shift(k)[win]) for k in range(0, 5)}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].bar(hpai[win].index, hpai[win], width=25, color='purple', alpha=0.6)
    axes[0].set_ylabel('Egg-type birds lost to HPAI (million)')
    ax2 = axes[0].twinx(); ax2.plot(price[win], color='#c0392b'); ax2.set_ylabel('Egg price (USD/dozen)')
    axes[0].set_title('HPAI in egg flocks (bars) vs egg price (line)')
    axes[1].bar(list(xc_h), list(xc_h.values()), color='purple')
    axes[1].set_title('corr(HPAI birds lost t-k, price change t)'); axes[1].set_xlabel('k (months)')
    save(fig, '07_hpai_vs_price.png')
    notes.append('corr(layers lag k) = ' + str({k: round(float(v), 2) for k, v in xc.items()}))
    notes.append('corr(HPAI lag k)   = ' + str({k: round(float(v), 2) for k, v in xc_h.items()}))
    return notes


# ---------------------------------------------------------------- features & evaluation
def make_features(price, layers, hpai):
    r = np.log(price).diff()
    X = pd.DataFrame(index=price.index)
    X['ret_0'], X['ret_1'], X['ret_2'] = r, r.shift(1), r.shift(2)
    X['vol_3'] = r.rolling(3).std()
    X['vol_6'] = r.rolling(6).std()               # ใช้แบ่งช่วงปกติ/ผันผวน (ไม่ใช่ feature)
    X['dev_12'] = np.log(price / price.rolling(12).mean())
    X['month_sin'] = np.sin(2 * np.pi * price.index.month / 12)
    X['month_cos'] = np.cos(2 * np.pi * price.index.month / 12)
    lay_log = np.log(layers).reindex(price.index).shift(C.LAYERS_LAG)
    X['lay_chg_1'] = lay_log.diff(1)
    X['lay_chg_3'] = lay_log.diff(3)
    X['lay_dev_12'] = lay_log - lay_log.rolling(12).mean()
    X['hpai_0'] = np.log1p(hpai)
    X['hpai_1'] = X['hpai_0'].shift(1)
    X['hpai_3m'] = np.log1p(hpai.rolling(3).sum())
    X['target'] = r.shift(-1)                     # log-return เดือนหน้า
    X['price'] = price
    X['price_next'] = price.shift(-1)
    # leakage check: target[t-1] ต้องเท่ากับ ret_0[t]
    assert (X['target'].shift(1) - X['ret_0']).abs().max() < 1e-12
    return X


def walk_forward(data, feats, make_model, test_idx):
    """ทุกเดือน t เทรนใหม่ด้วยแถวก่อน t เท่านั้น แล้วทาย 1 เดือนข้างหน้า"""
    preds = {}
    for t in test_idx:
        train = data[data.index < t].dropna(subset=feats + ['target'])
        model = make_model().fit(train[feats], train['target'])
        preds[t] = model.predict(data.loc[[t], feats])[0]
    return pd.Series(preds)


def evaluate(pred_ret, d):
    actual, pred = d['price_next'], d['price'] * np.exp(pred_ret)
    mae = (actual - pred).abs().mean()
    spikes = d['target'] > C.SPIKE
    return {
        'MAE ($/doz)': round(mae, 3),
        'RMSE ($/doz)': round(np.sqrt(((actual - pred) ** 2).mean()), 3),
        'MASE': round(mae / (actual - d['price']).abs().mean(), 3),
        'Spike recall': f'{(pred_ret[spikes] > C.WARN).sum()}/{spikes.sum()}',
        'False alarms': int(((pred_ret > C.WARN) & ~spikes).sum()),
    }


# ---------------------------------------------------------------- experiment A (R4): price only
def experiment_a(data):
    test_idx = data.index[(data.index >= C.TEST_START) & data['target'].notna()]
    dT = data.loc[test_idx]
    preds = {'Naive': pd.Series(0.0, index=test_idx)}
    for name, mk in MODELS.items():
        preds[name] = walk_forward(data, FEATS, mk, test_idx)
    res = pd.DataFrame({k: evaluate(v, dT) for k, v in preds.items()}).T
    best = res['MASE'].astype(float).drop('Naive').idxmin()
    res.to_csv(C.OUT_DIR / 'model_comparison_A.csv', index_label='model')

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    for ax, start in zip(axes, [test_idx[0], test_idx[-1] - pd.DateOffset(months=36)]):
        d = dT[dT.index >= start]
        x = d.index + pd.offsets.MonthBegin(1)
        ax.plot(x, d['price_next'], 'k-o', ms=3, label='Actual')
        ax.plot(x, d['price'], '--', c='grey', label='Naive (last month)')
        for name, c in [('Ridge', '#3182bd'), ('GBM', '#e6550d')]:
            ax.plot(x, d['price'] * np.exp(preds[name].loc[d.index]), '-x', c=c, ms=4, label=name)
        ax.set_ylabel('USD per dozen'); ax.legend(loc='upper left')
    axes[0].set_title('Next-month egg price: actual vs forecast (walk-forward)'); axes[1].set_title('Zoom: last 3 years')
    save(fig, '08_forecast_vs_actual.png')

    fig, ax = plt.subplots(figsize=(8, 3))
    m = res['MASE'].astype(float)
    ax.barh(m.index, m.values, color=['grey' if i == 'Naive' else ('#e6550d' if i == best else '#3182bd') for i in m.index])
    ax.axvline(1, c='k', ls='--'); ax.invert_yaxis()
    for i, v in enumerate(m.values):
        ax.text(v + 0.01, i, f'{v:.2f}', va='center')
    ax.set_title('MASE (lower is better, < 1 beats naive)')
    save(fig, '09_mase_A.png')

    pd.DataFrame(preds).assign(price=dT['price'], actual_next=dT['price_next']) \
        .to_csv(C.OUT_DIR / 'predictions_A.csv', index_label='date')
    return test_idx, dT, preds, res, best


# ---------------------------------------------------------------- experiment B: + layers / HPAI
def experiment_b(data, preds_a):
    sets = {'price only': FEATS, '+ layers': FEATS + GROUPS['layers'], '+ HPAI': FEATS + GROUPS['HPAI'],
            '+ layers + HPAI': FEATS + GROUPS['layers'] + GROUPS['HPAI']}
    test_b = data.index[(data.index >= C.TEST_B_START) & data['target'].notna()]
    pred_b = {'Naive': pd.Series(0.0, index=test_b)}
    for set_name, feats in sets.items():
        for m, mk in MODELS.items():
            key = f'{m} {set_name}'
            if set_name == 'price only':
                pred_b[key] = preds_a[m].reindex(test_b)
            else:
                idx = test_b[data.loc[test_b, feats].notna().all(axis=1)]
                pred_b[key] = walk_forward(data, feats, mk, idx).reindex(test_b)

    compare = lambda idx: pd.DataFrame({k: evaluate(v.loc[idx], data.loc[idx]) for k, v in pred_b.items()}).T
    common = test_b[pd.concat(pred_b, axis=1).notna().all(axis=1)]
    sel_idx, eval_idx = common[common <= C.SELECT_END], common[common > C.SELECT_END]

    res_sel = compare(sel_idx)
    mase_sel = res_sel['MASE'].astype(float)
    best_of = lambda s: mase_sel[[f'{m} {s}' for m in MODELS]].min()
    gain = {g: round(float(best_of('price only') - best_of(f'+ {g}')), 3) for g in GROUPS}
    keep = [g for g, v in gain.items() if v >= C.MIN_GAIN]
    sel_name = 'price only' if not keep else '+ ' + ' + '.join(keep)
    sel_model = min(MODELS, key=lambda m: mase_sel[f'{m} {sel_name}'])
    sel_key = f'{sel_model} {sel_name}'

    res_eval = compare(eval_idx)
    full_idx = test_b[(test_b > C.SELECT_END) & pred_b[sel_key].notna()]
    res_final = compare(full_idx).loc[['Naive', 'GBM price only', 'Ridge price only', sel_key]].drop_duplicates()
    res_sel.to_csv(C.OUT_DIR / 'model_comparison_B_selection.csv', index_label='model')
    res_eval.to_csv(C.OUT_DIR / 'model_comparison_B_evaluation.csv', index_label='model')
    res_final.to_csv(C.OUT_DIR / 'model_comparison_B_final.csv', index_label='model')

    # feature importance (train up to SELECT_END only)
    all_f = sets['+ layers + HPAI']
    tr = data[data.index <= C.SELECT_END].dropna(subset=all_f + ['target'])
    gbm = MODELS['GBM']().fit(tr[all_f], tr['target'])
    rdg = MODELS['Ridge']().fit(tr[all_f], tr['target'])
    imp = pd.DataFrame({'GBM importance': gbm.feature_importances_, 'Ridge |coef|': np.abs(rdg[-1].coef_)}, index=all_f)
    imp = imp / imp.sum()
    imp.to_csv(C.OUT_DIR / 'feature_importance_B.csv', index_label='feature')
    grp = imp.groupby(lambda f: 'layers' if f.startswith('lay') else ('HPAI' if f.startswith('hpai') else 'price')).sum()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    colors = ['purple' if f.startswith('hpai') else ('#31a354' if f.startswith('lay') else '#3182bd') for f in imp.index]
    imp['GBM importance'].plot.barh(ax=axes[0], color=colors); axes[0].invert_yaxis()
    axes[0].set_title('GBM feature importance (share)')
    grp.plot.bar(ax=axes[1], rot=0, color=['#3182bd', '#e6550d']); axes[1].set_title('Importance share by group')
    save(fig, '10_feature_importance_B.png')

    d = data.loc[test_b]; x = d.index + pd.offsets.MonthBegin(1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), gridspec_kw={'width_ratios': [2, 1]})
    ax = axes[0]
    ax.plot(x, d['price_next'], 'k-o', ms=3, label='Actual')
    ax.plot(x, d['price'], '--', c='grey', label='Naive')
    ax.plot(x, d['price'] * np.exp(pred_b['GBM price only']), '-x', c='#3182bd', ms=4, label='GBM price only')
    if sel_key != 'GBM price only':
        ax.plot(x, d['price'] * np.exp(pred_b[sel_key]), '-x', c='#e6550d', ms=4, label=f'{sel_key} (selected)')
    ax.axvline(pd.Timestamp(C.SELECT_END) + pd.offsets.MonthBegin(1), c='k', ls=':')
    ax.set_title('Next-month price: selection | evaluation period'); ax.set_ylabel('USD per dozen'); ax.legend(loc='upper left')
    m = res_eval['MASE'].astype(float)
    axes[1].barh(m.index, m.values, color=['grey' if i == 'Naive' else ('#e6550d' if i == sel_key else '#9ecae1') for i in m.index])
    axes[1].axvline(1, c='k', ls='--'); axes[1].invert_yaxis()
    for j, v in enumerate(m.values):
        axes[1].text(v + 0.01, j, f'{v:.2f}', va='center')
    axes[1].set_title('MASE, evaluation period (< 1 beats naive)')
    save(fig, '11_experiment_B.png')
    return dict(sel_idx=sel_idx, full_idx=full_idx, gain=gain, sel_key=sel_key, res_final=res_final)


# ---------------------------------------------------------------- error analysis (R5, R6)
def error_analysis(data, test_idx, dT, preds, best):
    vol_cut = data.loc[data.index < C.TEST_START, 'vol_6'].quantile(C.VOL_Q)
    volatile = (dT['vol_6'] > vol_cut).values
    rows = {}
    for name, pr in preds.items():
        e = (dT['price_next'] - dT['price'] * np.exp(pr)).abs()
        rows[name] = {'MAE normal': round(e[~volatile].mean(), 3), 'MAE volatile': round(e[volatile].mean(), 3)}
    seg = pd.DataFrame(rows).T
    seg['MASE normal'] = (seg['MAE normal'] / seg.loc['Naive', 'MAE normal']).round(3)
    seg['MASE volatile'] = (seg['MAE volatile'] / seg.loc['Naive', 'MAE volatile']).round(3)
    seg.to_csv(C.OUT_DIR / 'error_by_regime.csv', index_label='model')

    pr = preds[best]
    err = pd.DataFrame({
        'price_t': dT['price'].round(2), 'actual_next': dT['price_next'].round(2),
        'pred_next': (dT['price'] * np.exp(pr)).round(2),
        'actual_change_%': (dT['target'] * 100).round(1), 'pred_change_%': (pr * 100).round(1),
    })
    err['abs_error'] = (err['actual_next'] - err['pred_next']).abs()
    worst = err.sort_values('abs_error', ascending=False).head(10)
    worst.to_csv(C.OUT_DIR / 'worst_10_months.csv', index_label='date')
    under = (worst['pred_next'] < worst['actual_next']).mean()

    direction = pd.cut(dT['target'], [-1, -C.SPIKE, C.SPIKE, 1], labels=['drop >10%', 'normal', 'spike >10%'])
    tbl = err.groupby(direction, observed=False)['abs_error'].agg(['count', 'mean']).round(3)
    tbl['naive_mae'] = (dT['price_next'] - dT['price']).abs().groupby(direction, observed=False).mean().round(3)
    tbl.to_csv(C.OUT_DIR / 'error_by_direction.csv', index_label='direction')

    pct = dT['price_next'] / (dT['price'] * np.exp(pr)) - 1
    band = pd.DataFrame({'all': pct.quantile([0.1, 0.9]), 'normal': pct[~volatile].quantile([0.1, 0.9]),
                         'volatile': pct[volatile].quantile([0.1, 0.9])}).T.mul(100).round(1)
    band.columns = ['P10 (%)', 'P90 (%)']
    band.to_csv(C.OUT_DIR / 'error_band.csv', index_label='regime')

    train = data[data.index < test_idx[-1]].dropna(subset=FEATS + ['target'])
    ridge = MODELS['Ridge']().fit(train[FEATS], train['target'])
    coef = pd.Series(ridge[-1].coef_, index=FEATS).sort_values()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(coef.index, coef.values, color=np.where(coef > 0, '#e6550d', '#3182bd'))
    ax.set_title('Ridge coefficients (standardized features)')
    save(fig, '12_ridge_coefficients.png')
    return seg, under, band, vol_cut, volatile


# ---------------------------------------------------------------- main entry
def run():
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    price, layers, hpai = load_processed()

    notes = eda(price, layers, hpai)
    print('[analysis] EDA figures saved')

    data = make_features(price, layers, hpai)
    data.to_csv(C.PROCESSED_DIR / 'features.csv', index_label='date')

    test_idx, dT, preds, res, best = experiment_a(data)
    print(f'[analysis] experiment A done ({len(test_idx)} test months), best = {best}')
    b = experiment_b(data, preds)
    print(f'[analysis] experiment B done, selected = {b["sel_key"]}')
    seg, under, band, vol_cut, volatile = error_analysis(data, test_idx, dT, preds, best)

    verdict = lambda m: 'beats baseline' if m < 0.95 else ('worse than baseline' if m > 1.05 else 'close to baseline (<5%)')
    lines = ['=== EDA (R3) ===', *notes,
             '=== Experiment A: price only (R4) ===',
             f'Test {len(test_idx)} months ({test_idx[0]:%Y-%m} -> {test_idx[-1]:%Y-%m}), walk-forward',
             res.to_string(),
             f'Best model: {best} (MASE {res.loc[best, "MASE"]}) -> {verdict(float(res.loc[best, "MASE"]))}',
             '=== Experiment B: + layers / HPAI ===',
             f'Selection {b["sel_idx"][0]:%Y-%m} -> {b["sel_idx"][-1]:%Y-%m}: MASE gain vs price only = {b["gain"]} -> selected {b["sel_key"]}',
             f'Evaluation {b["full_idx"][0]:%Y-%m} -> {b["full_idx"][-1]:%Y-%m} ({len(b["full_idx"])} months):',
             b['res_final'].to_string(),
             '=== Error analysis (R5) ===',
             f'Volatile = vol_6 > {vol_cut * 100:.1f}% | normal {(~volatile).sum()} months, volatile {volatile.sum()} months',
             seg.to_string(),
             f'Top-10 worst months of {best}: under-predicted {under:.0%}',
             '=== Budget band (R6) ===',
             band.to_string()]
    summary = '\n'.join(lines)
    (C.OUT_DIR / 'summary.txt').write_text(summary, encoding='utf-8')
    print(summary)


if __name__ == '__main__':
    run()
