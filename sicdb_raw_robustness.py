from __future__ import annotations
import csv
import gzip
import json
from pathlib import Path
from config import DUCKDB_TEMP, HIRID_MAP_GLOB, HIRID_ROOT, RESULTS_DIR, SICDB_ROOT
import numpy as np
import pandas as pd
import statsmodels.api as sm
HERE = Path(__file__).resolve().parent
REV = HERE / 'results' / 'revision'
ROOT = SICDB_ROOT
OUT = REV / 'sicdb_raw_robustness_results.json'

def decode(raw: str, val: str, offset: int):
    if raw and raw.startswith('0x') and (len(raw) == 482):
        try:
            v = np.frombuffer(bytes.fromhex(raw[2:]), dtype='<f4').astype(float)
            t = offset + 60 * np.arange(60)
            ok = (v != 0) & np.isfinite(v)
            return (t[ok], v[ok])
        except ValueError:
            pass
    try:
        x = float(val)
        return (np.array([offset]), np.array([x])) if np.isfinite(x) else (np.array([]), np.array([]))
    except (TypeError, ValueError):
        return (np.array([]), np.array([]))

def match_index() -> pd.DataFrame:
    idx = pd.read_parquet(REV / 'sicdb_sham_index_raw.parquet')
    idx = idx[idx.pre_n >= 3].copy()
    real = idx[idx.grp == 'real'].set_index('rid')
    cand = idx[idx.grp == 'cand'].copy()
    cand = cand.join(real[['pre_mean', 'pre_slope', 'pre_temp', 'pressor', 'DrugID', 'case']], on='rid', rsuffix='_r')
    cand = cand[cand.pressor_r.notna()]
    cand['dist'] = (cand.pre_mean - cand.pre_mean_r).abs() / 5 + (cand.pre_slope - cand.pre_slope_r).abs() * 60 / 2 + (cand.pre_temp - cand.pre_temp_r).abs().fillna(0) + 10 * (cand.pressor != cand.pressor_r) + (cand.offset_h.abs() - 24).abs() / 2
    cand = cand[cand.b30.notna() | cand.b60.notna()]
    best = cand.sort_values('dist').groupby('rid').head(1).copy()
    r = real.reset_index()
    r = r[r.rid.astype(int).isin(set(best.rid.astype(int)))].copy()
    r['grp'] = 'real'
    b = best.copy()
    b['grp'] = 'sham'
    b['DrugID'] = b['DrugID_r']
    keep = ['iid', 'grp', 'rid', 'case', 't0', 'DrugID']
    return pd.concat([r[keep], b[keep]], ignore_index=True)

def isolated_spike(values: np.ndarray, valid: np.ndarray, jump: float=40.0, neighbor: float=20.0):
    keep = valid.copy()
    for i in range(1, len(values) - 1):
        if valid[i - 1] and valid[i] and valid[i + 1] and (abs(values[i] - values[i - 1]) >= jump) and (abs(values[i] - values[i + 1]) >= jump) and (abs(values[i - 1] - values[i + 1]) <= neighbor):
            keep[i] = False
    return keep

def run_mask(mask: np.ndarray, run: int) -> bool:
    n = 0
    for x in mask:
        n = n + 1 if bool(x) else 0
        if n >= run:
            return True
    return False

def cluster_diff(df: pd.DataFrame, outcome: str):
    z = df[[outcome, 'group', 'case']].dropna()
    X = sm.add_constant(z[['group']], has_constant='add')
    f = sm.OLS(z[outcome], X).fit(cov_type='cluster', cov_kwds={'groups': z.case})
    b = float(f.params.group)
    se = float(f.bse.group)
    return {'n': int(len(z)), 'clusters': int(z.case.nunique()), 'estimate': b, 'ci95': [b - 1.96 * se, b + 1.96 * se]}

def main():
    pairs = match_index()
    pairs['t0'] = pairs.t0.astype(float)
    pairs['row'] = np.arange(len(pairs), dtype=int)
    n = len(pairs)
    sums = np.zeros((n, 211), dtype=np.float64)
    counts = np.zeros((n, 211), dtype=np.int32)
    by_case = {}
    for case, g in pairs.groupby('case', sort=False):
        g = g.sort_values('t0')
        by_case[int(case)] = (g.t0.to_numpy(float), g.row.to_numpy(int))
    path = ROOT / 'data_float_h.csv.gz'
    rows = 0
    used = 0
    with gzip.open(path, 'rt', newline='', encoding='utf-8', errors='replace') as fh:
        rd = csv.reader(fh)
        header = next(rd)
        ic, idd, io, iv, ir = (header.index(k) for k in ('CaseID', 'DataID', 'Offset', 'Val', 'rawdata'))
        for row in rd:
            rows += 1
            if row[idd] != '703':
                continue
            case = int(row[ic])
            lookup = by_case.get(case)
            if lookup is None:
                continue
            tt, vv = decode(row[ir], row[iv], int(row[io]))
            if len(tt) == 0:
                continue
            starts, inds = lookup
            lo = max(0, np.searchsorted(starts, float(tt.min()) - 7200, side='right'))
            hi = np.searchsorted(starts, float(tt.max()) + 1800, side='left')
            for k in range(lo, hi):
                j = int(inds[k])
                dt = np.rint((tt - starts[k]) / 60.0).astype(int)
                ok = (dt >= -30) & (dt <= 180) & np.isfinite(vv)
                if not ok.any():
                    continue
                pos = dt[ok] + 30
                val = vv[ok]
                np.add.at(sums[j], pos, val)
                np.add.at(counts[j], pos, 1)
                used += int(ok.sum())
            if rows % 20000000 == 0:
                print(f'scanned rows={rows:,} used_points={used:,}', flush=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        values = sums / np.maximum(counts, 1)
    values[counts == 0] = np.nan
    np.savez_compressed(REV / 'sicdb_sham_minute_matrix.npz', values=values.astype(np.float32), iid=pairs.iid.to_numpy(), grp=pairs.grp.to_numpy().astype(str), rid=pairs.rid.to_numpy(), case=pairs.case.to_numpy(), drug_id=pairs.DrugID.to_numpy())
    records = []
    for i, row in pairs.iterrows():
        v = values[int(row.row)]
        pre = v[:30]
        post = v[30:151]
        pre_valid = np.isfinite(pre) & (pre >= 20) & (pre <= 250)
        baseline = float(np.nanmean(pre[pre_valid])) if pre_valid.sum() >= 3 else np.nan
        baseline_last = float(pre[np.where(pre_valid)[0][-1]]) if pre_valid.any() else np.nan
        valid = np.isfinite(post) & (post >= 20) & (post <= 250)
        keep = isolated_spike(post, valid)

        def event(frac=None, threshold=None, run=1):
            m = keep.copy()
            if frac is not None:
                m &= post <= frac * baseline
            if threshold is not None:
                m &= post < threshold
            return int(np.isfinite(baseline) and run_mask(m, run))
        vals_sched = []
        for minute in [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 105, 120, 135, 150, 165, 180]:
            vals_sched.append(v[30 + minute])
        cantais = int(np.isfinite(baseline_last) and any((np.isfinite(x) and x <= 0.85 * baseline_last for x in vals_sched)))
        records.append({'iid': int(row.iid), 'grp': row.grp, 'rid': int(row.rid), 'case': int(row.case), 'drug': 'metamizole' if int(row.DrugID) == 1610 else 'paracetamol', 'baseline': baseline, 'baseline_last': baseline_last, 'baseline_sd': float(np.nanstd(pre[pre_valid], ddof=1)) if pre_valid.sum() >= 2 else np.nan, 'n_post_out_of_range': int((np.isfinite(post) & ~((post >= 20) & (post <= 250))).sum()), 'n_isolated_spikes_removed': int(valid.sum() - keep.sum()), 'raw15': event(frac=0.85), 'raw20': event(frac=0.8), 'raw65': event(threshold=65), 'filtered15': event(frac=0.85), 'filtered20': event(frac=0.8), 'filtered65': event(threshold=65), 'persistent2_15': event(frac=0.85, run=2), 'persistent10_15': event(frac=0.85, run=10), 'persistent2_20': event(frac=0.8, run=2), 'persistent10_20': event(frac=0.8, run=10), 'persistent2_65': event(threshold=65, run=2), 'persistent10_65': event(threshold=65, run=10), 'cantais17': cantais, 'post_mean_30_120': float(np.nanmean(v[60:150])) if np.isfinite(v[60:150]).any() else np.nan})
    e = pd.DataFrame(records)
    e = e[e.rid.isin(e.loc[e.grp == 'sham', 'rid']) & e.rid.isin(e.loc[e.grp == 'real', 'rid'])].copy()
    result = {'source': str(path), 'scanned_rows': rows, 'used_points': used, 'n_pairs': int(e.rid.nunique()), 'n_real': int((e.grp == 'real').sum()), 'n_sham': int((e.grp == 'sham').sum()), 'rules': {'map_range_mmHg': [20, 250], 'isolated_spike_jump_mmHg': 40, 'isolated_spike_neighbor_difference_mmHg': 20, 'cantais_note': 'all 17 scheduled points through +180 min were reconstructed from the serialized minute values'}}
    for col in ['raw15', 'raw20', 'raw65', 'filtered15', 'filtered20', 'filtered65', 'persistent2_15', 'persistent10_15', 'persistent2_20', 'persistent10_20', 'persistent2_65', 'persistent10_65', 'cantais17']:
        real = e[e.grp == 'real']
        sham = e[e.grp == 'sham']
        z = e[['grp', 'case', col]].copy()
        z['group'] = (z.grp == 'real').astype(float)
        result[col] = {'real_rate': float(real[col].mean()), 'sham_rate': float(sham[col].mean()), 'risk_difference': cluster_diff(z, col), 'by_drug': {d: {'real_rate': float(e.loc[(e.grp == 'real') & (e.drug == d), col].mean()), 'sham_rate': float(e.loc[(e.grp == 'sham') & (e.drug == d), col].mean()), 'n_real': int(((e.grp == 'real') & (e.drug == d)).sum()), 'n_sham': int(((e.grp == 'sham') & (e.drug == d)).sum())} for d in ['metamizole', 'paracetamol']}}
    result['baseline_variability'] = {'real_mean_sd': float(e.loc[e.grp == 'real', 'baseline_sd'].mean()), 'sham_mean_sd': float(e.loc[e.grp == 'sham', 'baseline_sd'].mean()), 'real_median_sd': float(e.loc[e.grp == 'real', 'baseline_sd'].median()), 'sham_median_sd': float(e.loc[e.grp == 'sham', 'baseline_sd'].median()), 'smd': float((e.loc[e.grp == 'real', 'baseline_sd'].mean() - e.loc[e.grp == 'sham', 'baseline_sd'].mean()) / np.sqrt((e.loc[e.grp == 'real', 'baseline_sd'].var() + e.loc[e.grp == 'sham', 'baseline_sd'].var()) / 2))}
    result['artifact_audit'] = {'post_points_out_of_range': int(e.n_post_out_of_range.sum()), 'post_points_observed': int(e.shape[0] * 121), 'isolated_spikes_removed': int(e.n_isolated_spikes_removed.sum()), 'note': 'rates use the same physiologic range for baseline and post-dose values; missing minutes remain missing'}
    y = e[['grp', 'case', 'post_mean_30_120', 'baseline_sd', 'baseline']].copy()
    y['y'] = y.post_mean_30_120 - y.baseline
    y['group'] = (y.grp == 'real').astype(float)
    y = y.dropna(subset=['y', 'baseline_sd'])
    X = sm.add_constant(y[['group', 'baseline_sd']], has_constant='add')
    f = sm.OLS(y.y, X).fit(cov_type='cluster', cov_kwds={'groups': y.case})
    b = float(f.params.group)
    se = float(f.bse.group)
    result['adjusted_change_for_baseline_sd'] = {'n': int(len(y)), 'estimate': b, 'ci95': [b - 1.96 * se, b + 1.96 * se]}
    e.to_csv(REV / 'sicdb_raw_robustness_events.csv', index=False)
    result['source'] = path.as_posix()
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'n_pairs': result['n_pairs'], 'scanned_rows': rows, 'used_points': used}, ensure_ascii=False))
if __name__ == '__main__':
    main()
