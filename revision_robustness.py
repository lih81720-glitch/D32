from __future__ import annotations
import json
from pathlib import Path
from config import DUCKDB_TEMP, HIRID_MAP_GLOB, HIRID_ROOT, RESULTS_DIR, SICDB_ROOT
import numpy as np
import pandas as pd
import statsmodels.api as sm
HERE = Path(__file__).resolve().parent
REV = HERE / 'results' / 'revision'
OUT = REV / 'revision_robustness_results.json'

def _cluster_diff(z: pd.DataFrame, outcome: str, group: str, cluster: str, covars: list[str] | None=None):
    cols = [outcome, group, cluster] + (covars or [])
    q = z[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(q) < 30 or q[group].nunique() < 2:
        return None
    X = sm.add_constant(q[[group] + (covars or [])], has_constant='add')
    fit = sm.OLS(q[outcome], X).fit(cov_type='cluster', cov_kwds={'groups': q[cluster]})
    b = float(fit.params[group])
    se = float(fit.bse[group])
    return {'n': int(len(q)), 'clusters': int(q[cluster].nunique()), 'estimate': b, 'ci95': [b - 1.96 * se, b + 1.96 * se]}

def _smd(a: pd.Series, b: pd.Series) -> float:
    a = a.dropna().to_numpy(float)
    b = b.dropna().to_numpy(float)
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float('nan')

def _isolated_spike(values: np.ndarray, valid: np.ndarray, jump: float=40.0, neighbor_jump: float=20.0) -> np.ndarray:
    keep = valid.copy()
    for i in range(1, len(values) - 1):
        if not (valid[i - 1] and valid[i] and valid[i + 1]):
            continue
        if abs(values[i] - values[i - 1]) >= jump and abs(values[i] - values[i + 1]) >= jump and (abs(values[i - 1] - values[i + 1]) <= neighbor_jump):
            keep[i] = False
    return keep

def _run_at_least(mask: np.ndarray, run: int) -> bool:
    if run <= 1:
        return bool(mask.any())
    count = 0
    for x in mask:
        count = count + 1 if bool(x) else 0
        if count >= run:
            return True
    return False

def _series_summary(values: np.ndarray, baseline: float, rel: np.ndarray, run: int):
    valid = np.isfinite(values) & (values >= 20.0) & (values <= 250.0)
    post = rel >= 0
    post_values = values[post]
    post_valid = valid[post]
    post_rel = rel[post]
    raw15 = bool(np.any(post_valid & (post_values <= 0.85 * baseline))) if np.isfinite(baseline) else False
    raw20 = bool(np.any(post_valid & (post_values <= 0.8 * baseline))) if np.isfinite(baseline) else False
    raw65 = bool(np.any(post_valid & (post_values < 65.0))) if np.isfinite(baseline) and baseline >= 65 else False
    screened = _isolated_spike(post_values, post_valid)
    filt15 = bool(np.any(screened & (post_values <= 0.85 * baseline))) if np.isfinite(baseline) else False
    filt20 = bool(np.any(screened & (post_values <= 0.8 * baseline))) if np.isfinite(baseline) else False
    filt65 = bool(np.any(screened & (post_values < 65.0))) if np.isfinite(baseline) and baseline >= 65 else False
    p15 = _run_at_least(screened & (post_values <= 0.85 * baseline), run) if np.isfinite(baseline) else False
    p20 = _run_at_least(screened & (post_values <= 0.8 * baseline), run) if np.isfinite(baseline) else False
    p65 = _run_at_least(screened & (post_values < 65.0), run) if np.isfinite(baseline) and baseline >= 65 else False
    return {'raw15': raw15, 'raw20': raw20, 'raw65': raw65, 'filtered15': filt15, 'filtered20': filt20, 'filtered65': filt65, 'persistent15': p15, 'persistent20': p20, 'persistent65': p65, 'n_valid_post': int(screened.sum()), 'n_post': int(post_rel.size)}

def _summarise_events(d: pd.DataFrame, db: str, run: int, bins_per_minute: float) -> dict:
    matched = d[d.rid.isin(d.loc[d.grp == 'sham', 'rid'])].copy()
    matched = matched[matched.rid.isin(d.loc[d.grp == 'real', 'rid'])].copy()
    out = []
    for iid, z in matched.groupby('iid', sort=False):
        z = z.sort_values('bin')
        baseline = float(z.loc[(z.bin >= -30) & (z.bin < 0), 'v'].mean())
        post = z[(z.bin >= 0) & (z.bin < 120)]
        if len(post) == 0 or not np.isfinite(baseline):
            continue
        rel = post.bin.to_numpy(float)
        values = post.v.to_numpy(float)
        summ = _series_summary(values, baseline, rel, run)
        row = z.iloc[0][['iid', 'grp', 'rid', 'patientid', 'drug']].to_dict()
        row.update({'baseline': baseline, **summ})
        out.append(row)
    e = pd.DataFrame(out)
    result = {'database': db, 'native_bin_minutes': float(1 / bins_per_minute), 'persistence_run': int(run), 'n_rows': int(len(e)), 'n_real': int((e.grp == 'real').sum()), 'n_sham': int((e.grp == 'sham').sum())}
    for col in ['raw15', 'raw20', 'raw65', 'filtered15', 'filtered20', 'filtered65', 'persistent15', 'persistent20', 'persistent65']:
        q = e[['grp', 'patientid', col]].dropna()
        real = q[q.grp == 'real']
        sham = q[q.grp == 'sham']
        result[col] = {'real_rate': float(real[col].mean()), 'sham_rate': float(sham[col].mean()), 'real_n': int(len(real)), 'sham_n': int(len(sham)), 'risk_difference': _cluster_diff(q.assign(group=(q.grp == 'real').astype(float)), col, 'group', 'patientid')}
        result[col]['by_drug'] = {}
        for drug in ['metamizole', 'paracetamol']:
            rd = e[e.drug == drug]
            result[col]['by_drug'][drug] = {'real_rate': float(rd.loc[rd.grp == 'real', col].mean()), 'sham_rate': float(rd.loc[rd.grp == 'sham', col].mean()), 'n_real': int((rd.grp == 'real').sum()), 'n_sham': int((rd.grp == 'sham').sum())}
    baselines = []
    for iid, z in matched.groupby('iid', sort=False):
        z = z.sort_values('bin')
        pre = z.loc[(z.bin >= -30) & (z.bin < 0), 'v'].to_numpy(float)
        pre = pre[np.isfinite(pre) & (pre >= 20) & (pre <= 250)]
        if len(pre) < 2:
            continue
        r = z.iloc[0][['iid', 'grp', 'rid', 'patientid', 'drug']].to_dict()
        r['baseline_sd'] = float(pre.std(ddof=1))
        r['baseline_mad'] = float(np.median(np.abs(pre - np.median(pre))))
        r['baseline'] = float(pre.mean())
        baselines.append(r)
    b = pd.DataFrame(baselines)
    result['baseline_variability'] = {'n': int(len(b)), 'real_mean_sd': float(b.loc[b.grp == 'real', 'baseline_sd'].mean()), 'sham_mean_sd': float(b.loc[b.grp == 'sham', 'baseline_sd'].mean()), 'real_median_sd': float(b.loc[b.grp == 'real', 'baseline_sd'].median()), 'sham_median_sd': float(b.loc[b.grp == 'sham', 'baseline_sd'].median()), 'smd_sd': _smd(b.loc[b.grp == 'real', 'baseline_sd'], b.loc[b.grp == 'sham', 'baseline_sd']), 'smd_mad': _smd(b.loc[b.grp == 'real', 'baseline_mad'], b.loc[b.grp == 'sham', 'baseline_mad'])}
    yrows = []
    for iid, z in matched.groupby('iid', sort=False):
        z = z.sort_values('bin')
        pre = z.loc[(z.bin >= -30) & (z.bin < 0), 'v']
        post = z.loc[(z.bin >= 30) & (z.bin < 120), 'v']
        if pre.notna().sum() < 2 or post.notna().sum() < 1:
            continue
        rr = z.iloc[0][['iid', 'grp', 'rid', 'patientid', 'drug']].to_dict()
        rr['y'] = float(post.mean() - pre.mean())
        rr['baseline_sd'] = float(pre.std(ddof=1))
        yrows.append(rr)
    y = pd.DataFrame(yrows)
    y['group'] = (y.grp == 'real').astype(float)
    result['adjusted_change_for_baseline_sd'] = _cluster_diff(y, 'y', 'group', 'patientid', ['baseline_sd'])
    e.to_csv(REV / f'robustness_events_{db}.csv', index=False)
    return result

def _cantais_sicdb() -> dict:
    idx = pd.read_parquet(REV / 'sicdb_sham_index_raw.parquet')
    idx = idx[idx.pre_n >= 3].copy()
    real = idx[idx.grp == 'real'].set_index('rid')
    cand = idx[idx.grp == 'cand'].copy()
    cand = cand.join(real[['pre_mean', 'pre_slope', 'pre_temp', 'pressor', 'DrugID', 'case']], on='rid', rsuffix='_r')
    cand = cand[cand.pressor_r.notna()]
    cand['dist'] = (cand.pre_mean - cand.pre_mean_r).abs() / 5 + (cand.pre_slope - cand.pre_slope_r).abs() * 60 / 2 + (cand.pre_temp - cand.pre_temp_r).abs().fillna(0) + 10 * (cand.pressor != cand.pressor_r) + (cand.offset_h.abs() - 24).abs() / 2
    best = cand.sort_values('dist').groupby('rid').head(1)
    r = real.reset_index()
    b = best.reset_index(drop=True)
    pairs = []
    schedule = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 105, 120, 135, 150, 165, 180]
    for src, grp in [(r, 'real'), (b, 'sham')]:
        for row in src.to_dict('records'):
            base = row.get('b-10', np.nan)
            if not np.isfinite(base):
                continue
            vals = []
            for minute in schedule:
                bin_min = int(round(minute / 10) * 10)
                col = f'b{bin_min}'
                vals.append(row.get(col, np.nan))
            event = any((np.isfinite(v) and v <= 0.85 * base for v in vals))
            pairs.append({'rid': int(row['rid']), 'grp': grp, 'case': int(row['case']), 'event': int(event)})
    d = pd.DataFrame(pairs)
    d = d[d.rid.isin(d.loc[d.grp == 'sham', 'rid']) & d.rid.isin(d.loc[d.grp == 'real', 'rid'])]
    d['group'] = (d.grp == 'real').astype(float)
    fit = _cluster_diff(d, 'event', 'group', 'case')
    return {'n_real': int((d.grp == 'real').sum()), 'n_sham': int((d.grp == 'sham').sum()), 'real_rate': float(d.loc[d.grp == 'real', 'event'].mean()), 'sham_rate': float(d.loc[d.grp == 'sham', 'event'].mean()), 'risk_difference': fit, 'schedule_minutes': schedule, 'resolution_note': 'nearest available 10-min mean; not raw minute-level Cantais sampling'}

def _sicdb_matched_index() -> pd.DataFrame:
    idx = pd.read_parquet(REV / 'sicdb_sham_index_raw.parquet')
    idx = idx[idx.pre_n >= 3].copy()
    real = idx[idx.grp == 'real'].set_index('rid')
    cand = idx[idx.grp == 'cand'].copy()
    cand = cand.join(real[['pre_mean', 'pre_slope', 'pre_temp', 'pressor', 'DrugID', 'case']], on='rid', rsuffix='_r')
    cand = cand[cand.pressor_r.notna()]
    cand['dist'] = (cand.pre_mean - cand.pre_mean_r).abs() / 5 + (cand.pre_slope - cand.pre_slope_r).abs() * 60 / 2 + (cand.pre_temp - cand.pre_temp_r).abs().fillna(0) + 10 * (cand.pressor != cand.pressor_r) + (cand.offset_h.abs() - 24).abs() / 2
    best = cand.sort_values('dist').groupby('rid').head(1).copy()
    matched_rids = set(best.rid.astype(int))
    r = real.reset_index()
    r = r[r.rid.astype(int).isin(matched_rids)].copy()
    b = best.copy()
    r['grp'] = 'real'
    b['grp'] = 'sham'
    b['DrugID'] = b['DrugID_r']
    keep = ['iid', 'grp', 'rid', 'case', 'DrugID'] + [c for c in idx.columns if c.startswith('b') and c[1:].lstrip('-').isdigit()]
    return pd.concat([r[keep], b[keep]], ignore_index=True)

def main():
    hi_idx = pd.read_parquet(REV / 'hirid_sham_index.parquet')
    hi_bins = pd.read_parquet(REV / 'hirid_sham_bins.parquet')
    hi = hi_bins.merge(hi_idx[['iid', 'grp', 'rid', 'patientid', 'pharmaid']], on='iid', how='left')
    hi['drug'] = np.where(hi.pharmaid == 1000605, 'metamizole', 'paracetamol')
    sic_idx = _sicdb_matched_index()
    bcols = [c for c in sic_idx.columns if c.startswith('b') and c[1:].lstrip('-').isdigit()]
    recs = []
    meta = sic_idx[['iid', 'grp', 'rid', 'case', 'DrugID']]
    for c in bcols:
        q = meta.copy()
        q['bin'] = int(c[1:])
        q['v'] = sic_idx[c].to_numpy(float)
        q['n'] = 1
        q = q[np.isfinite(q.v)]
        recs.append(q)
    sic = pd.concat(recs, ignore_index=True)
    sic['patientid'] = sic['case'].astype(int)
    sic['drug'] = np.where(sic.DrugID.astype(int) == 1610, 'metamizole', 'paracetamol')
    sic = sic[['iid', 'bin', 'v', 'n', 'grp', 'rid', 'patientid', 'drug']]
    result = {'rules': {'physiologic_map_mmHg': [20, 250], 'isolated_spike_jump_mmHg': 40, 'isolated_spike_neighbor_difference_mmHg': 20, 'persistence': {'HiRID': 'five consecutive 2-min bins (10 min) and two bins (4 min)', 'SICdb': 'one 10-min mean (10 min) and two means (20 min)'}, 'pulse_pressure_filter': 'not applied; SBP and DBP are not in the frozen extraction'}, 'hirid': {'ten_min': _summarise_events(hi, 'HiRID', 5, 0.5), 'two_readings': _summarise_events(hi, 'HiRID', 2, 0.5)}, 'sicdb': {'ten_min': _summarise_events(sic, 'SICdb', 1, 0.1), 'two_readings': _summarise_events(sic, 'SICdb', 2, 0.1), 'cantais_style': _cantais_sicdb()}}
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'hirid_n': result['hirid']['ten_min']['n_rows'], 'sicdb_n': result['sicdb']['ten_min']['n_rows'], 'sicdb_cantais': result['sicdb']['cantais_style']}, ensure_ascii=False))
if __name__ == '__main__':
    main()
