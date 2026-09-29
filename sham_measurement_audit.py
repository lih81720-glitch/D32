from __future__ import annotations
import json
from pathlib import Path
from config import DUCKDB_TEMP, HIRID_MAP_GLOB, HIRID_ROOT, RESULTS_DIR, SICDB_ROOT
import numpy as np
import pandas as pd
import statsmodels.api as sm
HERE = Path(__file__).resolve().parent
REV = HERE / 'results' / 'revision'
SCHED = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 105, 120, 135, 150, 165, 180]

def hirid_matrix():
    idx = pd.read_parquet(REV / 'hirid_sham_index.parquet')
    idx = idx[idx.rid.isin(idx.loc[idx.grp == 'sham', 'rid'])]
    b = pd.read_parquet(REV / 'hirid_sham_bins.parquet')
    b = b[b.iid.isin(idx.iid) & (b.bin >= -30) & (b.bin < 180)]
    iids = idx.iid.to_numpy()
    pos = {v: i for i, v in enumerate(iids)}
    m = np.full((len(iids), 105), np.nan)
    m[b.iid.map(pos).to_numpy(), ((b.bin.to_numpy() + 30) // 2).astype(int)] = b.v.to_numpy()
    meta = pd.DataFrame({'grp': idx.grp.to_numpy(), 'rid': idx.rid.to_numpy(), 'cluster': idx.patientid.to_numpy(), 'drug': np.where(idx.pharmaid.to_numpy() == 1000605, 'metamizole', 'paracetamol')})
    return (m, meta, 2)

def sicdb_matrix():
    z = np.load(REV / 'sicdb_sham_minute_matrix.npz', allow_pickle=True)
    m = z['values'].astype(float)[:, :210]
    meta = pd.DataFrame({'grp': z['grp'], 'rid': z['rid'], 'cluster': z['case'], 'drug': np.where(z['drug_id'] == 1610, 'metamizole', 'paracetamol')})
    keep = meta.rid.isin(meta.loc[meta.grp == 'sham', 'rid']) & meta.rid.isin(meta.loc[meta.grp == 'real', 'rid'])
    return (m[keep.to_numpy()], meta[keep].reset_index(drop=True), 1)

def spike_mask(post: np.ndarray, valid: np.ndarray) -> np.ndarray:
    keep = valid.copy()
    a, c, d = (post[:, :-2], post[:, 1:-1], post[:, 2:])
    va, vc, vd = (valid[:, :-2], valid[:, 1:-1], valid[:, 2:])
    spike = va & vc & vd & (np.abs(c - a) >= 40) & (np.abs(c - d) >= 40) & (np.abs(a - d) <= 20)
    keep[:, 1:-1] &= ~spike
    return keep

def any_run(mask: np.ndarray, run: int) -> np.ndarray:
    if run == 1:
        return mask.any(axis=1)
    c = np.zeros(mask.shape[0], int)
    best = np.zeros(mask.shape[0], bool)
    for j in range(mask.shape[1]):
        c = np.where(mask[:, j], c + 1, 0)
        best |= c >= run
    return best

def derive(m: np.ndarray, step: int, screen: bool) -> pd.DataFrame:
    n_pre = 30 // step
    n_post = 120 // step
    pre, post = (m[:, :n_pre], m[:, n_pre:n_pre + n_post])
    fin_pre, fin_post = (np.isfinite(pre), np.isfinite(post))
    vpre = fin_pre & (pre >= 20) & (pre <= 250) if screen else fin_pre
    vpost = fin_post & (post >= 20) & (post <= 250) if screen else fin_post
    if screen:
        vpost = spike_mask(post, vpost)
    pre_v = np.where(vpre, pre, np.nan)
    post_v = np.where(vpost, post, np.nan)
    with np.errstate(all='ignore'):
        base = np.where(vpre.sum(1) >= 3, np.nanmean(pre_v, 1), np.nan)
        out = pd.DataFrame({'base': base, 'pre_sd': np.nanstd(pre_v, 1, ddof=1), 'post_sd': np.nanstd(post_v, 1, ddof=1), 'nadir_chg': np.nanmin(post_v, 1) - base, 'mean_chg': np.nanmean(post_v[:, 30 // step:], 1) - base})
    run10 = 10 // step
    for lab, cond in (('fall15', post_v <= 0.85 * base[:, None]), ('fall20', post_v <= 0.8 * base[:, None]), ('below65', post_v < 65)):
        cond = cond & vpost
        for sfx, run in (('', 1), ('_sus', run10)):
            ev = any_run(cond, run).astype(float)
            ev[~np.isfinite(base)] = np.nan
            if lab == 'below65':
                ev[~(base >= 65)] = np.nan
            out[lab + sfx] = ev
    last = np.full(len(m), np.nan)
    for j in range(n_pre):
        last = np.where(vpre[:, j], pre[:, j], last)
    full = m[:, n_pre:]
    vfull = np.isfinite(full) & ((full >= 20) & (full <= 250) if screen else True)
    cols = [min(t // step, full.shape[1] - 1) for t in SCHED]
    sv = np.where(vfull[:, cols], full[:, cols], np.nan)
    npts = np.isfinite(sv).sum(1)
    ev = (sv <= 0.85 * last[:, None]).any(1).astype(float)
    ev[~np.isfinite(last) | (npts < 10)] = np.nan
    out['cantais'] = ev
    return out

def rd_and_fraction(d: pd.DataFrame, col: str, rng, n_boot=500) -> dict:
    z = d[['grp', 'cluster', 'drug', col]].dropna()
    real, sham = (z[z.grp == 'real'], z[z.grp == 'sham'])
    r1, r0 = (real[col].mean(), sham[col].mean())
    zz = z.assign(x=(z.grp == 'real').astype(float))
    f = sm.OLS(zz[col], sm.add_constant(zz[['x']])).fit(cov_type='cluster', cov_kwds={'groups': zz.cluster})
    g = zz.groupby(['cluster', 'grp'])[col].agg(['sum', 'count']).unstack('grp').fillna(0)
    S1, N1, S0, N0 = (g[s, k].to_numpy() for s, k in (('sum', 'real'), ('count', 'real'), ('sum', 'sham'), ('count', 'sham')))
    fr = []
    for _ in range(n_boot):
        i = rng.integers(0, len(g), len(g))
        fr.append(S0[i].sum() / N0[i].sum() / (S1[i].sum() / N1[i].sum()))
    return {'n_real': int(len(real)), 'n_sham': int(len(sham)), 'rate_real': float(r1), 'rate_sham': float(r0), 'rd': [float(f.params.x), float(f.params.x - 1.96 * f.bse.x), float(f.params.x + 1.96 * f.bse.x)], 'fraction_without_dose': [float(r0 / r1), float(np.percentile(fr, 2.5)), float(np.percentile(fr, 97.5))], 'by_drug': {dr: [float(real.loc[real.drug == dr, col].mean()), float(sham.loc[sham.drug == dr, col].mean())] for dr in ('metamizole', 'paracetamol')}}

def smd(a, b):
    a, b = (a.dropna(), b.dropna())
    return float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2))

def main(dbs=('hirid', 'sicdb')):
    rng = np.random.default_rng(1)
    res = {}
    for db, loader in [x for x in (('hirid', hirid_matrix), ('sicdb', sicdb_matrix)) if x[0] in dbs]:
        m, meta, step = loader()
        res[db] = {'pairs': int(meta.rid.nunique()), 'step_min': step}
        for screen in (True, False):
            d = pd.concat([meta, derive(m, step, screen)], axis=1)
            key = 'screened' if screen else 'unscreened'
            R = {c: rd_and_fraction(d, c, rng) for c in ['fall15', 'fall20', 'below65', 'fall15_sus', 'fall20_sus', 'below65_sus', 'cantais']}
            real, sham = (d[d.grp == 'real'], d[d.grp == 'sham'])
            R['variability'] = {'pre_sd': [float(real.pre_sd.mean()), float(sham.pre_sd.mean()), smd(real.pre_sd, sham.pre_sd)], 'post_sd': [float(real.post_sd.mean()), float(sham.post_sd.mean()), smd(real.post_sd, sham.post_sd)], 'nadir_chg_quantiles': {q: [float(real.nadir_chg.quantile(q)), float(sham.nadir_chg.quantile(q))] for q in (0.05, 0.1, 0.25, 0.5)}, 'nadir_chg_mean': [float(real.nadir_chg.mean()), float(sham.nadir_chg.mean())], 'mean_chg': [float(real.mean_chg.mean()), float(sham.mean_chg.mean())]}
            for outcome in ('nadir_chg', 'mean_chg'):
                z = d.dropna(subset=[outcome]).assign(x=lambda t: (t.grp == 'real').astype(float))
                f = sm.OLS(z[outcome], sm.add_constant(z[['x']])).fit(cov_type='cluster', cov_kwds={'groups': z.cluster})
                R['variability'][outcome + '_diff'] = [float(f.params.x), float(f.params.x - 1.96 * f.bse.x), float(f.params.x + 1.96 * f.bse.x)]
            y = d.dropna(subset=['mean_chg', 'pre_sd']).assign(x=lambda t: (t.grp == 'real').astype(float))
            for lab, cov in (('unadjusted', ['x']), ('adjusted_pre_sd', ['x', 'pre_sd'])):
                f = sm.OLS(y.mean_chg, sm.add_constant(y[cov])).fit(cov_type='cluster', cov_kwds={'groups': y.cluster})
                R['variability']['mean_chg_diff_' + lab] = [float(f.params.x), float(f.params.x - 1.96 * f.bse.x), float(f.params.x + 1.96 * f.bse.x)]
            R['variability']['post_sd_quantiles'] = {q: [float(real.post_sd.quantile(q)), float(sham.post_sd.quantile(q))] for q in (0.5, 0.75, 0.9, 0.95)}
            res[db][key] = R
    (REV / 'sham_measurement_audit.json').write_text(json.dumps(res, indent=1))
    for db in res:
        for key in ('screened', 'unscreened'):
            R = res[db][key]
            print(db, key, {c: (round(100 * R[c]['rate_real'], 1), round(100 * R[c]['rate_sham'], 1), [round(100 * t, 1) for t in R[c]['rd']], round(R[c]['fraction_without_dose'][0], 2), R[c]['n_real']) for c in R if c != 'variability'})
            print('   var', json.dumps(R['variability']))
if __name__ == '__main__':
    import sys
    main(tuple(sys.argv[1:]) or ('hirid', 'sicdb'))
