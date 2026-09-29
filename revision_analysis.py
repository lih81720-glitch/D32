from __future__ import annotations
import json
import sys
from pathlib import Path
from config import DUCKDB_TEMP, HIRID_MAP_GLOB, HIRID_ROOT, RESULTS_DIR, SICDB_ROOT
import numpy as np
import pandas as pd
import statsmodels.api as sm
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metamizole_followup import SOURCES, load_events, normalize, selection_auc
from metamizole_clean_rerun import COV
RES = RESULTS_DIR
REV = RES / 'revision'
SIC_ROOT = SICDB_ROOT
ORDER = ['hirid_no_pressor', 'hirid_pressor', 'sicdb_no_pressor', 'sicdb_pressor']
KEYPREFIX = {'hirid_no_pressor': 'hirid_nonpressor', 'hirid_pressor': 'hirid_pressor', 'sicdb_no_pressor': 'sicdb_nonpressor', 'sicdb_pressor': 'sicdb_pressor'}

def fit(x: pd.DataFrame, outcome: str, terms: list[str], covars: list[str], cluster: str='patient', fe: str='patient'):
    cols = list(dict.fromkeys([fe, cluster, outcome] + terms + [c for c in covars if c in x]))
    z = x[cols].replace([np.inf, -np.inf], np.nan).dropna()
    names = list(dict.fromkeys(terms + [c for c in covars if c in z]))
    if len(z) < 30 or z[fe].nunique() < 5:
        return None
    zd = z.copy()
    zd[[outcome] + names] = zd[[outcome] + names] - z.groupby(fe, sort=False)[[outcome] + names].transform('mean')
    keep = [n for n in names if zd[n].abs().sum() > 1e-09]
    r = sm.OLS(zd[outcome], zd[keep]).fit(cov_type='cluster', cov_kwds={'groups': z[cluster]})
    return {'fit': r, 'n': int(len(z)), 'patients': int(z[fe].nunique()), 'idx': z.index}

def lincomb(res, weights: dict[str, float]):
    if res is None:
        return {'status': 'insufficient'}
    r = res['fit']
    if any((k not in r.params.index for k in weights)):
        return {'status': 'not_identified', 'n': res['n']}
    w = pd.Series(0.0, index=r.params.index)
    for k, v in weights.items():
        w[k] = v
    est = float(w @ r.params)
    se = float(np.sqrt(w @ r.cov_params() @ w))
    if not np.isfinite(se) or se < 1e-08:
        return {'status': 'not_estimable', 'n': res['n'], 'patients': res['patients']}
    return {'status': 'ok', 'n': res['n'], 'patients': res['patients'], 'estimate': est, 'se': se, 'ci95_low': est - 1.96 * se, 'ci95_high': est + 1.96 * se}

def pool(items: list[dict]):
    ok = [i for i in items if i and i.get('status') == 'ok']
    if len(ok) < 2:
        return None
    b = np.array([i['estimate'] for i in ok])
    v = np.array([i['se'] ** 2 for i in ok])
    w = 1 / v
    fe_ = (w * b).sum() / w.sum()
    se_fe = np.sqrt(1 / w.sum())
    Q = float((w * (b - fe_) ** 2).sum())
    tau2 = max(0.0, (Q - (len(b) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (v + tau2)
    re_ = (ws * b).sum() / ws.sum()
    se_re = np.sqrt(1 / ws.sum())
    diff = b[0] - b[1]
    se_d = np.sqrt(v[0] + v[1])
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(diff / se_d) / sqrt(2))))
    return {'fixed': [fe_, fe_ - 1.96 * se_fe, fe_ + 1.96 * se_fe], 'random_DL': [re_, re_ - 1.96 * se_re, re_ + 1.96 * se_re], 'tau2': tau2, 'Q': Q, 'difference_between_databases': [diff, diff - 1.96 * se_d, diff + 1.96 * se_d], 'p_difference': p}

def load_all() -> dict[str, pd.DataFrame]:
    cases = pd.read_csv(SIC_ROOT / 'cases.csv.gz', usecols=['CaseID', 'PatientID', 'AdmissionFormHasSepsis'])
    out = {}
    for name in ORDER:
        _, raw = load_events(SOURCES[name])
        x = normalize(name, raw)
        x['key'] = [f'{KEYPREFIX[name]}:{i}' for i in range(len(x))]
        if name.startswith('sicdb'):
            x = x.assign(cid=x.CaseID.astype(int)).merge(cases.rename(columns={'CaseID': 'cid'}), on='cid', how='left')
            x['patient_true'] = x.PatientID.astype('Int64').astype(str)
            x['sepsis_adm'] = (x.AdmissionFormHasSepsis == 740).astype(float)
        else:
            x['patient_true'] = x['patient']
        x['fever'] = np.where(x.pre_temp_use.notna(), (x.pre_temp_use >= 38.3).astype(float), np.nan)
        x['fever38'] = np.where(x.pre_temp_use.notna(), (x.pre_temp_use >= 38.0).astype(float), np.nan)
        x['gap_c'] = x.gap_hours - 3.0
        out[name] = x
    return out

def part_a(X: dict[str, pd.DataFrame]) -> dict:
    R: dict = {}
    cov4 = [c if c != 'pre_temp_model' else 'pre_temp_use' for c in COV]
    R['primary'] = {k: lincomb(fit(X[k], 'map', ['treatment'], COV), {'treatment': 1}) for k in ORDER}
    R['sicdb_patient_fe'] = {k: lincomb(fit(X[k], 'map', ['treatment'], COV, cluster='patient_true', fe='patient_true'), {'treatment': 1}) for k in ('sicdb_no_pressor', 'sicdb_pressor')}
    R['sicdb_patient_fe_ne'] = lincomb(fit(X['sicdb_pressor'], 'ne_outcome', ['treatment'], COV, cluster='patient_true', fe='patient_true'), {'treatment': 1})
    R['sicdb_admission_fe_patient_cluster'] = {k: lincomb(fit(X[k], 'map', ['treatment'], COV, cluster='patient_true'), {'treatment': 1}) for k in ('sicdb_no_pressor', 'sicdb_pressor')}
    R['hirid_np_temp4h'] = lincomb(fit(X['hirid_no_pressor'], 'map', ['treatment'], cov4), {'treatment': 1})
    R['hirid_np_no_temperature'] = lincomb(fit(X['hirid_no_pressor'], 'map', ['treatment'], [c for c in COV if c != 'pre_temp_model']), {'treatment': 1})
    R['carryover'] = {}
    for k in ORDER:
        x = X[k].copy()
        x['t_gap'] = x.treatment * x.gap_c
        res = fit(x, 'map', ['treatment', 't_gap'], COV)
        R['carryover'][k] = {'difference_at_3h': lincomb(res, {'treatment': 1}), 'change_per_hour': lincomb(res, {'t_gap': 1}), 'gap_hours_quartiles': [float(q) for q in x.gap_hours.quantile([0.25, 0.5, 0.75])]}
    R['carryover']['pooled_change_per_hour'] = pool([R['carryover'][k]['change_per_hour'] for k in ORDER])
    R['by_interval_band'] = {}
    for k in ORDER:
        x = X[k]
        R['by_interval_band'][k] = {lab: lincomb(fit(x[m], 'map', ['treatment'], COV), {'treatment': 1}) for lab, m in [('2-3h', x.gap_hours < 3), ('3-4h', (x.gap_hours >= 3) & (x.gap_hours < 4)), ('>=4h', x.gap_hours >= 4)]}

    def subgroup(k, g, covars, outcome='map'):
        x = X[k].copy()
        x = x[x[g].notna()]
        if x[g].sum() < 20:
            return {'status': 'too_few', 'n_subgroup': int(x[g].sum())}
        x['t_g'] = x.treatment * x[g]
        res = fit(x, outcome, ['treatment', 't_g', g], covars)
        return {'n_subgroup_doses': int(x[g].sum()), 'n_subgroup_model': int(x.loc[res['idx'], g].sum()) if res else 0, 'n_model': res['n'] if res else 0, 'subgroup': lincomb(res, {'treatment': 1, 't_g': 1}), 'others': lincomb(res, {'treatment': 1}), 'interaction': lincomb(res, {'t_g': 1})}
    R['fever'] = {k: subgroup(k, 'fever', cov4) for k in ORDER}
    R['fever38'] = {k: subgroup(k, 'fever38', cov4) for k in ORDER}
    for s in ('fever', 'fever38'):
        R[s]['pooled_no_pressor'] = pool([R[s][k]['subgroup'] for k in ('hirid_no_pressor', 'sicdb_no_pressor') if 'subgroup' in R[s][k]])
        R[s]['pooled_pressor'] = pool([R[s][k]['subgroup'] for k in ('hirid_pressor', 'sicdb_pressor') if 'subgroup' in R[s][k]])
        R[s]['pooled_all'] = pool([R[s][k]['subgroup'] for k in ORDER if 'subgroup' in R[s][k]])
    for k in ('hirid_pressor', 'sicdb_pressor'):
        X[k]['ne_high'] = np.where(X[k]['ne'].notna(), (X[k]['ne'] >= 0.1).astype(float), np.nan)
        med = X[k]['ne'].median()
        X[k]['ne_above_median'] = np.where(X[k]['ne'].notna(), (X[k]['ne'] >= med).astype(float), np.nan)
    R['ne_high'] = {k: {'map': subgroup(k, 'ne_high', COV), 'ne': subgroup(k, 'ne_high', COV, 'ne_outcome')} for k in ('hirid_pressor', 'sicdb_pressor')}
    R['ne_high']['pooled_map'] = pool([R['ne_high'][k]['map']['subgroup'] for k in ('hirid_pressor', 'sicdb_pressor')])
    R['ne_high']['pooled_ne'] = pool([R['ne_high'][k]['ne']['subgroup'] for k in ('hirid_pressor', 'sicdb_pressor')])
    R['ne_above_median'] = {k: {'map': subgroup(k, 'ne_above_median', COV), 'ne': subgroup(k, 'ne_above_median', COV, 'ne_outcome'), 'median': float(X[k]['ne'].median())} for k in ('hirid_pressor', 'sicdb_pressor')}
    R['sepsis_adm'] = {k: subgroup(k, 'sepsis_adm', COV) for k in ('sicdb_no_pressor', 'sicdb_pressor')}
    R['balance_analysed'] = {}
    R['missingness'] = {}
    vars_ = ['pre_map', 'pre_map_slope', 'pre_temp_use', 'ne', 'ne_change', 'gap_hours', 'icu_hours', 'met_24h', 'para_24h']
    for k in ORDER:
        x = X[k]
        cc = x[['patient', 'map', 'treatment'] + [c for c in COV if c in x]].replace([np.inf, -np.inf], np.nan).dropna().index
        a = x.loc[cc]
        R['balance_analysed'][k] = {}
        for v in vars_:
            m1, m0 = (a.loc[a.treatment == 1, v].dropna(), a.loc[a.treatment == 0, v].dropna())
            sd = np.sqrt((m1.var() + m0.var()) / 2)
            R['balance_analysed'][k][v] = float((m1.mean() - m0.mean()) / sd) if sd and np.isfinite(sd) and (sd > 0) else 0.0
        R.setdefault('auc_analysed', {})[k] = selection_auc(x.loc[cc], include_sequence=False)
        inn = x.index.isin(cc)
        R['missingness'][k] = {'cohort': int(len(x)), 'analysed': int(inn.sum()), 'metamizole_share_analysed': float(x.loc[inn, 'treatment'].mean()), 'metamizole_share_excluded': float(x.loc[~inn, 'treatment'].mean()) if (~inn).any() else None, 'pre_map_analysed': float(x.loc[inn, 'pre_map'].mean()), 'pre_map_excluded': float(x.loc[~inn, 'pre_map'].mean()) if (~inn).any() else None, 'icu_days_analysed': float(x.loc[inn, 'icu_hours'].median() / 24), 'icu_days_excluded': float(x.loc[~inn, 'icu_hours'].median() / 24) if (~inn).any() else None, 'unadjusted_map_change_analysed': float(x.loc[inn, 'map'].mean()), 'unadjusted_map_change_excluded': float(x.loc[~inn, 'map'].mean()) if (~inn).any() else None}
    R['pooling_map'] = {'no_pressor': pool([R['primary']['hirid_no_pressor'], R['primary']['sicdb_no_pressor']]), 'pressor': pool([R['primary']['hirid_pressor'], R['primary']['sicdb_pressor']])}
    R['pooling_ne'] = pool([lincomb(fit(X[k], 'ne_outcome', ['treatment'], COV), {'treatment': 1}) for k in ('hirid_pressor', 'sicdb_pressor')])
    R['pooling_escalation'] = pool([lincomb(fit(X[k], 'any_ne_up_or_new_pressor', ['treatment'], COV), {'treatment': 1}) for k in ('hirid_pressor', 'sicdb_pressor')])
    R['pooling_new_pressor'] = pool([lincomb(fit(X[k], 'any_ne_up_or_new_pressor', ['treatment'], COV), {'treatment': 1}) for k in ('hirid_no_pressor', 'sicdb_no_pressor')])
    return R

def bins_to_windows(bins: pd.DataFrame, key: str) -> pd.DataFrame:
    b = bins.copy()
    b['w'] = b.v * b.n

    def win(lo, hi):
        z = b[(b.bin >= lo) & (b.bin < hi)]
        return z.groupby(key).w.sum() / z.groupby(key).n.sum()
    out = pd.DataFrame({'base': win(-30, 0), 'p60_30': win(-60, -30), 'w30_60': win(30, 60), 'w60_90': win(60, 90), 'w90_120': win(90, 120), 'w120_150': win(120, 150), 'w150_180': win(150, 180), 'w30_120': win(30, 120)})
    return out

def part_b(X: dict[str, pd.DataFrame]) -> dict:
    R: dict = {}
    hb = pd.read_parquet(REV / 'hirid_xo_bins.parquet')
    parts_w = [bins_to_windows(hb, 'key')]
    if (REV / 'sicdb_xo_bins.parquet').exists():
        parts_w.append(bins_to_windows(pd.read_parquet(REV / 'sicdb_xo_bins.parquet'), 'key'))
    W = pd.concat(parts_w)
    hd = pd.read_parquet(REV / 'hirid_xo_dose.parquet')
    sd = pd.read_parquet(REV / 'sicdb_xo_dose.parquet')
    hc = pd.read_parquet(REV / 'hirid_xo_comed.parquet')
    sc = pd.read_parquet(REV / 'sicdb_xo_comed.parquet')
    hc['comed_pre'] = (hc.sed_pre + hc.bpl_pre > 0).astype(int)
    hc['comed_post'] = (hc.sed_post + hc.bpl_post > 0).astype(int)
    C = pd.concat([hc[['key', 'comed_pre', 'comed_post']], sc[['key', 'comed_pre', 'comed_post']]]).set_index('key')
    R['time_windows'] = {}
    R['dose'] = {}
    R['comedication'] = {}
    for k in ORDER:
        x = X[k].merge(W, left_on='key', right_index=True, how='left').merge(C, left_on='key', right_index=True, how='left')
        tw = {}
        for w in ['w30_60', 'w60_90', 'w90_120', 'w30_120']:
            x['y'] = x[w] - x['base']
            tw[w] = lincomb(fit(x, 'y', ['treatment'], COV), {'treatment': 1})
        ext = x[x.next_gap_minutes >= 180].copy()
        for w in ['w30_120', 'w120_150', 'w150_180']:
            ext['y'] = ext[w] - ext['base']
            tw['next180_' + w] = lincomb(fit(ext, 'y', ['treatment'], COV), {'treatment': 1})
        R['time_windows'][k] = tw
        cm = {'share_with_bolus_pre': float(x.comed_pre.mean()), 'share_with_bolus_post': float(x.comed_post.mean()), 'share_post_metamizole': float(x.loc[x.treatment == 1, 'comed_post'].mean()), 'share_post_paracetamol': float(x.loc[x.treatment == 0, 'comed_post'].mean())}
        cm['exclude_any_bolus'] = lincomb(fit(x[(x.comed_pre == 0) & (x.comed_post == 0)], 'map', ['treatment'], COV), {'treatment': 1})
        cm['adjust_pre_bolus'] = lincomb(fit(x, 'map', ['treatment'], COV + ['comed_pre']), {'treatment': 1})
        R['comedication'][k] = cm
        if k.startswith('hirid'):
            x = x.merge(hd, on='key', how='left')
            met = x.treatment == 1
            desc = {'met_dose_counts': x.loc[met, 'dose'].round(0).value_counts().head(6).to_dict(), 'met_route_counts': x.loc[met, 'route'].value_counts().head(6).to_dict(), 'met_units': x.loc[met, 'doseunit'].value_counts().head(3).to_dict(), 'para_pharmaid_counts': x.loc[~met, 'pharmaid'].value_counts().to_dict(), 'para_dose_counts': x.loc[~met, 'dose'].round(0).value_counts().head(6).to_dict()}
            x['met_bolus'] = (met & x.route.fillna('').str.contains('inj')).astype(float)
            x['met_infusion'] = (met & ~x.route.fillna('').str.contains('inj')).astype(float)
            x['met_high'] = (met & (x.dose >= 1000)).astype(float)
            x['met_low'] = (met & (x.dose < 1000)).astype(float)
        else:
            x = x.merge(sd, on='key', how='left')
            met = x.treatment == 1
            desc = {'met_amount_counts': x.loc[met, 'Amount'].round(2).value_counts().head(6).to_dict(), 'met_duration_quantiles': [float(q) for q in x.loc[met, 'duration_min'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])], 'para_amount_counts': x.loc[~met, 'Amount'].round(2).value_counts().head(6).to_dict(), 'para_duration_quantiles': [float(q) for q in x.loc[~met, 'duration_min'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])]}
            x['met_bolus'] = (met & (x.duration_min <= 5)).astype(float)
            x['met_infusion'] = (met & (x.duration_min > 5)).astype(float)
            x['met_high'] = (met & (x.Amount >= 1.5)).astype(float)
            x['met_low'] = (met & (x.Amount < 1.5)).astype(float)
        dres = fit(x, 'map', ['met_bolus', 'met_infusion'], COV)
        desc['met_bolus_vs_paracetamol'] = lincomb(dres, {'met_bolus': 1})
        desc['met_infusion_vs_paracetamol'] = lincomb(dres, {'met_infusion': 1})
        desc['n_met_bolus'] = int(x.met_bolus.sum())
        desc['n_met_infusion'] = int(x.met_infusion.sum())
        desc['n_met_bolus_model'] = int(x.loc[dres['idx'], 'met_bolus'].sum())
        desc['n_met_infusion_model'] = int(x.loc[dres['idx'], 'met_infusion'].sum())
        dres = fit(x, 'map', ['met_high', 'met_low'], COV)
        desc['met_high_vs_paracetamol'] = lincomb(dres, {'met_high': 1})
        desc['met_low_vs_paracetamol'] = lincomb(dres, {'met_low': 1})
        desc['n_met_high'] = int(x.met_high.sum())
        desc['n_met_low'] = int(x.met_low.sum())
        desc['n_met_high_model'] = int(x.loc[dres['idx'], 'met_high'].sum())
        desc['n_met_low_model'] = int(x.loc[dres['idx'], 'met_low'].sum())
        R['dose'][k] = desc
    for w in ['w30_60', 'w60_90', 'w90_120', 'next180_w120_150', 'next180_w150_180']:
        R['time_windows']['pooled_' + w] = {'no_pressor': pool([R['time_windows'][k][w] for k in ('hirid_no_pressor', 'sicdb_no_pressor')]), 'pressor': pool([R['time_windows'][k][w] for k in ('hirid_pressor', 'sicdb_pressor')])}
    return R

def sham_v2() -> dict:
    R: dict = {}
    si = pd.read_parquet(REV / 'hirid_sham_index.parquet')
    sb = pd.read_parquet(REV / 'hirid_sham_bins.parquet')
    W = bins_to_windows(sb, 'iid')
    nad = sb[(sb.bin >= 0) & (sb.bin < 120)].groupby('iid').v.min()
    s = si.merge(W, left_on='iid', right_index=True, how='left')
    s['nadir'] = s.iid.map(nad)
    s['drug'] = np.where(s.pharmaid == 1000605, 'metamizole', 'paracetamol')
    R['hirid'] = summarise_sham(s, 'patientid', clock=True)
    s[['iid', 'grp', 'rid', 'patientid', 'drug', 'matched', 'base', 'nadir', 'w30_120']].to_parquet(REV / 'sham_v2_hirid_events.parquet')
    ct = pd.read_parquet(REV / 'hirid_sham_cantais.parquet').merge(si[['iid', 'grp', 'patientid', 'rid']], on='iid')
    ct = ct[ct.points >= 10]
    both = set(ct[ct.grp == 'real'].rid) & set(ct[ct.grp == 'sham'].rid)
    ct = ct[ct.rid.isin(both)]
    R['hirid']['cantais'] = {g: {'events': int(z.event.sum()), 'n': int(len(z)), 'rate': float(z.event.mean())} for g, z in ct.groupby('grp')}
    ct['x'] = (ct.grp == 'real').astype(float)
    f = sm.OLS(ct.event, sm.add_constant(ct[['x']])).fit(cov_type='cluster', cov_kwds={'groups': ct.patientid})
    R['hirid']['cantais']['diff'] = [float(f.params.x), float(f.params.x - 1.96 * f.bse.x), float(f.params.x + 1.96 * f.bse.x)]
    curves = []
    sb2 = sb.merge(s[['iid', 'grp', 'drug', 'rid', 'matched', 'base', 'patientid']], on='iid')
    sb2 = sb2[sb2.rid.isin(s[s.grp == 'sham'].rid)]
    sb2 = pd.concat([sb2, sb2.assign(drug='all')])
    for (g, dname), z in sb2.groupby(['grp', 'drug']):
        z = z.assign(chg=z.v - z.base).dropna(subset=['chg'])
        for t, zz in z.groupby('bin'):
            m = zz.chg.mean()
            r = (zz.chg - m).groupby(zz.patientid).sum()
            se = np.sqrt((r ** 2).sum()) / len(zz)
            curves.append(('hirid', g, dname, t, m, se, len(zz)))
    if not (REV / 'sicdb_sham_index_raw.parquet').exists():
        pd.DataFrame(curves, columns=['db', 'grp', 'drug', 'bin', 'mean', 'se', 'n']).to_csv(REV / 'sham_v2_curves.csv', index=False)
        return R
    idx = pd.read_parquet(REV / 'sicdb_sham_index_raw.parquet')
    idx = idx[idx.pre_n >= 3]
    real = idx[idx.grp == 'real'].set_index('rid')
    cand = idx[idx.grp == 'cand'].copy()
    cand = cand.join(real[['pre_mean', 'pre_slope', 'pre_temp', 'pressor', 'DrugID', 'case']], on='rid', rsuffix='_r')
    cand = cand[cand.pressor_r.notna()]
    cand['dist'] = (cand.pre_mean - cand.pre_mean_r).abs() / 5 + (cand.pre_slope - cand.pre_slope_r).abs() * 60 / 2 + (cand.pre_temp - cand.pre_temp_r).abs().fillna(0) + 10 * (cand.pressor != cand.pressor_r) + (cand.offset_h.abs() - 24).abs() / 2
    cand = cand[cand.b30.notna() | cand.b60.notna()]
    best = cand.sort_values('dist').groupby('rid').head(1)
    bcols = [c for c in idx.columns if c.startswith('b') and c[1:].lstrip('-').isdigit()]

    def windows(df):
        o = pd.DataFrame(index=df.index)
        o['base'] = df[['b-30', 'b-20', 'b-10']].mean(axis=1)
        o['p60_30'] = df[['b-60', 'b-50', 'b-40']].mean(axis=1)
        o['w30_120'] = df[['b30', 'b40', 'b50', 'b60', 'b70', 'b80', 'b90', 'b100', 'b110']].mean(axis=1)
        o['w30_60'] = df[['b30', 'b40', 'b50']].mean(axis=1)
        o['w60_90'] = df[['b60', 'b70', 'b80']].mean(axis=1)
        o['w90_120'] = df[['b90', 'b100', 'b110']].mean(axis=1)
        return o
    r_ = real.reset_index()
    r_ = pd.concat([r_, windows(r_)], axis=1)
    r_['grp'] = 'real'
    r_['matched'] = r_.rid.isin(best.rid).astype(int)
    b_ = best.reset_index(drop=True)
    b_ = pd.concat([b_, windows(b_)], axis=1)
    b_['grp'] = 'sham'
    b_['matched'] = 1
    b_['DrugID'] = b_['DrugID_r']
    ss = pd.concat([r_, b_], ignore_index=True)
    ss['drug'] = np.where(ss.DrugID == 1610, 'metamizole', 'paracetamol')
    ss['patientid'] = ss.case
    R['sicdb'] = summarise_sham(ss, 'patientid', clock=False)
    ss[['grp', 'rid', 'patientid', 'drug', 'matched', 'pre_mean', 'base', 'nadir', 'w30_120']].to_parquet(REV / 'sham_v2_sicdb_events.parquet')
    sm_ = ss[ss.rid.isin(best.rid)]
    for g, z in pd.concat([sm_, sm_.assign(drug='all')]).groupby(['grp', 'drug']):
        z = z.copy()
        for c in bcols:
            t = int(c[1:])
            chg = (z[c] - z['base']).dropna()
            if len(chg) < 10:
                continue
            m = chg.mean()
            r = (chg - m).groupby(z.loc[chg.index, 'patientid']).sum()
            se = np.sqrt((r ** 2).sum()) / len(chg)
            curves.append(('sicdb', g[0], g[1], t, m, se, len(chg)))
    pd.DataFrame(curves, columns=['db', 'grp', 'drug', 'bin', 'mean', 'se', 'n']).to_csv(REV / 'sham_v2_curves.csv', index=False)
    return R

def summarise_sham(s: pd.DataFrame, pid: str, clock: bool) -> dict:
    out = {'real_isolated': int((s.grp == 'real').sum()), 'matched': int(((s.grp == 'real') & (s.matched == 1)).sum())}
    real = s[s.grp == 'real']
    out['matched_vs_unmatched'] = {c: [float(real.loc[real.matched == 1, c].mean()), float(real.loc[real.matched == 0, c].mean())] for c in ['pre_mean', 'pressor'] + (['clock_min'] if clock else []) if c in real}
    m = s[s.rid.isin(s[s.grp == 'sham'].rid)].copy()
    m['y'] = m.w30_120 - m.base
    m['y_alt'] = m.w30_120 - m.p60_30
    m['y_nadir'] = m.nadir - m.base
    m['pre_trend'] = m.base - m.p60_30
    for w in ['w30_60', 'w60_90', 'w90_120']:
        m['y_' + w] = m[w] - m.base
    m['x'] = (m.grp == 'real').astype(float)

    def contrast(df, col):
        z = df[[col, 'x', pid]].dropna()
        if len(z) < 30:
            return None
        f = sm.OLS(z[col], sm.add_constant(z[['x']])).fit(cov_type='cluster', cov_kwds={'groups': z[pid]})
        return {'n': int(len(z)), 'real_mean': float(z.loc[z.x == 1, col].mean()), 'sham_mean': float(z.loc[z.x == 0, col].mean()), 'diff': [float(f.params.x), float(f.params.x - 1.96 * f.bse.x), float(f.params.x + 1.96 * f.bse.x)]}
    out['patients'] = int(m[pid].nunique())
    for col in ['y', 'y_alt', 'y_nadir', 'pre_trend', 'y_w30_60', 'y_w60_90', 'y_w90_120']:
        out[col] = {'all': contrast(m, col)}
        for d in ('metamizole', 'paracetamol'):
            out[col][d] = contrast(m[m.drug == d], col)
    m['y_net'] = np.nan
    pair = m.pivot_table(index='rid', columns='grp', values='y')
    pair = pair.join(m[m.grp == 'real'].set_index('rid')[['drug', pid]])
    pair['net'] = pair.real - pair.sham
    z = pair.dropna(subset=['net'])
    z = z.assign(xm=(z.drug == 'metamizole').astype(float))
    f = sm.OLS(z.net, sm.add_constant(z[['xm']])).fit(cov_type='cluster', cov_kwds={'groups': z[pid]})
    out['net_metamizole_minus_paracetamol'] = [float(f.params.xm), float(f.params.xm - 1.96 * f.bse.xm), float(f.params.xm + 1.96 * f.bse.xm)]
    return out

def sham_definitions(n_boot: int=500, seed: int=1) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    h = pd.read_parquet(REV / 'sham_v2_hirid_events.parquet')
    sc = pd.read_parquet(REV / 'sham_v2_sicdb_events.parquet').assign(base=lambda d: d.pre_mean)
    ct = pd.read_parquet(REV / 'hirid_sham_cantais.parquet')
    for db, d in (('hirid', h), ('sicdb', sc)):
        d = d[d.rid.isin(d[d.grp == 'sham'].rid)].dropna(subset=['base', 'nadir']).copy()
        d = d[d.rid.isin(d[d.grp == 'real'].rid) & d.rid.isin(d[d.grp == 'sham'].rid)]
        defs = {'fall15': d.nadir <= 0.85 * d.base, 'fall20': d.nadir <= 0.8 * d.base}
        d['fall15'] = defs['fall15'].astype(float)
        d['fall20'] = defs['fall20'].astype(float)
        d['below65'] = np.where(d.base >= 65, (d.nadir < 65).astype(float), np.nan)
        cols = ['fall15', 'fall20', 'below65']
        if db == 'hirid':
            c = ct[ct.points >= 10][['iid', 'event']].rename(columns={'event': 'cantais'})
            d = d.merge(c, on='iid', how='left')
            cols.append('cantais')
        out[db] = {}
        for col in cols:
            z = d[['grp', 'patientid', 'drug', col]].dropna()
            real, sham = (z[z.grp == 'real'], z[z.grp == 'sham'])
            r1, r0 = (real[col].mean(), sham[col].mean())
            z = z.assign(x=(z.grp == 'real').astype(float))
            f = sm.OLS(z[col], sm.add_constant(z[['x']])).fit(cov_type='cluster', cov_kwds={'groups': z.patientid})
            pts = z.patientid.unique()
            g = {p: i for i, p in enumerate(pts)}
            zi = z.assign(pi=z.patientid.map(g))
            sums = zi.groupby(['pi', 'grp'])[col].agg(['sum', 'count']).unstack('grp').fillna(0)
            S1, N1 = (sums['sum', 'real'].values, sums['count', 'real'].values)
            S0, N0 = (sums['sum', 'sham'].values, sums['count', 'sham'].values)
            fr = []
            for _ in range(n_boot):
                i = rng.integers(0, len(pts), len(pts))
                a1, a0 = (S1[i].sum() / N1[i].sum(), S0[i].sum() / N0[i].sum())
                fr.append(a0 / a1)
            out[db][col] = {'n_real': int(len(real)), 'n_sham': int(len(sham)), 'rate_real': float(r1), 'rate_sham': float(r0), 'rd': [float(f.params.x), float(f.params.x - 1.96 * f.bse.x), float(f.params.x + 1.96 * f.bse.x)], 'fraction_without_dose': [float(r0 / r1), float(np.percentile(fr, 2.5)), float(np.percentile(fr, 97.5))], 'by_drug': {dr: {'rate_real': float(real.loc[real.drug == dr, col].mean()), 'rate_sham': float(sham.loc[sham.drug == dr, col].mean())} for dr in ('metamizole', 'paracetamol')}}
    return out
if __name__ == '__main__':
    part = sys.argv[1] if len(sys.argv) > 1 else 'a'
    X = load_all()
    outf = REV / 'revision_results.json'
    R = json.loads(outf.read_text()) if outf.exists() else {}
    if part in ('a', 'all'):
        R['part_a'] = part_a(X)
    if part in ('b', 'all'):
        R['part_b'] = part_b(X)
    if part in ('sham', 'all'):
        R['sham_v2'] = sham_v2()
    outf.write_text(json.dumps(R, indent=1, default=lambda o: None if isinstance(o, float) and (not np.isfinite(o)) else str(o)))
    print('written', outf)
