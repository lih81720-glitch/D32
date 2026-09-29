from __future__ import annotations
import json
from pathlib import Path
from config import DUCKDB_TEMP, HIRID_MAP_GLOB, HIRID_ROOT, RESULTS_DIR, SICDB_ROOT
import numpy as np
import pandas as pd
import statsmodels.api as sm
from math import erf, sqrt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parent
OUT = RESULTS_DIR

def load_events(path: Path) -> tuple[str, pd.DataFrame]:
    d = json.loads(path.read_text(encoding='utf-8'))
    rows = d.get('event_rows') or d.get('analysis', {}).get('event_rows', [])
    x = pd.DataFrame(rows)
    dataset = d.get('dataset', path.stem)
    return (dataset, x)
SOURCES = {'hirid_no_pressor': OUT / 'hirid_metamizole_crossover_analysis.json', 'hirid_pressor': OUT / 'hirid_metamizole_pressor_analysis.json', 'sicdb_pressor': OUT / 'sicdb_metamizole_pressor_analysis.json', 'sicdb_no_pressor': OUT / 'sicdb_metamizole_nopressor_analysis.json'}

def normalize(name: str, x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    if 'current_drug' in x:
        x['patient'] = x['patientid'].astype(str)
        x['treatment'] = (x.current_drug == 'metamizole').astype(float)
        x['previous_treatment'] = (x.previous_drug == 'metamizole').astype(float)
        x['pre_map_slope'] = x.get('pre_map_slope_min', np.nan) * 60
        x['pre_temp_use'] = x.get('pre_temp_4h', pd.Series(np.nan, index=x.index)).combine_first(x.get('temp_recent_4h', pd.Series(np.nan, index=x.index))).combine_first(x.get('pre_temp', pd.Series(np.nan, index=x.index)))
        x['pre_temp_model'] = x.get('pre_temp', pd.Series(np.nan, index=x.index)) if name == 'hirid_no_pressor' else x['pre_temp_use']
        x['gap_hours'] = x.get('prev_gap_minutes', np.nan) / 60
        next_col = 'current_next_at' if 'current_next_at' in x else 'next_at' if 'next_at' in x else None
        x['next_gap_minutes'] = (pd.to_datetime(x[next_col]) - pd.to_datetime(x['current_at'])).dt.total_seconds() / 60 if next_col else np.nan
        x['clean6'] = x['prev_gap_minutes'] >= 360
        x['gap4'] = (x['prev_gap_minutes'] >= 240) & (x['next_gap_minutes'] >= 240)
        x['gap6'] = (x['prev_gap_minutes'] >= 360) & (x['next_gap_minutes'] >= 360)
        x['map'] = x.get('map_change', np.nan)
        x['temp'] = x.get('temp_change', np.nan)
        x['placebo'] = x.get('placebo_change', np.nan)
        x['any_ne_up_or_new_pressor'] = x.get('any_ne_up_or_new_pressor', x.get('any_new_pressor', np.nan))
        x['ne'] = x.get('ne_current', np.nan)
        x['ne_change'] = x.get('ne_change_pre30', x.get('current_pre30_change', np.nan))
        x['ne_outcome'] = x.get('ne_change_0_120', np.nan)
        x['pressor_other'] = x.get('current_pressor_active', 0)
        x['pressor_flag'] = x.get('current_pressor_active', 0)
        if 'no_pressor' in name:
            x['ne'] = 0.0
            x['ne_change'] = 0.0
            x['pressor_other'] = 0
        x['patient_id'] = x['patient']
    elif 'drug' in x:
        x['patient'] = x['CaseID'].astype(str)
        x['treatment'] = (x.drug == 'metamizole').astype(float)
        x['previous_treatment'] = (x.previous_drug == 'metamizole').astype(float)
        x['pre_map_slope'] = x.get('pre_map_slope_hour', np.nan)
        x['pre_temp_use'] = x.get('temp_recent_4h', pd.Series(np.nan, index=x.index)).combine_first(x.get('pre_temp', pd.Series(np.nan, index=x.index)))
        x['pre_temp_model'] = x['pre_temp_use']
        x['gap_hours'] = x.get('prev_gap_hours', np.nan)
        x['next_gap_minutes'] = x.get('next_gap_sec', np.nan) / 60
        x['clean6'] = x['prev_gap_sec'] >= 21600 if 'prev_gap_sec' in x else False
        x['gap4'] = (x['prev_gap_sec'] >= 14400) & (x['next_gap_sec'] >= 14400) if 'prev_gap_sec' in x else False
        x['gap6'] = (x['prev_gap_sec'] >= 21600) & (x['next_gap_sec'] >= 21600) if 'prev_gap_sec' in x else False
        x['map'] = x.get('map_change', np.nan)
        x['temp'] = x.get('temp_change', np.nan)
        x['placebo'] = x.get('placebo_change', np.nan)
        x['any_ne_up_or_new_pressor'] = x.get('any_ne_up_or_new_pressor', np.nan)
        x['ne'] = x.get('ne_current', np.nan)
        x['ne_change'] = x.get('ne_change_pre30', np.nan)
        x['ne_outcome'] = x.get('ne_change_0_120', np.nan)
        x['pressor_other'] = ((x.get('epi_active', 0) == 1) | (x.get('vaso_active', 0) == 1)).astype(int)
        x['pressor_flag'] = 1
        x['patient_id'] = x['patient']
    return x

def fe(x: pd.DataFrame, outcome: str, covars: list[str]) -> dict:
    cols = ['patient', outcome, 'treatment'] + [c for c in covars if c in x]
    z = x[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(z) < 20 or z.patient.nunique() < 5:
        return {'status': 'insufficient', 'n': len(z), 'patients': int(z.patient.nunique())}
    names = ['treatment'] + [c for c in covars if c in z]
    g = z.groupby('patient', sort=False)
    zd = z.copy()
    zd[[outcome] + names] -= g[[outcome] + names].transform('mean')
    fit = sm.OLS(zd[outcome], zd[names]).fit(cov_type='cluster', cov_kwds={'groups': zd.patient})
    b, se = (float(fit.params['treatment']), float(fit.bse['treatment']))
    if not np.isfinite(se) or se < 1e-08:
        return {'status': 'not_estimable', 'n': len(z), 'patients': int(z.patient.nunique()), 'estimate': b, 'se': se, 'reason': 'near-zero clustered standard error indicates a degenerate fixed-effect design'}
    return {'status': 'ok', 'n': len(z), 'patients': int(z.patient.nunique()), 'estimate': b, 'se': se, 'ci95_low': b - 1.96 * se, 'ci95_high': b + 1.96 * se}

def sensitivity(name: str, x: pd.DataFrame) -> dict:
    cov = ['gap_hours', 'met_24h', 'para_24h', 'pre_map', 'pre_map_slope', 'pre_temp_model', 'ne', 'ne_change', 'pressor_other', 'icu_hours']
    out = {}
    for label, mask in [('min_interval_4h', x.gap4), ('min_interval_6h', x.gap6), ('clean_no_antipyretic_6h', x.clean6)]:
        y = x[mask].copy()
        out[label] = {'cohort': {'n': len(y), 'patients': int(y.patient.nunique())}, 'map': fe(y, 'map', cov), 'temperature': fe(y, 'temp', cov), 'placebo': fe(y, 'placebo', cov)}
    out['order_interaction'] = {'status': 'not_identifiable', 'reason': 'strict alternating sequence makes treatment × previous drug constant'}
    return out

def identified_models(x: pd.DataFrame) -> dict:
    cov = ['gap_hours', 'met_24h', 'para_24h', 'pre_map', 'pre_map_slope', 'pre_temp_model', 'ne', 'ne_change', 'pressor_other', 'icu_hours']
    return {'map': fe(x, 'map', cov), 'ne_outcome': fe(x, 'ne_outcome', cov), 'temperature': fe(x, 'temp', cov), 'placebo': fe(x, 'placebo', cov), 'binary_ne_up_or_new_pressor': fe(x, 'any_ne_up_or_new_pressor', cov), 'covariates_used': ['treatment'] + cov, 'dropped_for_exact_collinearity': ['previous_treatment']}

def smd(x: pd.DataFrame, col: str) -> dict:
    z = x[['treatment', col]].replace([np.inf, -np.inf], np.nan).dropna()
    a, b = (z.loc[z.treatment == 1, col], z.loc[z.treatment == 0, col])
    if len(a) < 2 or len(b) < 2:
        return {'n_met': len(a), 'n_para': len(b), 'smd': None}
    den = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return {'n_met': len(a), 'n_para': len(b), 'mean_met': float(a.mean()), 'mean_para': float(b.mean()), 'smd': float((a.mean() - b.mean()) / den) if den else 0.0}

def selection_auc(x: pd.DataFrame, include_sequence: bool=True) -> dict:
    cols = ['pre_map', 'pre_map_slope', 'ne', 'ne_change', 'pre_temp_use', 'previous_treatment', 'gap_hours', 'met_24h', 'para_24h', 'icu_hours']
    cols = [c for c in cols if c in x]
    if not include_sequence:
        cols = [c for c in cols if c not in {'previous_treatment', 'gap_hours', 'met_24h', 'para_24h'}]
    z = x[['patient', 'treatment'] + cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(z) < 30 or z.patient.nunique() < 5 or z.treatment.nunique() < 2:
        return {'status': 'insufficient', 'n': len(z), 'patients': int(z.patient.nunique())}
    for c in cols:
        if c != 'previous_treatment':
            z[c] = z[c] - z.groupby('patient')[c].transform('mean')
    groups = z.patient.astype(str)
    nsplit = min(5, groups.nunique())
    pred = np.full(len(z), np.nan)
    cv = GroupKFold(n_splits=nsplit)
    model = LogisticRegression(max_iter=500, solver='liblinear')
    for tr, te in cv.split(z[cols], z.treatment, groups):
        model.fit(z.iloc[tr][cols], z.iloc[tr].treatment)
        pred[te] = model.predict_proba(z.iloc[te][cols])[:, 1]
    return {'status': 'ok', 'n': len(z), 'patients': int(z.patient.nunique()), 'met': int(z.treatment.sum()), 'para': int((1 - z.treatment).sum()), 'auc': float(roc_auc_score(z.treatment, pred))}

def combined(a: pd.DataFrame, b: pd.DataFrame, label: str) -> dict:
    x = pd.concat([a.assign(dataset=0), b.assign(dataset=1)], ignore_index=True)
    x['patient'] = x['patient'].astype(str) + '_' + x.dataset.astype(str)
    cov = ['gap_hours', 'met_24h', 'para_24h', 'pre_map', 'pre_map_slope', 'pre_temp_model', 'ne', 'ne_change', 'pressor_other', 'icu_hours']
    cov = [c for c in cov if c in x]
    out = {'label': label, 'n': len(x), 'patients': int(x.patient.nunique()), 'map': fe(x, 'map', cov), 'temperature': fe(x, 'temp', cov), 'placebo': fe(x, 'placebo', cov)}
    z = x[['patient', 'treatment', 'dataset', 'map'] + cov].replace([np.inf, -np.inf], np.nan).dropna()
    out['heterogeneity'] = {'status': 'not_identifiable', 'reason': 'dataset is constant within patient, so dataset × treatment is collinear with treatment under patient fixed effects'}
    return out

def meta_pair(a: dict, b: dict, labels: tuple[str, str]=('HiRID', 'SICdb')) -> dict:
    vals = np.array([a['estimate'], b['estimate']], dtype=float)
    ses = np.array([a['se'], b['se']], dtype=float)
    w = 1.0 / ses ** 2
    fixed = float(np.sum(w * vals) / np.sum(w))
    fixed_se = float(1.0 / np.sqrt(np.sum(w)))
    q = float(np.sum(w * (vals - fixed) ** 2))
    c = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (q - 1.0) / c) if c else 0.0
    wr = 1.0 / (ses ** 2 + tau2)
    random = float(np.sum(wr * vals) / np.sum(wr))
    random_se = float(1.0 / np.sqrt(np.sum(wr)))
    diff = float(vals[0] - vals[1])
    diff_se = float(np.sqrt(np.sum(ses ** 2)))
    z = diff / diff_se if diff_se else np.nan
    p = float(erf(abs(z) / np.sqrt(2))) if np.isfinite(z) else None
    return {'studies': {labels[0]: a, labels[1]: b}, 'difference_HiRID_minus_SICdb': {'estimate': diff, 'se': diff_se, 'ci95_low': diff - 1.96 * diff_se, 'ci95_high': diff + 1.96 * diff_se, 'two_sided_p': 1 - p if p is not None else None}, 'fixed_effect': {'estimate': fixed, 'se': fixed_se, 'ci95_low': fixed - 1.96 * fixed_se, 'ci95_high': fixed + 1.96 * fixed_se}, 'random_effects_DL': {'estimate': random, 'se': random_se, 'ci95_low': random - 1.96 * random_se, 'ci95_high': random + 1.96 * random_se, 'tau2': tau2, 'Q': q}}

def main() -> None:
    loaded = {}
    missing = []
    for name, path in SOURCES.items():
        if not path.exists():
            missing.append(name)
            continue
        _, raw = load_events(path)
        if raw.empty:
            missing.append(name)
            continue
        loaded[name] = normalize(name, raw)
    result = {'missing': missing, 'cohorts': {}, 'identified_models': {}, 'sensitivity': {}, 'smd': {}, 'selection_auc': {}, 'selection_auc_state_only': {}, 'identifiability': {}}
    for name, x in loaded.items():
        result['cohorts'][name] = {'n': len(x), 'patients': int(x.patient.nunique())}
        result['identified_models'][name] = identified_models(x)
        result['identifiability'][name] = {'treatment_plus_previous_unique': sorted((x.treatment + x.previous_treatment).dropna().unique().tolist()), 'violations': int((x.treatment + x.previous_treatment != 1).sum())}
        result['sensitivity'][name] = sensitivity(name, x)
        smd_cols = ['pre_map', 'pre_map_slope', 'pre_temp_use', 'ne', 'ne_change', 'gap_hours', 'met_24h', 'para_24h', 'icu_hours', 'previous_treatment', 'pressor_other']
        result['smd'][name] = {c: smd(x, c) for c in smd_cols if c in x}
        result['selection_auc'][name] = selection_auc(x, include_sequence=True)
        result['selection_auc_state_only'][name] = selection_auc(x, include_sequence=False)
    if 'hirid_no_pressor' in loaded and 'sicdb_no_pressor' in loaded:
        result['combined_no_pressor'] = combined(loaded['hirid_no_pressor'], loaded['sicdb_no_pressor'], 'no_pressor')
        result['between_dataset_no_pressor'] = {'map': meta_pair(result['identified_models']['hirid_no_pressor']['map'], result['identified_models']['sicdb_no_pressor']['map']), 'binary_ne_up_or_new_pressor': meta_pair(result['identified_models']['hirid_no_pressor']['binary_ne_up_or_new_pressor'], result['identified_models']['sicdb_no_pressor']['binary_ne_up_or_new_pressor'])}
    if 'hirid_pressor' in loaded and 'sicdb_pressor' in loaded:
        result['combined_pressor'] = combined(loaded['hirid_pressor'], loaded['sicdb_pressor'], 'pressor')
        result['between_dataset_pressor'] = {'map': meta_pair(result['identified_models']['hirid_pressor']['map'], result['identified_models']['sicdb_pressor']['map']), 'binary_ne_up_or_new_pressor': meta_pair(result['identified_models']['hirid_pressor']['binary_ne_up_or_new_pressor'], result['identified_models']['sicdb_pressor']['binary_ne_up_or_new_pressor'])}
    (OUT / 'metamizole_followup_results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(json.dumps({'missing': missing, 'cohorts': result['cohorts'], 'selection_auc': result['selection_auc'], 'selection_auc_state_only': result['selection_auc_state_only']}, ensure_ascii=False))
if __name__ == '__main__':
    main()
