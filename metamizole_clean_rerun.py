from __future__ import annotations
import json
from pathlib import Path
from config import DUCKDB_TEMP, HIRID_MAP_GLOB, HIRID_ROOT, RESULTS_DIR, SICDB_ROOT
import numpy as np
import pandas as pd
from metamizole_followup import SOURCES, load_events, normalize, fe
ROOT = Path(__file__).resolve().parent
OUT = RESULTS_DIR
COV = ['gap_hours', 'met_24h', 'para_24h', 'pre_map', 'pre_map_slope', 'pre_temp_model', 'ne', 'ne_change', 'pressor_other', 'icu_hours']

def cohort(x: pd.DataFrame) -> dict:
    return {'n': len(x), 'patients': int(x.patient.nunique()), 'metamizole': int(x.treatment.sum()), 'paracetamol': int((1 - x.treatment).sum())}

def binary_rates(x: pd.DataFrame) -> dict:
    cols = ['patient', 'treatment', 'any_ne_up_or_new_pressor', 'new_pressor', 'sustained_map_below65']
    z = x[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=['treatment'])
    out = {}
    for c in cols[2:]:
        if c not in z:
            continue
        out[c] = {}
        for label, m in [('metamizole', z.treatment == 1), ('paracetamol', z.treatment == 0)]:
            q = z.loc[m, c].dropna()
            out[c][label] = {'events': int(q.sum()), 'n': int(len(q)), 'rate': float(q.mean()) if len(q) else None}
    return out

def prepare_special(x: pd.DataFrame, name: str) -> pd.DataFrame:
    x = x.copy()
    if 'sustained_map_below65' not in x:
        x['sustained_map_below65'] = np.nan
    if 'new_pressor' not in x:
        if 'new_ne' in x:
            x['new_pressor'] = (x['new_ne'].fillna(0) == 1).astype(int)
        else:
            ne = (x['ne'].fillna(0) <= 0) & (x.get('post_ne_present', 0) == 1)
            epi = (x.get('epi_active', 0) == 0) & (x.get('post_epi_present', 0) == 1)
            vaso = (x.get('vaso_active', 0) == 0) & (x.get('post_vaso_present', 0) == 1)
            x['new_pressor'] = (ne | epi | vaso).astype(int)
    x['did_map'] = x['map'] - x['placebo']
    return x

def filter_modes(x: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {'primary': x}
    pre10 = x['pre10_exclude'].astype(bool) if 'pre10_exclude' in x else pd.Series(False, index=x.index)
    out['pre10'] = x.loc[~pre10].copy()
    pairs = x.groupby('patient').apply(lambda z: set(zip(z.previous_treatment, z.treatment)), include_groups=False)
    keep = pairs[pairs.apply(lambda s: {(0.0, 1.0), (1.0, 0.0)}.issubset(s))].index
    out['bidirectional'] = x[x.patient.isin(keep)].copy()
    out['min_interval_180'] = x[(x['gap_hours'] >= 3) & (x['next_gap_minutes'] >= 180)].copy()
    out['min_interval_240'] = x[(x['gap_hours'] >= 4) & (x['next_gap_minutes'] >= 240)].copy()
    out['min_interval_360'] = x[(x['gap_hours'] >= 6) & (x['next_gap_minutes'] >= 360)].copy()
    out['clean_6h'] = x[x['clean6']].copy()
    if 'temp_fullcase' in x:
        out['fullcase_temperature'] = x[x.temp_fullcase.notna()].copy()
        out['fullcase_temperature']['pre_temp_model'] = out['fullcase_temperature']['temp_fullcase']
    return out

def run_one(name: str, raw: pd.DataFrame) -> dict:
    x = prepare_special(normalize(name, raw), name)
    modes = filter_modes(x)
    result = {'cohorts': {}, 'models': {}, 'binary_rates': {}, 'did_map': {}, 'order_interaction': {'status': 'not_identifiable', 'reason': 'strict alternating sequence makes treatment × previous treatment constant'}}
    for mode, z in modes.items():
        result['cohorts'][mode] = cohort(z)
        result['models'][mode] = {o: fe(z, o, COV) for o in ['map', 'ne_outcome', 'temp', 'placebo', 'any_ne_up_or_new_pressor', 'sustained_map_below65']}
        result['binary_rates'][mode] = binary_rates(z)
        result['did_map'][mode] = fe(z, 'did_map', COV)
    result['posthoc_new_pressor_trend_adjusted'] = fe(x, 'new_pressor', COV + ['placebo'])
    result['covariates_used'] = ['treatment'] + COV
    result['dropped_for_exact_collinearity'] = ['previous_treatment']
    return result

def main() -> None:
    result = {'specification': 'identifiable_clean_rerun_v1', 'sources': {}, 'datasets': {}}
    for name, path in SOURCES.items():
        _, raw = load_events(path)
        result['sources'][name] = str(path)
        result['datasets'][name] = run_one(name, raw)
    result['posthoc_difference_in_difference'] = {name: {'estimate': d['did_map']['primary']} for name, d in result['datasets'].items()}
    out = OUT / 'metamizole_clean_rerun_results.json'
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(json.dumps({'specification': result['specification'], 'datasets': list(result['datasets']), 'output': str(out)}, ensure_ascii=False))
if __name__ == '__main__':
    main()
