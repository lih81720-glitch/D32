from __future__ import annotations
import bisect
import csv
import gzip
import json
import struct
from pathlib import Path
from config import DUCKDB_TEMP, HIRID_MAP_GLOB, HIRID_ROOT, RESULTS_DIR, SICDB_ROOT
import numpy as np
import pandas as pd
import statsmodels.api as sm
ROOT = SICDB_ROOT
OUT = RESULTS_DIR
MED = ROOT / 'medication.csv.gz'
CASES = ROOT / 'cases.csv.gz'
FLOAT_H = ROOT / 'data_float_h.csv.gz'
MET_ID, PARA_ID = (1610, 1420)
NE_ID, VASO_ID, EPI_ID = (1562, 1550, 1502)
TARGET_IDS = {MET_ID, PARA_ID}
PRESSOR_IDS = {NE_ID, VASO_ID, EPI_ID}

def load_medication() -> pd.DataFrame:
    cols = ['CaseID', 'DrugID', 'Offset', 'OffsetDrugEnd', 'IsSingleDose', 'AmountPerMinute']
    chunks = []
    for chunk in pd.read_csv(MED, compression='gzip', usecols=cols, chunksize=300000):
        x = chunk[chunk.DrugID.isin(TARGET_IDS | PRESSOR_IDS)].copy()
        if not x.empty:
            chunks.append(x)
    return pd.concat(chunks, ignore_index=True)

def make_cohort(med: pd.DataFrame, active_value: int=1) -> tuple[pd.DataFrame, dict]:
    target = med[med.DrugID.isin(TARGET_IDS)].copy()
    target['drug'] = target.DrugID.map({MET_ID: 'metamizole', PARA_ID: 'paracetamol'})
    target = target.sort_values(['CaseID', 'Offset', 'DrugID']).reset_index(drop=True)
    press = med[med.DrugID.isin(PRESSOR_IDS)].copy()
    press['start'] = press.Offset.astype(float)
    press['end'] = np.maximum(press.Offset.astype(float) + 60, press.OffsetDrugEnd.astype(float))
    press_by_case = {int(k): v.sort_values('start') for k, v in press.groupby('CaseID')}
    active = np.zeros(len(target), dtype=np.int8)
    for case, idx in target.groupby('CaseID', sort=False).groups.items():
        p = press_by_case.get(int(case))
        if p is None:
            continue
        starts = np.sort(p.start.to_numpy())
        ends = np.sort(p.end.to_numpy())
        si = ei = count = 0
        for row_idx, t in zip(np.asarray(idx), target.loc[idx, 'Offset'].to_numpy()):
            while si < len(starts) and starts[si] <= t:
                count += 1
                si += 1
            while ei < len(ends) and ends[ei] <= t:
                count -= 1
                ei += 1
            active[row_idx] = int(count > 0)
    target['pressor_active'] = active
    target['previous_drug'] = target.groupby('CaseID').drug.shift(1)
    target['previous_offset'] = target.groupby('CaseID').Offset.shift(1)
    target['next_offset'] = target.groupby('CaseID').Offset.shift(-1)
    target['prev_gap_sec'] = target.Offset - target.previous_offset
    target['next_gap_sec'] = target.next_offset - target.Offset
    edges = target[(target.pressor_active == active_value) & target.previous_drug.notna() & (target.drug != target.previous_drug) & (target.prev_gap_sec >= 7200) & (target.next_gap_sec >= 7200)].copy()
    edges = edges.reset_index(drop=True)
    pairs = edges.groupby('CaseID').apply(lambda x: set(zip(x.previous_drug, x.drug)) == {('metamizole', 'paracetamol'), ('paracetamol', 'metamizole')}, include_groups=False)
    audit = {'target_records': int(len(target)), 'target_cases': int(target.CaseID.nunique()), 'pressor_records': int(len(press)), 'pressor_active_target_records': int((target.pressor_active == active_value).sum()), 'strict_120_edges': int(len(edges)), 'strict_120_patients': int(edges.CaseID.nunique()), 'strict_120_bidirectional_patients': int(pairs.sum())}
    return (edges, audit)

def interval_rate(rows: pd.DataFrame, t: float, drug_id: int) -> float | None:
    x = rows[(rows.DrugID == drug_id) & (rows.start <= t) & (rows.end > t)]
    if x.empty:
        return None
    return float(x.AmountPerMinute.fillna(0).sum())

def interval_avg(rows: pd.DataFrame, start: float, end: float, drug_id: int, weight: float | None) -> float | None:
    x = rows[(rows.DrugID == drug_id) & (rows.end > start) & (rows.start < end)].copy()
    if x.empty or weight is None or (not np.isfinite(weight)) or (weight <= 0):
        return None
    overlap = (np.minimum(x.end, end) - np.maximum(x.start, start)).clip(lower=0)
    den = float(overlap.sum())
    if den <= 0:
        return None
    return float((x.AmountPerMinute.fillna(0) * overlap).sum() / den / weight * 1000000)

def add_dose_covariates(edges: pd.DataFrame, med: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    press = med[med.DrugID.isin(PRESSOR_IDS)].copy()
    press['start'] = press.Offset.astype(float)
    press['end'] = np.maximum(press.Offset.astype(float) + 60, press.OffsetDrugEnd.astype(float))
    press_by_case = {int(k): v for k, v in press.groupby('CaseID')}
    weights = cases.set_index('CaseID')['WeightOnAdmission'].astype(float).div(1000).to_dict()
    out = []
    for row in edges.itertuples(index=False):
        p = press_by_case.get(int(row.CaseID), pd.DataFrame(columns=press.columns))
        weight = weights.get(int(row.CaseID))
        t = float(row.Offset)
        current = interval_rate(p, t, NE_ID)
        prev = interval_rate(p, t - 1800, NE_ID)
        ne_current = None if current is None or not weight else current / weight * 1000000
        ne_prev = None if prev is None or not weight else prev / weight * 1000000
        post_avg = interval_avg(p, t, t + 7200, NE_ID, weight)
        post_rows = p[(p.DrugID == NE_ID) & (p.end > t) & (p.start <= t + 7200)]
        post_up = int(ne_current is not None and (not post_rows.empty) and (post_rows.AmountPerMinute / weight * 1000000 > ne_current * 1.1).any())
        out.append({'weight_kg': weight, 'ne_current': ne_current, 'ne_30m': ne_prev, 'ne_change_pre30': None if ne_current is None or ne_prev is None else ne_current - ne_prev, 'ne_change_0_120': None if ne_current is None or post_avg is None else post_avg - ne_current, 'post_ne_avg': post_avg, 'post_ne_any_up': post_up, 'post_ne_present': int(post_avg is not None), 'post_epi_present': int(not p[(p.DrugID == EPI_ID) & (p.end > t) & (p.start <= t + 7200)].empty), 'post_vaso_present': int(not p[(p.DrugID == VASO_ID) & (p.end > t) & (p.start <= t + 7200)].empty), 'epi_active': int(not p[(p.DrugID == EPI_ID) & (p.start <= t) & (p.end > t)].empty), 'vaso_active': int(not p[(p.DrugID == VASO_ID) & (p.start <= t) & (p.end > t)].empty)})
    return pd.concat([edges.reset_index(drop=True), pd.DataFrame(out)], axis=1)

def decode_raw(raw: str | None, val: float | None, offset: int) -> list[tuple[int, float]]:
    if raw and raw.startswith('0x'):
        try:
            data = bytes.fromhex(raw[2:])
            if len(data) == 240:
                vals = struct.unpack('<60f', data)
                return [(offset + 60 * i, float(v)) for i, v in enumerate(vals) if v != 0.0 and np.isfinite(v)]
        except (ValueError, struct.error):
            pass
    return [] if val is None or not np.isfinite(val) else [(offset, float(val))]

def stream_vitals(events: pd.DataFrame) -> pd.DataFrame:
    case_map: dict[int, list[int]] = {}
    for i, case in enumerate(events.CaseID.astype(int)):
        case_map.setdefault(case, []).append(i)
    case_offsets = {c: sorted(events.loc[idx, 'Offset'].astype(float).tolist()) for c, idx in case_map.items()}
    buffers = [{'map_pre': [], 'map_post': [], 'map_p1': [], 'map_p2': [], 'temp_recent': None, 'temp_full': [], 'temp_post': []} for _ in range(len(events))]
    with gzip.open(FLOAT_H, 'rt', newline='', encoding='utf-8', errors='replace') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            data_id = row['DataID']
            if data_id not in {'703', '709'}:
                continue
            case = int(row['CaseID'])
            if case not in case_map:
                continue
            base = int(row['Offset'])
            val = float(row['Val']) if row['Val'] not in ('', 'NULL', 'null') else None
            points = decode_raw(row.get('rawdata'), val, base)
            starts = case_offsets[case]
            for point_t, point_v in points:
                lo = bisect.bisect_left(starts, point_t - 7200)
                hi = bisect.bisect_right(starts, point_t + 7200)
                for event_idx in case_map[case][lo:hi]:
                    dt = point_t - float(events.iloc[event_idx].Offset)
                    b = buffers[event_idx]
                    if data_id == '703':
                        if -1800 <= dt < 0:
                            b['map_pre'].append((dt, point_v))
                            b['map_p2'].append(point_v)
                        elif 1800 <= dt < 7200:
                            b['map_post'].append(point_v)
                        elif -3600 <= dt < -1800:
                            b['map_p1'].append(point_v)
                    else:
                        if -14400 <= dt < 0 and (b['temp_recent'] is None or point_t > b['temp_recent'][0]):
                            b['temp_recent'] = (point_t, point_v)
                        if -1800 <= dt < 0:
                            b['temp_full'].append(point_v)
                        elif 5400 <= dt <= 7200:
                            b['temp_post'].append(point_v)
    for i, b in enumerate(buffers):
        pre = b['map_pre']
        events.loc[i, 'pre_map'] = np.mean([v for _, v in pre]) if pre else np.nan
        events.loc[i, 'post_map_30_120'] = np.mean(b['map_post']) if b['map_post'] else np.nan
        events.loc[i, 'placebo_pre_60_30'] = np.mean(b['map_p1']) if b['map_p1'] else np.nan
        events.loc[i, 'placebo_pre_30_0'] = np.mean(b['map_p2']) if b['map_p2'] else np.nan
        events.loc[i, 'pre_map_slope_hour'] = np.polyfit([x / 60 for x, _ in pre], [v for _, v in pre], 1)[0] if len(pre) >= 2 else np.nan
        events.loc[i, 'temp_recent_4h'] = b['temp_recent'][1] if b['temp_recent'] else np.nan
        events.loc[i, 'temp_fullcase'] = np.mean(b['temp_full']) if b['temp_full'] else np.nan
        events.loc[i, 'temp_post_90_120'] = np.mean(b['temp_post']) if b['temp_post'] else np.nan
    events['map_change'] = events.post_map_30_120 - events.pre_map
    events['placebo_change'] = events.placebo_pre_30_0 - events.placebo_pre_60_30
    events['temp_change'] = events.temp_post_90_120 - events.temp_recent_4h
    return events
BASE_COVARS = ['treatment', 'previous_treatment', 'prev_gap_hours', 'met_24h', 'para_24h', 'pre_map', 'pre_map_slope_hour', 'temp_recent_4h', 'ne_current', 'ne_change_pre30', 'epi_active', 'vaso_active', 'icu_hours']

def fixed_effects(df: pd.DataFrame, outcome: str) -> dict:
    cols = ['CaseID', outcome] + BASE_COVARS
    x = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) == 0 or x.CaseID.nunique() < 2:
        return {'status': 'insufficient', 'n': int(len(x)), 'patients': int(x.CaseID.nunique())}
    g = x.groupby('CaseID', sort=False)
    demean = [outcome] + BASE_COVARS
    xd = x.copy()
    xd[demean] = xd[demean] - g[demean].transform('mean')
    fit = sm.OLS(xd[outcome], xd[BASE_COVARS]).fit(cov_type='cluster', cov_kwds={'groups': xd.CaseID})
    coef, se = (float(fit.params['treatment']), float(fit.bse['treatment']))
    return {'status': 'ok', 'outcome': outcome, 'n': int(len(x)), 'patients': int(x.CaseID.nunique()), 'estimate': coef, 'se': se, 'ci95_low': coef - 1.96 * se, 'ci95_high': coef + 1.96 * se}

def eq(m: dict, margin: float) -> dict:
    if m.get('status') != 'ok':
        return {'status': 'not_judged', 'margin': margin}
    return {'status': 'ci_within_margin' if m['ci95_low'] >= -margin and m['ci95_high'] <= margin else 'ci_crosses_margin', 'margin': margin}

def run_mode(df: pd.DataFrame, mode: str) -> dict:
    x = df.copy()
    if mode == 'pre10':
        x = x[~x.pre10_exclude]
    elif mode == 'bidirectional':
        keep = x.groupby('CaseID').apply(lambda z: set(zip(z.previous_drug, z.drug)) == {('metamizole', 'paracetamol'), ('paracetamol', 'metamizole')}, include_groups=False)
        x = x[x.CaseID.isin(keep[keep].index)]
    elif mode == '180':
        x = x[(x.prev_gap_sec >= 10800) & (x.next_gap_sec >= 10800)]
    elif mode == 'fullcase':
        x = x[x.temp_fullcase.notna()].copy()
        x['temp_recent_4h'] = x.temp_fullcase
    models = {}
    for outcome in ['map_change', 'ne_change_0_120', 'temp_change', 'placebo_change']:
        models[outcome] = fixed_effects(x, outcome)
    models['map_change']['equivalence'] = eq(models['map_change'], 3.0)
    models['ne_change_0_120']['equivalence'] = eq(models['ne_change_0_120'], 0.02)
    models['binary_ne_up_or_new_pressor_lpm'] = fixed_effects(x, 'any_ne_up_or_new_pressor')
    return {'mode': mode, 'cohort': {'n': int(len(x)), 'patients': int(x.CaseID.nunique()), 'metamizole': int((x.drug == 'metamizole').sum()), 'paracetamol': int((x.drug == 'paracetamol').sum())}, 'models': models, 'event_rows': x.to_dict(orient='records')}

def main() -> None:
    med = load_medication()
    cases = pd.read_csv(CASES, compression='gzip', usecols=['CaseID', 'WeightOnAdmission'])
    edges, audit = make_cohort(med)
    edges = add_dose_covariates(edges, med, cases)
    edges['treatment'] = (edges.drug == 'metamizole').astype(float)
    edges['previous_treatment'] = (edges.previous_drug == 'metamizole').astype(float)
    edges['prev_gap_hours'] = edges.prev_gap_sec / 3600
    edges['icu_hours'] = edges.Offset / 3600
    target = med[med.DrugID.isin(TARGET_IDS)].sort_values(['CaseID', 'Offset'])
    target_by_case = {int(k): v for k, v in target.groupby('CaseID')}

    def prior_count(row, drug_id):
        z = target_by_case.get(int(row.CaseID))
        if z is None:
            return 0
        return int(((z.Offset >= row.Offset - 86400) & (z.Offset < row.Offset) & (z.DrugID == drug_id)).sum())
    edges['met_24h'] = [prior_count(r, MET_ID) for r in edges.itertuples()]
    edges['para_24h'] = [prior_count(r, PARA_ID) for r in edges.itertuples()]
    edges = stream_vitals(edges)
    edges['pre10_exclude'] = (edges.ne_current.notna() & edges.ne_30m.notna() & (edges.ne_change_pre30.abs() / edges.ne_30m.abs().replace(0, np.nan) >= 0.1)).fillna(False)
    edges['any_ne_up_or_new_pressor'] = ((edges.post_ne_any_up == 1) | (edges.ne_current.fillna(0) == 0) & (edges.post_ne_present == 1) | (edges.epi_active == 0) & (edges.post_epi_present == 1) | (edges.vaso_active == 0) & (edges.post_vaso_present == 1)).astype(int)
    for mode in ['primary', 'pre10', 'bidirectional', '180', 'fullcase']:
        result = {'dataset': 'SICdb 1.0.8', 'mode': mode, 'audit': audit, 'analysis': run_mode(edges, mode)}
        outfile = OUT / ('sicdb_metamizole_pressor_analysis.json' if mode == 'primary' else f'sicdb_metamizole_pressor_analysis_{mode}.json')
        outfile.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
        print(json.dumps({'mode': mode, 'audit': audit, 'cohort': result['analysis']['cohort'], 'models': result['analysis']['models']}, ensure_ascii=True))
if __name__ == '__main__':
    main()
