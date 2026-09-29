from __future__ import annotations
import json
import os
from pathlib import Path
from config import DUCKDB_TEMP, HIRID_MAP_GLOB, HIRID_ROOT, RESULTS_DIR, SICDB_ROOT
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
ROOT = HIRID_ROOT
OUT = RESULTS_DIR
OUT.mkdir(parents=True, exist_ok=True)
PHARMA_GLOB = str(ROOT / 'raw_stage' / 'pharma_records' / 'csv' / 'part-*.csv').replace('\\', '/')
OBS_GLOB = str(ROOT / 'raw_stage' / 'observation_tables' / 'csv' / 'part-*.csv').replace('\\', '/')
MAP_GLOB = HIRID_MAP_GLOB
GENERAL_CSV = str(ROOT / 'reference' / 'general_table.csv').replace('\\', '/')
CSV_COLUMNS = "{patientid: 'INTEGER', pharmaid: 'INTEGER', givenat: 'TIMESTAMP', enteredentryat: 'TIMESTAMP', givendose: 'DOUBLE', cumulativedose: 'DOUBLE', fluidamount_calc: 'DOUBLE', cumulfluidamount_calc: 'DOUBLE', doseunit: 'VARCHAR', route: 'VARCHAR', infusionid: 'VARCHAR', typeid: 'INTEGER', subtypeid: 'DOUBLE', recordstatus: 'INTEGER'}"
OBS_COLUMNS = "{datetime: 'TIMESTAMP', entertime: 'TIMESTAMP', patientid: 'INTEGER', status: 'INTEGER', stringvalue: 'VARCHAR', type: 'VARCHAR', value: 'DOUBLE', variableid: 'INTEGER'}"
MET_ID = 1000605
PARA_IDS = (1000489, 1000490)
TARGET = (MET_ID,) + PARA_IDS
NE_IDS = (1000462, 1000656, 1000657, 1000658)
EPI_IDS = (71, 1000750, 1000649, 1000650, 1000655)
VASO_IDS = (112, 113)
PRESSORS = NE_IDS + EPI_IDS + VASO_IDS
MODE = os.environ.get('HIRID_PRESSOR_MODE', 'primary').lower()
if MODE not in {'primary', 'pre10', 'bidirectional', '180', 'fullcase'}:
    raise ValueError('HIRID_PRESSOR_MODE must be primary, pre10, bidirectional, 180, or fullcase')
con = duckdb.connect()
con.execute("SET memory_limit='768MB'")
con.execute('SET threads=2')
con.execute('SET preserve_insertion_order=false')
con.execute(f"SET temp_directory='{DUCKDB_TEMP.as_posix()}'")
con.execute(f"\n    CREATE TEMP TABLE target AS\n    SELECT ROW_NUMBER() OVER () AS target_id,\n           patientid::INTEGER AS patientid,\n           pharmaid::INTEGER AS pharmaid,\n           CASE WHEN pharmaid={MET_ID} THEN 'metamizole' ELSE 'paracetamol' END AS drug,\n           givenat::TIMESTAMP AS givenat,\n           route, givendose::DOUBLE AS givendose,\n           infusionid::VARCHAR AS infusionid\n    FROM read_csv('{PHARMA_GLOB}', union_by_name=true, header=true, columns={CSV_COLUMNS}, ignore_errors=true)\n    WHERE pharmaid IN {TARGET}\n      AND givenat IS NOT NULL\n      AND route IN ('iv-inj', 'iv-inf', 'cv-inj', 'cv-inf')\n      AND givendose > 0\n      AND (recordstatus & 2) = 0 AND (recordstatus & 32) = 0\n    ")
con.execute(f"\n    CREATE TEMP TABLE pressor_raw AS\n    WITH x AS (\n      SELECT patientid::INTEGER AS patientid, pharmaid::INTEGER AS pharmaid,\n             givenat::TIMESTAMP AS givenat, givendose::DOUBLE AS givendose,\n             doseunit::VARCHAR AS doseunit, infusionid::VARCHAR AS infusionid,\n             LAG(givenat) OVER (PARTITION BY patientid, infusionid ORDER BY givenat) AS prev_at\n      FROM read_csv('{PHARMA_GLOB}', union_by_name=true, header=true, columns={CSV_COLUMNS}, ignore_errors=true)\n      WHERE pharmaid IN {PRESSORS}\n        AND patientid IN (SELECT DISTINCT patientid FROM target)\n        AND givenat IS NOT NULL AND givendose > 0\n        AND route IN ('cv-inf', 'iv-inf')\n        AND (recordstatus & 2) = 0 AND (recordstatus & 32) = 0\n    )\n    SELECT *,\n           CASE WHEN prev_at IS NOT NULL AND EPOCH(givenat-prev_at) > 0\n                     AND EPOCH(givenat-prev_at) <= 30*60\n                THEN givendose / (EPOCH(givenat-prev_at)/60.0)\n                ELSE NULL END AS dose_rate_ug_min\n    FROM x\n    ")
con.execute('\n    CREATE TEMP TABLE pressor_intervals AS\n    SELECT patientid, pharmaid, infusionid, givenat AS start_at,\n           CASE WHEN LEAD(givenat) OVER (PARTITION BY patientid, infusionid ORDER BY givenat) IS NOT NULL\n                     AND LEAD(givenat) OVER (PARTITION BY patientid, infusionid ORDER BY givenat) <= givenat + INTERVAL 15 MINUTE\n                THEN LEAD(givenat) OVER (PARTITION BY patientid, infusionid ORDER BY givenat)\n                ELSE givenat + INTERVAL 10 MINUTE END AS stop_at\n    FROM pressor_raw\n    ')
con.execute('\n    CREATE TEMP TABLE timeline AS\n    SELECT t.*,\n           LAG(t.drug) OVER (PARTITION BY t.patientid ORDER BY t.givenat, t.target_id) AS previous_drug,\n           LAG(t.givenat) OVER (PARTITION BY t.patientid ORDER BY t.givenat, t.target_id) AS previous_at,\n           LEAD(t.givenat) OVER (PARTITION BY t.patientid ORDER BY t.givenat, t.target_id) AS next_at,\n           CASE WHEN EXISTS (\n             SELECT 1 FROM pressor_intervals p\n             WHERE p.patientid=t.patientid AND p.start_at<=t.givenat AND t.givenat<p.stop_at\n           ) THEN 1 ELSE 0 END AS pressor_active\n    FROM target t\n    ')
con.execute('\n    CREATE TEMP TABLE edges AS\n    SELECT t.target_id AS current_id, t.patientid, t.drug AS current_drug,\n           t.previous_drug, t.givenat AS current_at, t.previous_at, t.next_at,\n           EPOCH(t.givenat-t.previous_at)/60.0 AS prev_gap_minutes,\n           EPOCH(t.next_at-t.givenat)/60.0 AS next_gap_minutes,\n           t.pressor_active\n    FROM timeline t\n    WHERE t.pressor_active=1\n      AND t.previous_drug IS NOT NULL AND t.drug<>t.previous_drug\n      AND t.previous_at IS NOT NULL AND t.next_at IS NOT NULL\n      AND EPOCH(t.givenat-t.previous_at)/60.0 >= 120\n      AND EPOCH(t.next_at-t.givenat)/60.0 >= 120\n    ')
EDGE_COUNT = int(con.execute('SELECT COUNT(*) FROM edges').fetchone()[0])
con.execute(f"\n    CREATE TEMP TABLE map_temp AS\n    SELECT e.*,\n           AVG(m.vm5) FILTER (WHERE m.datetime>=e.current_at-INTERVAL 30 MINUTE AND m.datetime<e.current_at) AS pre_map,\n           REGR_SLOPE(m.vm5, EPOCH(m.datetime-e.current_at)) FILTER (WHERE m.datetime>=e.current_at-INTERVAL 30 MINUTE AND m.datetime<e.current_at) AS pre_map_slope_min,\n           AVG(m.vm5) FILTER (WHERE m.datetime>=e.current_at+INTERVAL 30 MINUTE AND m.datetime<e.current_at+INTERVAL 120 MINUTE) AS post_map_30_120,\n           AVG(m.vm5) FILTER (WHERE m.datetime>=e.current_at-INTERVAL 60 MINUTE AND m.datetime<e.current_at-INTERVAL 30 MINUTE) AS placebo_pre_60_30,\n           AVG(m.vm5) FILTER (WHERE m.datetime>=e.current_at-INTERVAL 30 MINUTE AND m.datetime<e.current_at) AS placebo_pre_30_0,\n           ARG_MAX(m.vm2,m.datetime) FILTER (WHERE m.datetime>=e.current_at-INTERVAL 4 HOUR AND m.datetime<e.current_at AND m.vm2 IS NOT NULL) AS temp_recent_4h,\n           AVG(m.vm2) FILTER (WHERE m.datetime>=e.current_at-INTERVAL 30 MINUTE AND m.datetime<e.current_at) AS temp_fullcase,\n           AVG(m.vm2) FILTER (WHERE m.datetime>=e.current_at+INTERVAL 90 MINUTE AND m.datetime<=e.current_at+INTERVAL 120 MINUTE) AS temp_post_90_120\n    FROM edges e JOIN read_parquet('{MAP_GLOB}') m\n      ON m.patientid=e.patientid\n     AND m.datetime BETWEEN e.current_at-INTERVAL 4 HOUR AND e.current_at+INTERVAL 120 MINUTE\n    GROUP BY ALL\n    ")
con.execute(f"\n    CREATE TEMP TABLE weight_obs AS\n    SELECT patientid::INTEGER AS patientid, datetime::TIMESTAMP AS datetime, value::DOUBLE AS weight_kg\n    FROM read_csv('{OBS_GLOB}', union_by_name=true, header=true, columns={OBS_COLUMNS}, ignore_errors=true)\n    WHERE variableid=10000400 AND value BETWEEN 20 AND 300\n    ")
con.execute('\n    CREATE TEMP TABLE edge_doses AS\n    SELECT e.current_id,\n           w.weight_kg,\n           n0.dose_rate_ug_min / NULLIF(w.weight_kg,0) AS ne_current,\n           n30.dose_rate_ug_min / NULLIF(w.weight_kg,0) AS ne_30m,\n           npost.post_ne_avg,\n           npost.post_ne_any_up,\n           npost.post_ne_present,\n           npost.post_epi_present,\n           npost.post_vaso_present,\n           CASE WHEN n0.dose_rate_ug_min IS NOT NULL AND n30.dose_rate_ug_min IS NOT NULL\n                THEN (n0.dose_rate_ug_min-n30.dose_rate_ug_min)/NULLIF(w.weight_kg,0) END AS ne_change_pre30,\n           CASE WHEN n0.dose_rate_ug_min IS NOT NULL AND npost.post_ne_avg IS NOT NULL\n                THEN npost.post_ne_avg - n0.dose_rate_ug_min/NULLIF(w.weight_kg,0) END AS ne_change_0_120\n    FROM edges e\n    LEFT JOIN LATERAL (\n      SELECT weight_kg FROM weight_obs w\n      WHERE w.patientid=e.patientid AND w.datetime<=e.current_at AND w.datetime>=e.current_at-INTERVAL 7 DAY\n      ORDER BY w.datetime DESC LIMIT 1\n    ) w ON TRUE\n    LEFT JOIN LATERAL (\n      SELECT dose_rate_ug_min FROM pressor_raw p\n      WHERE p.patientid=e.patientid AND p.pharmaid IN (1000462,1000656,1000657,1000658)\n        AND p.givenat<=e.current_at AND p.givenat>e.current_at-INTERVAL 30 MINUTE AND p.dose_rate_ug_min IS NOT NULL\n      ORDER BY p.givenat DESC LIMIT 1\n    ) n0 ON TRUE\n    LEFT JOIN LATERAL (\n      SELECT dose_rate_ug_min FROM pressor_raw p\n      WHERE p.patientid=e.patientid AND p.pharmaid IN (1000462,1000656,1000657,1000658)\n        AND p.givenat<=e.current_at-INTERVAL 30 MINUTE AND p.givenat>e.current_at-INTERVAL 60 MINUTE AND p.dose_rate_ug_min IS NOT NULL\n      ORDER BY p.givenat DESC LIMIT 1\n    ) n30 ON TRUE\n    LEFT JOIN LATERAL (\n      SELECT AVG(CASE WHEN p.pharmaid IN (1000462,1000656,1000657,1000658) THEN p.dose_rate_ug_min/NULLIF(w.weight_kg,0) END) AS post_ne_avg,\n             MAX(CASE WHEN p.pharmaid IN (1000462,1000656,1000657,1000658) THEN 1 ELSE 0 END) AS post_ne_present,\n             MAX(CASE WHEN p.pharmaid IN (1000462,1000656,1000657,1000658) AND n0.dose_rate_ug_min IS NOT NULL AND p.dose_rate_ug_min > n0.dose_rate_ug_min*1.10 THEN 1 ELSE 0 END) AS post_ne_any_up,\n             MAX(CASE WHEN p.pharmaid IN (71,1000750,1000649,1000650,1000655) THEN 1 ELSE 0 END) AS post_epi_present,\n             MAX(CASE WHEN p.pharmaid IN (112,113) THEN 1 ELSE 0 END) AS post_vaso_present\n      FROM pressor_raw p\n      WHERE p.patientid=e.patientid AND p.givenat>e.current_at AND p.givenat<=e.current_at+INTERVAL 120 MINUTE\n        AND p.dose_rate_ug_min IS NOT NULL\n    ) npost ON TRUE\n    ')
con.execute('\n    CREATE TEMP TABLE edge_context AS\n    SELECT e.current_id,\n           MAX(CASE WHEN p.pharmaid IN (71,1000750,1000649,1000650,1000655) THEN 1 ELSE 0 END) AS epi_active,\n           MAX(CASE WHEN p.pharmaid IN (112,113) THEN 1 ELSE 0 END) AS vaso_active\n    FROM edges e LEFT JOIN pressor_intervals p\n      ON p.patientid=e.patientid AND p.start_at<=e.current_at AND e.current_at<p.stop_at\n    GROUP BY e.current_id\n    ')
con.execute("\n    CREATE TEMP TABLE edge_counts AS\n    SELECT e.current_id,\n           COUNT(*) FILTER (WHERE t.givenat>=e.current_at-INTERVAL 24 HOUR AND t.givenat<e.current_at AND t.drug='metamizole') AS met_24h,\n           COUNT(*) FILTER (WHERE t.givenat>=e.current_at-INTERVAL 24 HOUR AND t.givenat<e.current_at AND t.drug='paracetamol') AS para_24h\n    FROM edges e LEFT JOIN target t ON t.patientid=e.patientid\n    GROUP BY e.current_id\n    ")
con.execute(f"\n    CREATE TEMP TABLE edge_outcomes AS\n    SELECT m.*, g.admissiontime,\n           (m.post_map_30_120-m.pre_map) AS map_change,\n           (m.placebo_pre_30_0-m.placebo_pre_60_30) AS placebo_change,\n           (m.temp_post_90_120-m.temp_recent_4h) AS temp_change,\n           d.weight_kg, d.ne_current, d.ne_30m, d.ne_change_pre30, d.ne_change_0_120,\n           d.post_ne_avg, d.post_ne_any_up, d.post_ne_present, d.post_epi_present, d.post_vaso_present,\n           c.epi_active, c.vaso_active,\n           CASE WHEN COALESCE(d.post_ne_any_up,0)=1\n                     OR (COALESCE(d.ne_current,0)=0 AND COALESCE(d.post_ne_present,0)=1)\n                     OR (c.epi_active=0 AND COALESCE(d.post_epi_present,0)=1)\n                     OR (c.vaso_active=0 AND COALESCE(d.post_vaso_present,0)=1)\n                THEN 1 ELSE 0 END AS any_ne_up_or_new_pressor,\n           COALESCE(ec.met_24h,0) AS met_24h, COALESCE(ec.para_24h,0) AS para_24h,\n           EPOCH(m.current_at-g.admissiontime)/3600.0 AS icu_hours\n    FROM map_temp m\n    LEFT JOIN edge_doses d ON d.current_id=m.current_id\n    LEFT JOIN edge_context c ON c.current_id=m.current_id\n    LEFT JOIN edge_counts ec ON ec.current_id=m.current_id\n    LEFT JOIN read_csv('{GENERAL_CSV}', header=true, union_by_name=true) g ON g.patientid=m.patientid\n    ")
events = con.execute('SELECT * FROM edge_outcomes').df()
events['treatment'] = (events.current_drug == 'metamizole').astype(float)
events['previous_treatment'] = (events.previous_drug == 'metamizole').astype(float)
events['prev_gap_hours'] = events.prev_gap_minutes / 60.0
events['pre_map_slope_hour'] = events.pre_map_slope_min * 60.0
events['pre10_exclude'] = (events.ne_current.notna() & events.ne_30m.notna() & (events.ne_change_pre30.abs() / events.ne_30m.abs().replace(0, np.nan) >= 0.1)).fillna(False)
BASE_COVARS = ['treatment', 'previous_treatment', 'prev_gap_hours', 'met_24h', 'para_24h', 'pre_map', 'pre_map_slope_hour', 'temp_recent_4h', 'ne_current', 'ne_change_pre30', 'epi_active', 'vaso_active', 'icu_hours']

def fixed_effects(df: pd.DataFrame, outcome: str) -> dict:
    cols = ['patientid', outcome] + BASE_COVARS
    x = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) == 0 or x.patientid.nunique() < 2:
        return {'status': 'insufficient', 'outcome': outcome, 'n': int(len(x)), 'patients': int(x.patientid.nunique())}
    demean = [outcome] + BASE_COVARS
    g = x.groupby('patientid', sort=False)
    xd = x.copy()
    xd[demean] = xd[demean] - g[demean].transform('mean')
    fit = sm.OLS(xd[outcome], xd[BASE_COVARS]).fit(cov_type='cluster', cov_kwds={'groups': xd.patientid})
    coef = float(fit.params['treatment'])
    se = float(fit.bse['treatment'])
    return {'status': 'ok', 'outcome': outcome, 'n': int(len(x)), 'patients': int(x.patientid.nunique()), 'estimate': coef, 'se': se, 'ci95_low': coef - 1.96 * se, 'ci95_high': coef + 1.96 * se, 'covariates': BASE_COVARS}

def binary_counts(df: pd.DataFrame, outcome: str) -> dict:
    x = df[['patientid', 'current_drug', outcome] + BASE_COVARS[1:]].replace([np.inf, -np.inf], np.nan).dropna()
    by = x.groupby('current_drug')[outcome].agg(n='size', events='sum').to_dict('index')
    p = x.groupby(['patientid', 'current_drug'])[outcome].agg(['size', 'sum']).reset_index()
    both = p.groupby('patientid')['current_drug'].nunique()
    both_ids = both[both == 2].index
    q = p[p.patientid.isin(both_ids)].assign(any_event=lambda z: (z['sum'] > 0).astype(int)).pivot(index='patientid', columns='current_drug', values='any_event').dropna()
    discordant = int((q['metamizole'] != q['paracetamol']).sum()) if not q.empty else 0
    return {'n': int(len(x)), 'patients': int(x.patientid.nunique()), 'by_drug': by, 'patients_with_both': int(len(both_ids)), 'discordant_patients': discordant}

def eq(model: dict, margin: float) -> dict:
    if model.get('status') != 'ok':
        return {'status': 'not_judged', 'margin': margin}
    lo, hi = (model['ci95_low'], model['ci95_high'])
    return {'status': 'ci_within_margin' if lo >= -margin and hi <= margin else 'ci_crosses_margin', 'margin': margin, 'ci95_low': lo, 'ci95_high': hi}

def cohort(df: pd.DataFrame) -> dict:
    return {'n': int(len(df)), 'patients': int(df.patientid.nunique()), 'metamizole': int((df.current_drug == 'metamizole').sum()), 'paracetamol': int((df.current_drug == 'paracetamol').sum()), 'transition_edges': int(len(df))}

def choose_cohort() -> pd.DataFrame:
    df = events.copy()
    if MODE == 'pre10':
        return df[~df.pre10_exclude].copy()
    if MODE == 'bidirectional':
        pairs = df.groupby('patientid')['previous_drug'].agg(lambda s: set(s.dropna())).to_dict()
        dirs = df.assign(direction=df.previous_drug + '->' + df.current_drug).groupby('patientid')['direction'].agg(lambda s: set(s)).to_dict()
        keep = {pid for pid, d in dirs.items() if 'metamizole->paracetamol' in d and 'paracetamol->metamizole' in d}
        return df[df.patientid.isin(keep)].copy()
    if MODE == '180':
        return df[(df.prev_gap_minutes >= 180) & (df.next_gap_minutes >= 180)].copy()
    if MODE == 'fullcase':
        out = df[df.temp_fullcase.notna()].copy()
        out['temp_recent_4h'] = out['temp_fullcase']
        return out
    return df
analysis = {'mode': MODE, 'outcome_blind_before_freeze': False, 'cohort': cohort(choose_cohort()), 'dose_definition': 'givendose microgram increment divided by within-infusion elapsed minutes, then divided by latest body weight observation within 7 days; all NE rates reported in ug/kg/min', 'models': {}, 'sensitivity': {}}
df = choose_cohort()
analysis['models']['map_change'] = fixed_effects(df, 'map_change')
analysis['models']['map_change']['equivalence'] = eq(analysis['models']['map_change'], 3.0)
analysis['models']['ne_change_0_120'] = fixed_effects(df, 'ne_change_0_120')
analysis['models']['ne_change_0_120']['equivalence'] = eq(analysis['models']['ne_change_0_120'], 0.02)
analysis['models']['temp_change'] = fixed_effects(df, 'temp_change')
analysis['models']['placebo_map'] = fixed_effects(df, 'placebo_change')
analysis['models']['binary_ne_up_or_new_pressor_lpm'] = fixed_effects(df, 'any_ne_up_or_new_pressor')
analysis['binary_event_counts'] = binary_counts(df, 'any_ne_up_or_new_pressor')
analysis['sensitivity']['pre10_excluded'] = {'cohort': cohort(events[~events.pre10_exclude]), 'run_mode': 'pre10'}
analysis['sensitivity']['bidirectional'] = {'run_mode': 'bidirectional'}
analysis['sensitivity']['washout_180'] = {'run_mode': '180'}
analysis['sensitivity']['fullcase_temperature'] = {'run_mode': 'fullcase'}
analysis['coverage'] = {'candidate_edges_before_outcome_windows': EDGE_COUNT, 'weight_present': int(events.weight_kg.notna().sum()), 'ne_current_present': int(events.ne_current.notna().sum()), 'ne_pre30_change_present': int(events.ne_change_pre30.notna().sum()), 'temp_recent4h_present': int(events.temp_recent_4h.notna().sum())}
analysis['sample_audit_crosscheck'] = {'patients': 580, 'bidirectional_patients': 327, 'participating_administrations': 2836, 'transition_edges': 1909, 'note': 'outcome-blind sample audit counts endpoints; outcome models use current-side transition edges'}
result = {'dataset': 'HiRID 1.1.1', 'mode': MODE, 'outcome_blind_before_freeze': False, 'analysis': analysis, 'event_rows': df.to_dict(orient='records')}
outfile = OUT / ('hirid_metamizole_pressor_analysis.json' if MODE == 'primary' else f'hirid_metamizole_pressor_analysis_{MODE}.json')
outfile.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
print(json.dumps({'mode': MODE, 'cohort': analysis['cohort'], 'models': analysis['models'], 'coverage': analysis['coverage']}, indent=2, ensure_ascii=True, default=str))
