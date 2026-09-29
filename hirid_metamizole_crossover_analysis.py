from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
from config import DUCKDB_TEMP, HIRID_MAP_GLOB, HIRID_ROOT, RESULTS_DIR, SICDB_ROOT
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.conditional_models import ConditionalLogit
ROOT = HIRID_ROOT
OUT = RESULTS_DIR
OUT.mkdir(parents=True, exist_ok=True)
PHARMA_GLOB = str(ROOT / 'raw_stage' / 'pharma_records' / 'csv' / 'part-*.csv').replace('\\', '/')
MAP_GLOB = HIRID_MAP_GLOB
GENERAL_CSV = str(ROOT / 'reference' / 'general_table.csv').replace('\\', '/')
CSV_COLUMNS = "{patientid: 'INTEGER', pharmaid: 'INTEGER', givenat: 'TIMESTAMP', enteredentryat: 'TIMESTAMP', givendose: 'DOUBLE', cumulativedose: 'DOUBLE', fluidamount_calc: 'DOUBLE', cumulfluidamount_calc: 'DOUBLE', doseunit: 'VARCHAR', route: 'VARCHAR', infusionid: 'VARCHAR', typeid: 'INTEGER', subtypeid: 'DOUBLE', recordstatus: 'INTEGER'}"
MET_ID = 1000605
PARA_IDS = (1000489, 1000490)
TARGET = (MET_ID,) + PARA_IDS
PRESSORS = (1000462, 1000656, 1000657, 1000658, 71, 1000750, 1000649, 1000650, 1000655, 112, 113)
NE_IDS = (1000462, 1000656, 1000657, 1000658)
SEDATIVE_BOLUS = (1000700, 1001054, 251, 1001052, 1000691, 1000239, 245, 390, 442, 1000400, 1000857)
BP_LOWERING = (1000763, 1001047, 1000252, 1000695, 1000867, 1000243, 1000868, 1001048, 1001141, 1001142, 117, 1000596, 1000436, 94, 242, 1000477)
MODE = os.environ.get('HIRID_ANALYSIS_MODE', 'primary').lower()
if MODE not in {'primary', '180', 'old'}:
    raise ValueError('HIRID_ANALYSIS_MODE must be primary, 180, or old')
con = duckdb.connect()
con.execute("SET memory_limit='768MB'")
con.execute('SET threads=2')
con.execute('SET preserve_insertion_order=false')
con.execute(f"SET temp_directory='{DUCKDB_TEMP.as_posix()}'")
con.execute(f"\n    CREATE TEMP TABLE target AS\n    SELECT ROW_NUMBER() OVER () AS target_id,\n           patientid::INTEGER AS patientid,\n           pharmaid::INTEGER AS pharmaid,\n           CASE WHEN pharmaid={MET_ID} THEN 'metamizole' ELSE 'paracetamol' END AS drug,\n           givenat::TIMESTAMP AS givenat,\n           route, givendose::DOUBLE AS givendose, infusionid::VARCHAR AS infusionid\n    FROM read_csv('{PHARMA_GLOB}', union_by_name=true, header=true, columns={CSV_COLUMNS}, ignore_errors=true)\n    WHERE pharmaid IN {TARGET} AND givenat IS NOT NULL\n      AND route IN ('iv-inj','iv-inf','cv-inj','cv-inf') AND givendose>0\n      AND (recordstatus & 2)=0 AND (recordstatus & 32)=0\n    ")
con.execute(f"\n    CREATE TEMP TABLE pressor_raw AS\n    SELECT patientid::INTEGER AS patientid, pharmaid::INTEGER AS pharmaid,\n           givenat::TIMESTAMP AS givenat, givendose::DOUBLE AS givendose,\n           infusionid::VARCHAR AS infusionid\n    FROM read_csv('{PHARMA_GLOB}', union_by_name=true, header=true, columns={CSV_COLUMNS}, ignore_errors=true)\n    WHERE pharmaid IN {PRESSORS} AND patientid IN (SELECT DISTINCT patientid FROM target)\n      AND givenat IS NOT NULL AND givendose>0 AND route IN ('cv-inf','iv-inf')\n      AND (recordstatus & 2)=0 AND (recordstatus & 32)=0\n    ")
con.execute(f'\n    CREATE TEMP TABLE pressor_intervals AS\n    WITH x AS (\n      SELECT *, LEAD(givenat) OVER (PARTITION BY patientid, infusionid ORDER BY givenat) AS next_at,\n                   LAG(givendose) OVER (PARTITION BY patientid, infusionid ORDER BY givenat) AS prev_dose\n      FROM pressor_raw\n    )\n    SELECT patientid, pharmaid, infusionid, givenat AS start_at,\n           CASE WHEN next_at IS NOT NULL AND next_at <= givenat+INTERVAL 15 MINUTE\n                THEN next_at ELSE givenat+INTERVAL 10 MINUTE END AS stop_at,\n           CASE WHEN prev_dose IS NOT NULL AND givendose<>prev_dose THEN 1 ELSE 0 END AS dose_change,\n           givendose\n    FROM x\n    ')
con.execute('\n    CREATE TEMP TABLE residual AS\n    SELECT t.target_id,\n           MAX(CASE WHEN o.givenat<t.givenat THEN 1 ELSE 0 END) AS other_prev4h,\n           MAX(CASE WHEN o.givenat>t.givenat THEN 1 ELSE 0 END) AS other_next2h\n    FROM target t LEFT JOIN target o\n      ON o.patientid=t.patientid AND o.drug<>t.drug\n     AND ((o.givenat>=t.givenat-INTERVAL 4 HOUR AND o.givenat<t.givenat)\n       OR (o.givenat>t.givenat AND o.givenat<=t.givenat+INTERVAL 2 HOUR))\n    GROUP BY t.target_id\n    ')
con.execute(f"\n    CREATE TEMP TABLE cointervention AS\n    SELECT t.target_id,\n           MAX(CASE WHEN p.pharmaid::INTEGER IN {SEDATIVE_BOLUS} THEN 1 ELSE 0 END) AS sedative_2h,\n           MAX(CASE WHEN p.pharmaid::INTEGER IN {BP_LOWERING} THEN 1 ELSE 0 END) AS bp_lowering_2h\n    FROM target t LEFT JOIN read_csv('{PHARMA_GLOB}', union_by_name=true, header=true, columns={CSV_COLUMNS}, ignore_errors=true) p\n      ON p.patientid::INTEGER=t.patientid AND p.pharmaid::INTEGER NOT IN {TARGET}\n     AND p.givenat::TIMESTAMP BETWEEN t.givenat-INTERVAL 2 HOUR AND t.givenat+INTERVAL 2 HOUR\n     AND p.givendose::DOUBLE>0\n     AND (p.recordstatus::INTEGER & 2)=0 AND (p.recordstatus::INTEGER & 32)=0\n     AND p.route IN ('iv-inj','iv-inf','cv-inj','cv-inf')\n    GROUP BY t.target_id\n    ")
con.execute(f"\n    CREATE TEMP TABLE timeline AS\n    SELECT t.*, g.admissiontime::TIMESTAMP AS admissiontime,\n           LAG(t.drug) OVER (PARTITION BY t.patientid ORDER BY t.givenat,t.target_id) AS prev_drug,\n           LAG(t.pharmaid) OVER (PARTITION BY t.patientid ORDER BY t.givenat,t.target_id) AS prev_pharmaid,\n           LAG(t.givenat) OVER (PARTITION BY t.patientid ORDER BY t.givenat,t.target_id) AS prev_at,\n           LEAD(t.givenat) OVER (PARTITION BY t.patientid ORDER BY t.givenat,t.target_id) AS next_at,\n           CASE WHEN EXISTS (SELECT 1 FROM pressor_intervals p WHERE p.patientid=t.patientid AND p.start_at<=t.givenat AND t.givenat<p.stop_at) THEN 1 ELSE 0 END AS pressor_active,\n           CASE WHEN EXISTS (SELECT 1 FROM pressor_intervals p WHERE p.patientid=t.patientid AND p.start_at>=t.givenat-INTERVAL 30 MINUTE AND p.start_at<t.givenat AND p.dose_change=1) THEN 1 ELSE 0 END AS pressor_pre30_change,\n           CASE WHEN EXISTS (SELECT 1 FROM pressor_intervals p WHERE p.patientid=t.patientid AND p.start_at BETWEEN t.givenat-INTERVAL 2 HOUR AND t.givenat+INTERVAL 2 HOUR AND p.dose_change=1) THEN 1 ELSE 0 END AS pressor_change_2h,\n           COALESCE(r.other_prev4h,0) AS other_prev4h, COALESCE(r.other_next2h,0) AS other_next2h,\n           COALESCE(c.sedative_2h,0) AS sedative_2h, COALESCE(c.bp_lowering_2h,0) AS bp_lowering_2h\n    FROM target t\n    LEFT JOIN read_csv('{GENERAL_CSV}', header=true, auto_detect=true, ignore_errors=true) g ON g.patientid::INTEGER=t.patientid\n    LEFT JOIN residual r ON r.target_id=t.target_id\n    LEFT JOIN cointervention c ON c.target_id=t.target_id\n    ")
con.execute('\n    CREATE TEMP TABLE timeline_ordered AS\n    SELECT t.*, LAG(t.target_id) OVER (PARTITION BY t.patientid ORDER BY t.givenat,t.target_id) AS previous_id_ordered\n    FROM timeline t\n    ')
con.execute(f"\n    CREATE TEMP TABLE candidate_edges AS\n    SELECT cur.target_id AS current_id, prev.target_id AS previous_id,\n           cur.patientid, cur.givenat AS current_at, prev.givenat AS previous_at,\n           cur.pharmaid AS current_pharmaid, cur.drug AS current_drug,\n           prev.pharmaid AS previous_pharmaid, prev.drug AS previous_drug,\n           EPOCH(cur.givenat-prev.givenat)/60.0 AS prev_gap_minutes,\n           cur.admissiontime,\n           cur.other_prev4h AS current_other_prev4h, cur.other_next2h AS current_other_next2h,\n           prev.other_prev4h AS previous_other_prev4h, prev.other_next2h AS previous_other_next2h,\n           cur.sedative_2h AS current_sedative, prev.sedative_2h AS previous_sedative,\n           cur.bp_lowering_2h AS current_bp_lowering, prev.bp_lowering_2h AS previous_bp_lowering,\n           cur.pressor_active AS current_pressor_active,\n           cur.pressor_pre30_change AS current_pre30_change,\n           prev.pressor_pre30_change AS previous_pre30_change,\n           cur.next_at AS current_next_at\n    FROM timeline_ordered cur JOIN timeline_ordered prev ON prev.target_id=cur.previous_id_ordered\n    WHERE cur.drug<>prev.drug\n      AND cur.pressor_active=0\n      AND {('prev.prev_at IS NOT NULL AND EPOCH(prev.givenat-prev.prev_at)/60.0>=120 AND cur.next_at IS NOT NULL AND EPOCH(cur.next_at-cur.givenat)/60.0>=120 AND EPOCH(cur.givenat-prev.givenat)/60.0>=120 AND cur.pressor_pre30_change=0 AND prev.pressor_pre30_change=0' if MODE == 'primary' else 'prev.prev_at IS NOT NULL AND EPOCH(prev.givenat-prev.prev_at)/60.0>=180 AND cur.next_at IS NOT NULL AND EPOCH(cur.next_at-cur.givenat)/60.0>=180 AND EPOCH(cur.givenat-prev.givenat)/60.0>=180 AND cur.pressor_pre30_change=0 AND prev.pressor_pre30_change=0' if MODE == '180' else 'cur.pressor_change_2h=0 AND prev.pressor_change_2h=0 AND cur.other_prev4h=0 AND cur.other_next2h=0 AND prev.other_prev4h=0 AND prev.other_next2h=0 AND cur.sedative_2h=0 AND prev.sedative_2h=0 AND cur.bp_lowering_2h=0 AND prev.bp_lowering_2h=0')}\n    ")

def rows(sql: str):
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]
candidate_counts = rows("\n    WITH p AS (\n      SELECT patientid,\n             COUNT(*) FILTER (WHERE previous_drug='metamizole' AND current_drug='paracetamol') AS a,\n             COUNT(*) FILTER (WHERE previous_drug='paracetamol' AND current_drug='metamizole') AS b\n      FROM candidate_edges GROUP BY patientid\n    ), e AS (SELECT COUNT(*) AS transition_edges FROM candidate_edges)\n    SELECT COUNT(*) FILTER (WHERE a>0 AND b>0) AS bidirectional_patients,\n           COUNT(*) AS any_transition_patients,\n           (SELECT transition_edges FROM e) AS transition_edges\n    FROM p\n    ")[0]
con.execute("\n    CREATE TEMP TABLE primary_events AS\n    SELECT e.*\n    FROM candidate_edges e\n    WHERE e.patientid IN (\n      SELECT patientid FROM candidate_edges GROUP BY patientid\n      HAVING COUNT(*) FILTER (WHERE previous_drug='metamizole' AND current_drug='paracetamol')>0\n         AND COUNT(*) FILTER (WHERE previous_drug='paracetamol' AND current_drug='metamizole')>0\n    )\n    ")
con.execute(f"\n    CREATE TEMP TABLE event_covariates AS\n    SELECT e.*,\n           AVG(m.vm5) FILTER (WHERE m.datetime>=e.current_at-INTERVAL 30 MINUTE AND m.datetime<e.current_at) AS pre_map,\n           REGR_SLOPE(m.vm5, EPOCH(m.datetime-e.current_at)) FILTER (WHERE m.datetime>=e.current_at-INTERVAL 30 MINUTE AND m.datetime<e.current_at) AS pre_map_slope_min,\n           AVG(m.vm2) FILTER (WHERE m.datetime>=e.current_at-INTERVAL 30 MINUTE AND m.datetime<e.current_at) AS pre_temp,\n           AVG(m.vm5) FILTER (WHERE m.datetime>=e.current_at+INTERVAL 30 MINUTE AND m.datetime<e.current_at+INTERVAL 120 MINUTE) AS post_map_30_120,\n           AVG(m.vm5) FILTER (WHERE m.datetime>=e.current_at-INTERVAL 60 MINUTE AND m.datetime<e.current_at-INTERVAL 30 MINUTE) AS placebo_pre_60_30,\n           AVG(m.vm5) FILTER (WHERE m.datetime>=e.current_at-INTERVAL 30 MINUTE AND m.datetime<e.current_at) AS placebo_pre_30_0,\n           AVG(m.vm2) FILTER (WHERE m.datetime>=e.current_at+INTERVAL 90 MINUTE AND m.datetime<=e.current_at+INTERVAL 120 MINUTE) AS temp_90_120,\n           ARG_MAX(m.vm2, m.datetime) FILTER (WHERE m.datetime>=e.current_at-INTERVAL 4 HOUR AND m.datetime<e.current_at AND m.vm2 IS NOT NULL) AS pre_temp_4h,\n           COUNT(*) FILTER (WHERE m.datetime>=e.current_at-INTERVAL 30 MINUTE AND m.datetime<e.current_at AND m.vm5 IS NOT NULL) AS pre_map_n,\n           COUNT(*) FILTER (WHERE m.datetime>=e.current_at+INTERVAL 30 MINUTE AND m.datetime<e.current_at+INTERVAL 120 MINUTE AND m.vm5 IS NOT NULL) AS post_map_n\n    FROM primary_events e JOIN read_parquet('{MAP_GLOB}') m ON m.patientid=e.patientid\n      AND m.datetime BETWEEN e.current_at-INTERVAL 4 HOUR AND e.current_at+INTERVAL 120 MINUTE\n    GROUP BY ALL\n    ")
con.execute("\n    CREATE TEMP TABLE event_counts AS\n    SELECT e.current_id,\n           COUNT(*) FILTER (WHERE t.givenat>=e.current_at-INTERVAL 24 HOUR AND t.givenat<e.current_at AND t.drug='metamizole') AS met_24h,\n           COUNT(*) FILTER (WHERE t.givenat>=e.current_at-INTERVAL 24 HOUR AND t.givenat<e.current_at AND t.drug='paracetamol') AS para_24h\n    FROM primary_events e LEFT JOIN target t ON t.patientid=e.patientid\n    GROUP BY e.current_id\n    ")
con.execute(f'\n    CREATE TEMP TABLE event_pressor_outcomes AS\n    SELECT e.current_id,\n           MAX(CASE WHEN p.givenat>e.current_at AND p.givenat<=e.current_at+INTERVAL 120 MINUTE THEN 1 ELSE 0 END) AS any_new_pressor,\n           MAX(CASE WHEN p.pharmaid IN {NE_IDS} AND p.givenat>e.current_at AND p.givenat<=e.current_at+INTERVAL 120 MINUTE THEN 1 ELSE 0 END) AS new_ne\n    FROM primary_events e LEFT JOIN pressor_raw p ON p.patientid=e.patientid\n    GROUP BY e.current_id\n    ')
con.execute(f"\n    CREATE TEMP TABLE low_runs AS\n    WITH pts AS (\n      SELECT e.current_id,\n             CAST(ROUND(EPOCH(m.datetime-e.current_at)/120.0) AS INTEGER) AS bin,\n             m.vm5\n      FROM primary_events e JOIN read_parquet('{MAP_GLOB}') m\n        ON m.patientid=e.patientid AND m.datetime>=e.current_at AND m.datetime<=e.current_at+INTERVAL 120 MINUTE\n      WHERE m.vm5 IS NOT NULL AND m.vm5<65\n    ), groups AS (\n      SELECT current_id, bin,\n             bin-ROW_NUMBER() OVER (PARTITION BY current_id ORDER BY bin) AS grp\n      FROM pts\n    ), runs AS (\n      SELECT current_id, grp, COUNT(*) AS run_len FROM groups GROUP BY current_id, grp\n    )\n    SELECT current_id, MAX(run_len) AS max_low_run_bins FROM runs GROUP BY current_id\n    ")
events = con.execute('\n    SELECT c.*, (c.post_map_30_120-c.pre_map) AS map_change,\n           (c.placebo_pre_30_0-c.placebo_pre_60_30) AS placebo_change,\n           (c.temp_90_120-c.pre_temp) AS temp_change,\n           CASE WHEN COALESCE(l.max_low_run_bins,0)>=5 THEN 1 ELSE 0 END AS sustained_map_below65,\n           COALESCE(po.any_new_pressor,0) AS any_new_pressor,\n           COALESCE(po.new_ne,0) AS new_ne,\n           COALESCE(ec.met_24h,0) AS met_24h, COALESCE(ec.para_24h,0) AS para_24h,\n           (EPOCH(c.current_at-c.admissiontime)/3600.0) AS icu_hours\n    FROM event_covariates c\n    LEFT JOIN low_runs l ON l.current_id=c.current_id\n    LEFT JOIN event_pressor_outcomes po ON po.current_id=c.current_id\n    LEFT JOIN event_counts ec ON ec.current_id=c.current_id\n    ').df()

def ci(coef: float, se: float) -> dict:
    return {'estimate': float(coef), 'se': float(se), 'ci95_low': float(coef - 1.96 * se), 'ci95_high': float(coef + 1.96 * se)}

def equivalence_judgement(model: dict, margin: float=3.0) -> dict:
    treatment = model.get('treatment', {}) if isinstance(model, dict) else {}
    low, high = (treatment.get('ci95_low'), treatment.get('ci95_high'))
    if model.get('status') != 'ok' or low is None or high is None:
        return {'status': 'not_judged', 'margin_mmHg': margin}
    if low >= -margin and high <= margin:
        status = 'clinically_equivalent_ci_within_margin'
    elif high < -margin or low > margin:
        status = 'clinically_different_ci_excludes_margin'
    else:
        status = 'inconclusive_ci_crosses_margin'
    return {'status': status, 'margin_mmHg': margin, 'ci95_low': float(low), 'ci95_high': float(high)}
BASE_X = ['treatment', 'previous_treatment', 'prev_gap_hours', 'met_24h', 'para_24h', 'pre_map', 'pre_map_slope_hour', 'pre_temp', 'icu_hours']

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x['treatment'] = (x.current_drug == 'metamizole').astype(float)
    x['previous_treatment'] = (x.previous_drug == 'metamizole').astype(float)
    x['prev_gap_hours'] = x.prev_gap_minutes / 60.0
    x['pre_map_slope_hour'] = x.pre_map_slope_min * 60.0
    x['transition_direction'] = x.previous_treatment
    x['treatment_x_direction'] = x.treatment * x.transition_direction
    return x

def fixed_effects_ols(df: pd.DataFrame, outcome: str, interaction: bool=False) -> dict:
    x = prepare(df)
    cols = BASE_X + (['treatment_x_direction'] if interaction else [])
    keep = ['patientid', outcome] + cols
    x = x[keep].replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty or x.patientid.nunique() < 2:
        return {'status': 'insufficient', 'n': int(len(x)), 'patients': int(x.patientid.nunique())}
    if interaction and x['treatment_x_direction'].nunique() <= 1:
        return {'status': 'not_identifiable', 'n': int(len(x)), 'patients': int(x.patientid.nunique()), 'reason': 'strict alternating treatment makes treatment x previous-drug interaction constant'}
    demean_cols = [outcome] + cols
    g = x.groupby('patientid', sort=False)
    xd = x.copy()
    means = g[demean_cols].transform('mean')
    xd[demean_cols] = xd[demean_cols] - means
    fit = sm.OLS(xd[outcome], xd[cols]).fit(cov_type='cluster', cov_kwds={'groups': xd.patientid})
    out = {'status': 'ok', 'n': int(len(x)), 'patients': int(x.patientid.nunique()), 'outcome': outcome, 'covariates': cols}
    for name in ['treatment'] + (['treatment_x_direction'] if interaction else []):
        out[name] = ci(fit.params[name], fit.bse[name])
    return out

def conditional_logit(df: pd.DataFrame, outcome: str) -> dict:
    x = prepare(df)
    cols = ['treatment', 'previous_treatment', 'prev_gap_hours', 'met_24h', 'para_24h', 'pre_map', 'pre_map_slope_hour', 'pre_temp', 'icu_hours']
    x = x[['patientid', outcome] + cols].replace([np.inf, -np.inf], np.nan).dropna()
    outcome_events = int(x[outcome].sum())
    varying = x.groupby('patientid')[outcome].nunique()
    x = x[x.patientid.isin(varying[varying > 1].index)]
    if len(x) < 20 or x.patientid.nunique() < 5:
        return {'status': 'insufficient', 'n': int(len(x)), 'patients': int(x.patientid.nunique()), 'outcome': outcome, 'outcome_events_before_stratification': outcome_events}
    try:
        scale_cols = ['prev_gap_hours', 'met_24h', 'para_24h', 'pre_map', 'pre_map_slope_hour', 'pre_temp', 'icu_hours']
        for col in scale_cols:
            sd = x[col].std()
            if sd and np.isfinite(sd):
                x[col] = (x[col] - x[col].mean()) / sd
        model = ConditionalLogit(x[outcome].astype(int), x[cols], groups=x.patientid)
        fit = model.fit(disp=False, maxiter=200)
        coef = fit.params['treatment']
        if fit.bse is None or not np.isfinite(fit.bse['treatment']):
            return {'status': 'not_estimable', 'n': int(len(x)), 'patients': int(x.patientid.nunique()), 'outcome': outcome, 'outcome_events_before_stratification': outcome_events, 'log_coef': float(coef), 'reason': 'conditional logistic fit did not provide a finite covariance estimate'}
        se = fit.bse['treatment']
        log_low = float(coef - 1.96 * se)
        log_high = float(coef + 1.96 * se)
        if not np.isfinite(log_low) or not np.isfinite(log_high) or abs(se) > 10 or (max(abs(log_low), abs(log_high)) > 20):
            return {'status': 'not_estimable', 'n': int(len(x)), 'patients': int(x.patientid.nunique()), 'outcome': outcome, 'outcome_events_before_stratification': outcome_events, 'log_coef': float(coef), 'se': float(se), 'reason': 'conditional logistic covariance produced a non-reportable confidence interval'}
        return {'status': 'ok', 'n': int(len(x)), 'patients': int(x.patientid.nunique()), 'outcome': outcome, 'odds_ratio': float(np.exp(coef)), 'ci95_low': float(np.exp(log_low)), 'ci95_high': float(np.exp(log_high)), 'log_coef': float(coef), 'se': float(se)}
    except Exception as exc:
        message = repr(exc)
        status = 'not_estimable' if 'covariance' in message.lower() or 'hessian' in message.lower() else 'error'
        return {'status': status, 'n': int(len(x)), 'patients': int(x.patientid.nunique()), 'outcome': outcome, 'outcome_events_before_stratification': outcome_events, 'error': message}

def cohort_summary(df: pd.DataFrame) -> dict:
    return {'n': int(len(df)), 'patients': int(df.patientid.nunique()), 'metamizole_events': int((df.current_drug == 'metamizole').sum()), 'paracetamol_events': int((df.current_drug == 'paracetamol').sum()), 'metamizole_mean_map_change': float(df.loc[df.current_drug == 'metamizole', 'map_change'].mean()), 'paracetamol_mean_map_change': float(df.loc[df.current_drug == 'paracetamol', 'map_change'].mean())}

def binary_pair_counts(df: pd.DataFrame, outcome: str) -> dict:
    x = prepare(df)
    cols = ['patientid', 'current_drug', outcome] + BASE_X
    x = x[cols].replace([np.inf, -np.inf], np.nan).dropna()
    by_drug = x.groupby('current_drug')[outcome].agg(n='size', events='sum').to_dict('index')
    per_patient_drug = x.groupby(['patientid', 'current_drug'], sort=False)[outcome].agg(n='size', events='sum').reset_index()
    both = per_patient_drug.groupby('patientid')['current_drug'].nunique()
    both_ids = both[both == 2].index
    paired = per_patient_drug[per_patient_drug.patientid.isin(both_ids)]
    any_event = paired.assign(event=(paired.events > 0).astype(int)).pivot(index='patientid', columns='current_drug', values='event')
    any_event = any_event.dropna()
    discordant = int((any_event['metamizole'] != any_event['paracetamol']).sum()) if not any_event.empty else 0
    return {'outcome': outcome, 'complete_covariate_rows': int(len(x)), 'complete_covariate_patients': int(x.patientid.nunique()), 'by_drug': by_drug, 'patients_with_both_drugs': int(len(both_ids)), 'discordant_patients_any_event_by_drug': discordant}

def complete_case_audit(df: pd.DataFrame) -> dict:
    fields = ['map_change', 'previous_drug', 'prev_gap_minutes', 'met_24h', 'para_24h', 'pre_map', 'pre_map_slope_min', 'pre_temp', 'icu_hours']
    start = int(len(df))
    current = df
    sequential = []
    for field in fields:
        present = current[field].notna()
        if pd.api.types.is_numeric_dtype(current[field]):
            present = present & np.isfinite(current[field])
        current = current[present]
        sequential.append({'field_added': field, 'n': int(len(current)), 'patients': int(current.patientid.nunique())})
    missing = {field: {'n_missing': int(df[field].isna().sum()), 'n_present': int(df[field].notna().sum())} for field in fields}
    return {'starting_event_rows': start, 'starting_patients': int(df.patientid.nunique()), 'fields': fields, 'missing_by_field': missing, 'sequential_complete_case': sequential, 'complete_case_n': int(len(current)), 'complete_case_patients': int(current.patientid.nunique())}
analysis = {'mode': MODE, 'primary_cohort': cohort_summary(events), 'models': {}, 'sensitivity': {}}
analysis['complete_case_audit'] = complete_case_audit(events)
analysis['models']['primary_map'] = fixed_effects_ols(events, 'map_change')
analysis['models']['primary_map']['equivalence'] = equivalence_judgement(analysis['models']['primary_map'])
analysis['models']['temperature_effect'] = fixed_effects_ols(events, 'temp_change')
analysis['temperature_descriptive'] = {'metamizole': {'n': int(events.loc[events.current_drug == 'metamizole', 'temp_change'].notna().sum()), 'mean_change': float(events.loc[events.current_drug == 'metamizole', 'temp_change'].mean())}, 'paracetamol': {'n': int(events.loc[events.current_drug == 'paracetamol', 'temp_change'].notna().sum()), 'mean_change': float(events.loc[events.current_drug == 'paracetamol', 'temp_change'].mean())}}
temp4 = events.copy()
temp4['pre_temp'] = temp4['pre_temp_4h']
analysis['sensitivity']['recent_temperature_4h_posthoc'] = fixed_effects_ols(temp4, 'map_change')
analysis['sensitivity']['recent_temperature_4h_posthoc']['estimand'] = 'primary MAP model with pre-dose temperature replaced by most recent value within 4h; post hoc, not prespecified'
analysis['sensitivity']['recent_temperature_4h_posthoc']['equivalence'] = equivalence_judgement(analysis['sensitivity']['recent_temperature_4h_posthoc'])
analysis['models']['primary_placebo'] = fixed_effects_ols(events, 'placebo_change')
analysis['models']['order_interaction'] = fixed_effects_ols(events, 'map_change', interaction=True)
analysis['models']['map_below65'] = conditional_logit(events, 'sustained_map_below65')
analysis['models']['new_ne_or_pressor'] = conditional_logit(events, 'any_new_pressor')
analysis['binary_pair_counts'] = {'sustained_map_below65': binary_pair_counts(events, 'sustained_map_below65'), 'any_new_pressor': binary_pair_counts(events, 'any_new_pressor')}
analysis['sensitivity']['washout_180'] = {'status': 'reported_as_separate_mode', 'run_mode': '180'}
analysis['sensitivity']['fever_ge_38_3'] = fixed_effects_ols(events[events.pre_temp >= 38.3], 'map_change')
analysis['sensitivity']['fever_lt_38_3'] = fixed_effects_ols(events[events.pre_temp < 38.3], 'map_change')
analysis['sensitivity']['fever_ge_38_3']['equivalence'] = equivalence_judgement(analysis['sensitivity']['fever_ge_38_3'])
analysis['sensitivity']['fever_lt_38_3']['equivalence'] = equivalence_judgement(analysis['sensitivity']['fever_lt_38_3'])
analysis['sensitivity']['old_exclusion'] = {'status': 'reported_as_separate_mode', 'run_mode': 'old'}
analysis['paracetamol_pharmaids'] = {}
for pid in PARA_IDS:
    sub = events[(events.current_pharmaid == MET_ID) | (events.current_pharmaid == pid)]
    analysis['paracetamol_pharmaids'][str(pid)] = {'routes_in_source': rows(f'SELECT route, COUNT(*) AS n FROM target WHERE pharmaid={pid} GROUP BY route ORDER BY route'), 'cohort': cohort_summary(sub), 'map_model': fixed_effects_ols(sub, 'map_change')}
analysis['definitions'] = {'primary': 'no pressor at current administration; adjacent target drug switch; >=120m to previous and next target administration; no reconstructed pressor dose change in prior 30m at either endpoint; bidirectional patient sequences', 'main_estimand': 'within-patient metamizole minus paracetamol difference in MAP change, post 30-120m minus pre -30-0m', 'sealed': 'pressor-active layer excluded from HiRID outcome analysis pending SICdb', 'caveat': 'This output includes formal primary and prespecified falsification results; sensitivity cohorts requiring a separate rebuild are explicitly marked not run rather than silently substituted.', 'variable_mapping': 'official benchmark metavariable mapping: vm5 aggregates invasive MAP variableid 110; vm2 aggregates temperature variables including core temperature variableid 410'}
result = {'dataset': 'HiRID 1.1.1', 'mode': MODE, 'outcome_blind_before_freeze': False, 'event_rows': events.to_dict(orient='records'), 'analysis': analysis}
(OUT / ('hirid_metamizole_crossover_analysis.json' if MODE == 'primary' else f'hirid_metamizole_crossover_analysis_{MODE}.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
print(json.dumps({'cohort': analysis['primary_cohort'], 'models': analysis['models'], 'sensitivity': analysis['sensitivity']}, indent=2, ensure_ascii=True, default=str))

