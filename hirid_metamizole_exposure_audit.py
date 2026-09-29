from __future__ import annotations
import json
import csv
import math
from pathlib import Path
from config import DUCKDB_TEMP, HIRID_MAP_GLOB, HIRID_ROOT, RESULTS_DIR, SICDB_ROOT
from statistics import NormalDist
import duckdb
ROOT = HIRID_ROOT
OUT = RESULTS_DIR
OUT.mkdir(parents=True, exist_ok=True)
PHARMA_GLOB = str(ROOT / 'raw_stage' / 'pharma_records' / 'csv' / 'part-*.csv').replace('\\', '/')
MAP_GLOB = HIRID_MAP_GLOB
CSV_COLUMNS = "{patientid: 'INTEGER', pharmaid: 'INTEGER', givenat: 'TIMESTAMP', enteredentryat: 'TIMESTAMP', givendose: 'DOUBLE', cumulativedose: 'DOUBLE', fluidamount_calc: 'DOUBLE', cumulfluidamount_calc: 'DOUBLE', doseunit: 'VARCHAR', route: 'VARCHAR', infusionid: 'VARCHAR', typeid: 'INTEGER', subtypeid: 'DOUBLE', recordstatus: 'INTEGER'}"
MET = (1000605,)
PARA = (1000489, 1000490)
TARGET = MET + PARA
PRESSORS = (1000462, 1000656, 1000657, 1000658, 71, 1000750, 1000649, 1000650, 1000655, 112, 113)
SEDATIVE_BOLUS = (1000700, 1001054, 251, 1001052, 1000691, 1000239, 245, 390, 442, 1000400, 1000857)
BP_LOWERING = (1000763, 1001047, 1000252, 1000695, 1000867, 1000243, 1000868, 1001048, 1001141, 1001142, 117, 1000596, 1000436, 94, 242, 1000477)
FLUID_IDS = (1000546, 1000090)
con = duckdb.connect()
con.execute("SET memory_limit='768MB'")
con.execute('SET threads=2')
con.execute('SET preserve_insertion_order=false')
con.execute(f"SET temp_directory='{DUCKDB_TEMP.as_posix()}'")
con.execute(f"\n    CREATE TEMP TABLE target AS\n    SELECT ROW_NUMBER() OVER () AS target_id,\n           patientid::INTEGER AS patientid,\n           pharmaid::INTEGER AS pharmaid,\n           givenat::TIMESTAMP AS givenat,\n           route,\n           givendose::DOUBLE AS givendose,\n           infusionid::VARCHAR AS infusionid\n    FROM read_csv('{PHARMA_GLOB}', union_by_name=true, header=true, columns={CSV_COLUMNS}, ignore_errors=true)\n    WHERE pharmaid IN {TARGET}\n      AND givenat IS NOT NULL\n      AND route IN ('iv-inj', 'iv-inf', 'cv-inj', 'cv-inf')\n      AND givendose > 0\n      AND (recordstatus & 2) = 0 AND (recordstatus & 32) = 0\n    ")
con.execute('CREATE TEMP TABLE paired AS SELECT patientid FROM target GROUP BY patientid HAVING BOOL_OR(pharmaid IN (1000605)) AND BOOL_OR(pharmaid IN (1000489,1000490))')
con.execute(f"\n    CREATE TEMP TABLE pressor_raw AS\n    SELECT patientid::INTEGER AS patientid, pharmaid::INTEGER AS pharmaid,\n           givenat::TIMESTAMP AS givenat, givendose::DOUBLE AS givendose,\n           infusionid::VARCHAR AS infusionid\n    FROM read_csv('{PHARMA_GLOB}', union_by_name=true, header=true, columns={CSV_COLUMNS}, ignore_errors=true)\n    WHERE pharmaid IN {PRESSORS}\n      AND patientid IN (SELECT patientid FROM paired)\n      AND givenat IS NOT NULL AND givendose > 0\n      AND route IN ('cv-inf', 'iv-inf')\n      AND (recordstatus & 2) = 0 AND (recordstatus & 32) = 0\n    ")
con.execute('\n    CREATE TEMP TABLE pressor_intervals AS\n    WITH x AS (\n      SELECT *, LEAD(givenat) OVER (PARTITION BY patientid, infusionid ORDER BY givenat) AS next_at,\n                   LAG(givendose) OVER (PARTITION BY patientid, infusionid ORDER BY givenat) AS prev_dose\n      FROM pressor_raw\n    )\n    SELECT patientid, pharmaid, infusionid, givenat AS start_at,\n           CASE WHEN next_at IS NOT NULL AND next_at <= givenat + INTERVAL 15 MINUTE\n                THEN next_at ELSE givenat + INTERVAL 10 MINUTE END AS stop_at,\n           givendose, prev_dose,\n           CASE WHEN prev_dose IS NOT NULL AND givendose <> prev_dose THEN 1 ELSE 0 END AS dose_change\n    FROM x\n    ')
con.execute('\n    CREATE TEMP TABLE pressor_state AS\n    SELECT t.target_id,\n           COUNT(DISTINCT p.infusionid) AS active_infusion_count,\n           COUNT(DISTINCT p.pharmaid) AS active_pressor_product_count,\n           MAX(p.dose_change) FILTER (WHERE p.start_at BETWEEN t.givenat - INTERVAL 2 HOUR AND t.givenat + INTERVAL 2 HOUR) AS pressor_change_2h,\n           SUM(p.givendose) AS raw_active_dose_sum\n    FROM target t LEFT JOIN pressor_intervals p\n      ON p.patientid=t.patientid AND p.start_at <= t.givenat AND t.givenat < p.stop_at\n    GROUP BY t.target_id\n    ')
con.execute('\n    CREATE TEMP TABLE pressor_change_window AS\n    SELECT t.target_id,\n           MAX(CASE WHEN p.dose_change=1 THEN 1 ELSE 0 END) AS pressor_change_2h\n    FROM target t LEFT JOIN pressor_intervals p\n      ON p.patientid=t.patientid\n     AND p.start_at BETWEEN t.givenat - INTERVAL 2 HOUR AND t.givenat + INTERVAL 2 HOUR\n    GROUP BY t.target_id\n    ')
con.execute('\n    CREATE TEMP TABLE residual AS\n    SELECT t.target_id,\n           MAX(CASE WHEN o.givenat < t.givenat THEN 1 ELSE 0 END) AS other_drug_prev4h,\n           MAX(CASE WHEN o.givenat > t.givenat THEN 1 ELSE 0 END) AS other_drug_next2h\n    FROM target t LEFT JOIN target o\n      ON o.patientid=t.patientid AND o.pharmaid <> t.pharmaid\n     AND ((o.givenat >= t.givenat - INTERVAL 4 HOUR AND o.givenat < t.givenat)\n       OR (o.givenat > t.givenat AND o.givenat <= t.givenat + INTERVAL 2 HOUR))\n    GROUP BY t.target_id\n    ')
con.execute(f"\n    CREATE TEMP TABLE cointervention AS\n    SELECT t.target_id,\n           MAX(CASE WHEN p.pharmaid IN {SEDATIVE_BOLUS} THEN 1 ELSE 0 END) AS sedative_bolus_2h,\n           MAX(CASE WHEN p.pharmaid IN {BP_LOWERING} THEN 1 ELSE 0 END) AS bp_lowering_2h,\n           COUNT(*) AS any_named_intervention_2h\n    FROM target t LEFT JOIN read_csv('{PHARMA_GLOB}', union_by_name=true, header=true, columns={CSV_COLUMNS}, ignore_errors=true) p\n      ON p.patientid::INTEGER=t.patientid\n     AND p.pharmaid::INTEGER NOT IN {TARGET}\n     AND p.givenat::TIMESTAMP BETWEEN t.givenat - INTERVAL 2 HOUR AND t.givenat + INTERVAL 2 HOUR\n     AND p.givendose::DOUBLE > 0\n     AND (p.recordstatus::INTEGER & 2) = 0 AND (p.recordstatus::INTEGER & 32) = 0\n     AND p.route IN ('iv-inj', 'iv-inf', 'cv-inj', 'cv-inf')\n    GROUP BY t.target_id\n    ")
con.execute('\n    CREATE TEMP TABLE annotated AS\n    SELECT t.*, COALESCE(ps.active_infusion_count,0) AS active_infusion_count,\n           COALESCE(ps.active_pressor_product_count,0) AS active_pressor_product_count,\n           COALESCE(pc.pressor_change_2h,0) AS pressor_change_2h,\n           COALESCE(ps.raw_active_dose_sum,0) AS raw_active_dose_sum,\n           COALESCE(r.other_drug_prev4h,0) AS other_drug_prev4h,\n           COALESCE(r.other_drug_next2h,0) AS other_drug_next2h,\n           COALESCE(c.sedative_bolus_2h,0) AS sedative_bolus_2h,\n           COALESCE(c.bp_lowering_2h,0) AS bp_lowering_2h,\n           COALESCE(c.any_named_intervention_2h,0) AS any_named_intervention_2h,\n           CASE WHEN COALESCE(r.other_drug_prev4h,0)=1 OR COALESCE(r.other_drug_next2h,0)=1 THEN 1 ELSE 0 END AS residual_excluded,\n           CASE WHEN COALESCE(ps.pressor_change_2h,0)=1 OR COALESCE(c.sedative_bolus_2h,0)=1 OR COALESCE(c.bp_lowering_2h,0)=1 THEN 1 ELSE 0 END AS cointervention_excluded\n    FROM target t\n    LEFT JOIN pressor_state ps ON ps.target_id=t.target_id\n    LEFT JOIN pressor_change_window pc ON pc.target_id=t.target_id\n    LEFT JOIN residual r ON r.target_id=t.target_id\n    LEFT JOIN cointervention c ON c.target_id=t.target_id\n    ')
summary = con.execute("\n    SELECT CASE WHEN pharmaid=1000605 THEN 'metamizole' ELSE 'paracetamol' END AS drug,\n           active_infusion_count > 0 AS pressor_active,\n           COUNT(*) AS administrations,\n           COUNT(DISTINCT patientid) AS patients,\n           COUNT(*) FILTER (WHERE residual_excluded=0 AND cointervention_excluded=0) AS retained_primary,\n           COUNT(DISTINCT patientid) FILTER (WHERE residual_excluded=0 AND cointervention_excluded=0) AS retained_primary_patients,\n           COUNT(*) FILTER (WHERE residual_excluded=1) AS residual_excluded_n,\n           COUNT(*) FILTER (WHERE cointervention_excluded=1) AS cointervention_excluded_n,\n           COUNT(*) FILTER (WHERE active_infusion_count>0 AND pressor_change_2h=1) AS pressor_changed_n\n    FROM annotated\n    GROUP BY ALL ORDER BY drug, pressor_active\n    ").fetchall()
cols = [d[0] for d in con.description]
result = {'dataset': 'HiRID 1.1.1', 'interval_definition': 'pressor infusion active when a valid cv-inf/iv-inf record covers givenat; consecutive records joined if gap <=15m, terminal record held for 10m', 'residual_definition': 'other target antipyretic in prior 4h or subsequent 2h', 'cointervention_definition': 'named sedative or BP-lowering IV/cv record within +/-2h, or pressor dose-change marker within +/-2h', 'fluid_bolus': 'not counted as a reliable Pharma exposure: HiRID reference exposes saline/colloid as cumulative observation variables rather than a validated bolus event; this remains an unmeasured cointervention for the primary exclusion', 'pressor_dose': 'raw givendose sum retained as a per-drug covariate; no cross-drug dose equivalence assumed', 'summary': [dict(zip(cols, row)) for row in summary]}
effective_pairs = con.execute('\n    WITH patient_status AS (\n      SELECT patientid,\n             active_infusion_count > 0 AS pressor_active,\n             COUNT(*) FILTER (WHERE pharmaid=1000605) AS met_events,\n             COUNT(*) FILTER (WHERE pharmaid IN (1000489,1000490)) AS para_events\n      FROM annotated\n      WHERE residual_excluded=0 AND cointervention_excluded=0\n      GROUP BY patientid, pressor_active\n    )\n    SELECT pressor_active,\n           COUNT(*) FILTER (WHERE met_events>0 AND para_events>0) AS paired_patients,\n           SUM(met_events) FILTER (WHERE met_events>0 AND para_events>0) AS retained_metamizole_events,\n           SUM(para_events) FILTER (WHERE met_events>0 AND para_events>0) AS retained_paracetamol_events,\n           COUNT(*) FILTER (WHERE met_events>0) AS patients_with_metamizole,\n           COUNT(*) FILTER (WHERE para_events>0) AS patients_with_paracetamol\n    FROM patient_status\n    GROUP BY pressor_active\n    ORDER BY pressor_active\n    ').fetchall()
effective_pair_cols = [d[0] for d in con.description]
effective_pairs = [dict(zip(effective_pair_cols, row)) for row in effective_pairs]

def conservative_binary_power(n: int, p0: float=0.1, delta: float=0.05, alpha: float=0.05) -> float:
    if not n:
        return 0.0
    p1 = p0 + delta
    pbar = (p0 + p1) / 2.0
    se = math.sqrt(2.0 * pbar * (1.0 - pbar) / n)
    mu = delta / se
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    return NormalDist().cdf(-z - mu) + (1.0 - NormalDist().cdf(z - mu))
for row in effective_pairs:
    n = int(row['paired_patients'] or 0)
    row['power_proxy_5pp'] = conservative_binary_power(n)
    row['power_proxy_assumptions'] = 'two independent binary proportions; p0=10%, p1=15%, alpha=0.05; conservative proxy for paired design'
result['effective_within_patient_sample'] = effective_pairs
overall_pair = con.execute('\n    SELECT COUNT(*) AS paired_patients,\n           SUM(met_events) AS retained_metamizole_events,\n           SUM(para_events) AS retained_paracetamol_events\n    FROM (\n      SELECT patientid,\n             COUNT(*) FILTER (WHERE pharmaid=1000605) AS met_events,\n             COUNT(*) FILTER (WHERE pharmaid IN (1000489,1000490)) AS para_events\n      FROM annotated\n      WHERE residual_excluded=0 AND cointervention_excluded=0\n      GROUP BY patientid\n    )\n    WHERE met_events>0 AND para_events>0\n    ').fetchone()
result['effective_within_patient_sample_overall'] = {'paired_patients': int(overall_pair[0] or 0), 'retained_metamizole_events': int(overall_pair[1] or 0), 'retained_paracetamol_events': int(overall_pair[2] or 0), 'note': 'overall pairing allows the two retained drugs to occur in different pressor states; the state-stratified table above requires the same state'}
reason_rows = con.execute("\n    SELECT CASE WHEN pharmaid=1000605 THEN 'metamizole' ELSE 'paracetamol' END AS drug,\n           COUNT(*) AS administrations,\n           COUNT(*) FILTER (WHERE other_drug_prev4h=1) AS other_drug_prev4h_n,\n           COUNT(*) FILTER (WHERE other_drug_next2h=1) AS other_drug_next2h_n,\n           COUNT(*) FILTER (WHERE sedative_bolus_2h=1) AS sedative_bolus_2h_n,\n           COUNT(*) FILTER (WHERE bp_lowering_2h=1) AS bp_lowering_2h_n,\n           COUNT(*) FILTER (WHERE pressor_change_2h=1) AS pressor_change_2h_n,\n           COUNT(*) FILTER (WHERE residual_excluded=1) AS residual_union_n,\n           COUNT(*) FILTER (WHERE cointervention_excluded=1) AS cointervention_union_n,\n           COUNT(*) FILTER (WHERE residual_excluded=1 OR cointervention_excluded=1) AS any_exclusion_n\n    FROM annotated\n    GROUP BY drug\n    ORDER BY drug\n    ").fetchall()
reason_cols = [d[0] for d in con.description]
result['exclusion_reason_distribution'] = [dict(zip(reason_cols, row)) for row in reason_rows]
curve_rows = con.execute(f"\n    SELECT CAST(FLOOR(EPOCH(m.datetime - t.givenat) / 120.0) * 2 AS INTEGER) AS minute,\n           COUNT(*) AS map_observations,\n           COUNT(DISTINCT t.target_id) AS administrations,\n           COUNT(DISTINCT t.patientid) AS patients,\n           AVG(m.vm5) AS mean_map\n    FROM target t\n    JOIN read_parquet('{MAP_GLOB}') m\n      ON m.patientid=t.patientid\n     AND m.datetime BETWEEN t.givenat - INTERVAL 30 MINUTE AND t.givenat + INTERVAL 120 MINUTE\n     AND m.vm5 IS NOT NULL\n    WHERE CAST(FLOOR(EPOCH(m.datetime - t.givenat) / 120.0) * 2 AS INTEGER) BETWEEN -30 AND 118\n    GROUP BY minute\n    ORDER BY minute\n    ").fetchall()
curve_cols = [d[0] for d in con.description]
curve = [dict(zip(curve_cols, row)) for row in curve_rows]
pre_values = [r['mean_map'] for r in curve if r['minute'] < 0 and r['mean_map'] is not None]
pre_mean = sum(pre_values) / len(pre_values) if pre_values else None
for r in curve:
    r['change_vs_pre30_0'] = r['mean_map'] - pre_mean if pre_mean is not None and r['mean_map'] is not None else None
result['givenat_map_curve'] = {'definition': 'all eligible metamizole/paracetamol administrations pooled; MAP vm5; 2-minute bins from -30 to +120 minutes', 'pre30_0_mean_map': pre_mean, 'bins': curve}
(OUT / 'givenat_map_curve.csv').write_text('minute,map_observations,administrations,patients,mean_map,change_vs_pre30_0\n' + '\n'.join((f"{r['minute']},{r['map_observations']},{r['administrations']},{r['patients']},{r['mean_map']},{r['change_vs_pre30_0']}" for r in curve)) + '\n', encoding='utf-8')
try:
    import matplotlib.pyplot as plt
    xs = [r['minute'] for r in curve]
    ys = [r['mean_map'] for r in curve]
    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, ys, color='#1f4e79', linewidth=2)
    plt.axvline(0, color='#b22222', linestyle='--', linewidth=1)
    plt.xlabel('Minutes from givenat')
    plt.ylabel('Mean MAP (mmHg)')
    plt.title('Pooled MAP trajectory around antipyretic administration')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT / 'givenat_map_curve.png', dpi=160)
    plt.close()
    result['givenat_map_curve']['plot'] = str(OUT / 'givenat_map_curve.png')
except Exception as exc:
    result['givenat_map_curve']['plot_error'] = repr(exc)
con.execute(f"\n    CREATE TEMP TABLE fluid_raw AS\n    SELECT patientid::INTEGER AS patientid, pharmaid::INTEGER AS pharmaid,\n           givenat::TIMESTAMP AS givenat, fluidamount_calc::DOUBLE AS fluidamount_calc,\n           givendose::DOUBLE AS givendose\n    FROM read_csv('{PHARMA_GLOB}', union_by_name=true, header=true, columns={CSV_COLUMNS}, ignore_errors=true)\n    WHERE pharmaid IN {FLUID_IDS}\n      AND givenat IS NOT NULL AND route IN ('iv-inj','iv-inf','cv-inj','cv-inf')\n      AND (recordstatus & 2) = 0 AND (recordstatus & 32) = 0\n      AND COALESCE(fluidamount_calc, 0) > 0\n    ")
fluid_counts = con.execute('SELECT pharmaid, COUNT(*) AS records, SUM(fluidamount_calc) AS total_ml, MAX(fluidamount_calc) AS max_record_ml FROM fluid_raw GROUP BY pharmaid ORDER BY pharmaid').fetchall()
fluid_count_cols = [d[0] for d in con.description]
fluid_sensitivity = con.execute("\n    SELECT CASE WHEN t.pharmaid=1000605 THEN 'metamizole' ELSE 'paracetamol' END AS drug,\n           COUNT(*) AS administrations,\n           COUNT(*) FILTER (WHERE COALESCE(x.fluid_ml_60m,0) >= 250) AS fluid_bolus_excluded_n,\n           COUNT(*) FILTER (WHERE COALESCE(x.fluid_ml_60m,0) < 250) AS retained_under_fluid_rule_n\n    FROM target t\n    LEFT JOIN (\n      SELECT t2.target_id, SUM(f.fluidamount_calc) AS fluid_ml_60m\n      FROM target t2 JOIN fluid_raw f\n        ON f.patientid=t2.patientid\n       AND f.givenat BETWEEN t2.givenat - INTERVAL 30 MINUTE AND t2.givenat + INTERVAL 30 MINUTE\n      GROUP BY t2.target_id\n    ) x ON x.target_id=t.target_id\n    GROUP BY drug ORDER BY drug\n    ").fetchall()
fluid_sens_cols = [d[0] for d in con.description]
result['fluid_bolus_sensitivity'] = {'definition': 'reference-matched Pharma fluid amount >=250 mL within +/-30 minutes of givenat', 'fluid_ids': list(FLUID_IDS), 'reference_limitation': 'hirid_variable_reference.csv contains NaCl 2.5% and NaCl concentrated ampoule, but no dedicated isotonic crystalloid or colloid solution IDs; saline/colloid observation variables were not treated as Pharma boluses', 'fluid_records': [dict(zip(fluid_count_cols, row)) for row in fluid_counts], 'by_target_drug': [dict(zip(fluid_sens_cols, row)) for row in fluid_sensitivity]}
(OUT / 'hirid_metamizole_exposure_audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(result, ensure_ascii=True, indent=2))
