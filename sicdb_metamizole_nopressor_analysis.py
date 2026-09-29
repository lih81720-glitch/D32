from __future__ import annotations
import json
import numpy as np
import sicdb_metamizole_pressor_analysis as base

def main() -> None:
    med = base.load_medication()
    cases = __import__('pandas').read_csv(base.CASES, compression='gzip', usecols=['CaseID', 'WeightOnAdmission'])
    edges, audit = base.make_cohort(med, active_value=0)
    edges = base.add_dose_covariates(edges, med, cases)
    edges['treatment'] = (edges.drug == 'metamizole').astype(float)
    edges['previous_treatment'] = (edges.previous_drug == 'metamizole').astype(float)
    edges['prev_gap_hours'] = edges.prev_gap_sec / 3600
    edges['icu_hours'] = edges.Offset / 3600
    target = med[med.DrugID.isin(base.TARGET_IDS)].sort_values(['CaseID', 'Offset'])
    target_by_case = {int(k): v for k, v in target.groupby('CaseID')}

    def prior_count(row, drug_id):
        z = target_by_case.get(int(row.CaseID))
        if z is None:
            return 0
        return int(((z.Offset >= row.Offset - 86400) & (z.Offset < row.Offset) & (z.DrugID == drug_id)).sum())
    edges['met_24h'] = [prior_count(r, base.MET_ID) for r in edges.itertuples()]
    edges['para_24h'] = [prior_count(r, base.PARA_ID) for r in edges.itertuples()]
    edges = base.stream_vitals(edges)
    edges['ne_current'] = edges.ne_current.fillna(0.0)
    edges['ne_30m'] = edges.ne_30m.fillna(0.0)
    edges['ne_change_pre30'] = edges.ne_change_pre30.fillna(0.0)
    edges['epi_active'] = edges.epi_active.fillna(0).astype(int)
    edges['vaso_active'] = edges.vaso_active.fillna(0).astype(int)
    edges['pre10_exclude'] = False
    edges['any_ne_up_or_new_pressor'] = ((edges.post_ne_any_up == 1) | (edges.post_ne_present == 1) | (edges.post_epi_present == 1) | (edges.post_vaso_present == 1)).astype(int)
    for mode in ['primary', 'pre10', 'bidirectional', '180', 'fullcase']:
        result = {'dataset': 'SICdb 1.0.8', 'stratum': 'no_pressor', 'mode': mode, 'audit': audit, 'analysis': base.run_mode(edges, mode)}
        outfile = base.OUT / ('sicdb_metamizole_nopressor_analysis.json' if mode == 'primary' else f'sicdb_metamizole_nopressor_analysis_{mode}.json')
        outfile.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
        print(json.dumps({'mode': mode, 'audit': audit, 'cohort': result['analysis']['cohort'], 'models': result['analysis']['models']}, ensure_ascii=True))
if __name__ == '__main__':
    main()
