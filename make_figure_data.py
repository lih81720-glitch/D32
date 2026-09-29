from __future__ import annotations
import bisect
import csv
import gzip
import json
import struct
import sys
from pathlib import Path
from config import DUCKDB_TEMP, HIRID_MAP_GLOB, HIRID_ROOT, RESULTS_DIR, SICDB_ROOT
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
RES = RESULTS_DIR
OUT = RES / 'figure_data'
OUT.mkdir(parents=True, exist_ok=True)
HIRID_ROOT = HIRID_ROOT
PHARMA_GLOB = str(HIRID_ROOT / 'raw_stage' / 'pharma_records' / 'csv' / 'part-*.csv').replace('\\', '/')
MAP_GLOB = HIRID_MAP_GLOB
CSV_COLUMNS = "{patientid: 'INTEGER', pharmaid: 'INTEGER', givenat: 'TIMESTAMP', enteredentryat: 'TIMESTAMP', givendose: 'DOUBLE', cumulativedose: 'DOUBLE', fluidamount_calc: 'DOUBLE', cumulfluidamount_calc: 'DOUBLE', doseunit: 'VARCHAR', route: 'VARCHAR', infusionid: 'VARCHAR', typeid: 'INTEGER', subtypeid: 'DOUBLE', recordstatus: 'INTEGER'}"
TARGET_IDS = (1000605, 1000489, 1000490)
PRESSOR_IDS = (1000462, 1000656, 1000657, 1000658, 71, 1000750, 1000649, 1000650, 1000655, 112, 113)
SIC_ROOT = SICDB_ROOT
FLOAT_H = SIC_ROOT / 'data_float_h.csv.gz'

def event_rows(path: Path) -> pd.DataFrame:
    d = json.loads(path.read_text(encoding='utf-8'))
    return pd.DataFrame(d.get('event_rows') or d.get('analysis', {}).get('event_rows', []))

def hirid() -> None:
    import duckdb
    con = duckdb.connect()
    con.execute("SET memory_limit='2500MB'")
    con.execute('SET threads=4')
    con.execute('SET preserve_insertion_order=false')
    con.execute(f"SET temp_directory='{DUCKDB_TEMP.as_posix()}'")
    con.execute(f"\n        CREATE TEMP TABLE eligible AS\n        SELECT patientid::INTEGER AS patientid, pharmaid::INTEGER AS pharmaid, givenat::TIMESTAMP AS givenat\n        FROM read_csv('{PHARMA_GLOB}', union_by_name=true, header=true, columns={CSV_COLUMNS}, ignore_errors=true)\n        WHERE pharmaid IN {TARGET_IDS} AND givenat IS NOT NULL\n          AND route IN ('iv-inj','iv-inf','cv-inj','cv-inf') AND givendose > 0\n          AND (recordstatus & 2) = 0 AND (recordstatus & 32) = 0\n    ")
    con.execute('\n        CREATE TEMP TABLE both_patient AS\n        SELECT patientid FROM eligible GROUP BY patientid\n        HAVING BOOL_OR(pharmaid = 1000605) AND BOOL_OR(pharmaid IN (1000489, 1000490))\n    ')
    con.execute('CREATE TEMP TABLE elig_b AS SELECT e.rowid AS rid, e.* FROM eligible e JOIN both_patient USING (patientid)')
    con.execute(f"\n        CREATE TEMP TABLE real_summary AS\n        SELECT e.rid, e.patientid, e.pharmaid, e.givenat,\n               AVG(m.vm5) FILTER (WHERE m.datetime >= e.givenat - INTERVAL 30 MINUTE AND m.datetime < e.givenat) AS pre_mean,\n               REGR_SLOPE(m.vm5, EPOCH(m.datetime - e.givenat)) FILTER (WHERE m.datetime >= e.givenat - INTERVAL 30 MINUTE AND m.datetime < e.givenat) AS pre_slope,\n               AVG(m.vm2) FILTER (WHERE m.datetime >= e.givenat - INTERVAL 30 MINUTE AND m.datetime < e.givenat) AS pre_temp,\n               COUNT(*) FILTER (WHERE m.datetime >= e.givenat - INTERVAL 30 MINUTE AND m.datetime < e.givenat) AS pre_n,\n               COUNT(*) FILTER (WHERE m.datetime >= e.givenat AND m.datetime <= e.givenat + INTERVAL 120 MINUTE) AS post_n\n        FROM elig_b e JOIN read_parquet('{MAP_GLOB}') m\n          ON m.patientid = e.patientid AND m.vm5 IS NOT NULL\n         AND m.datetime BETWEEN e.givenat - INTERVAL 30 MINUTE AND e.givenat + INTERVAL 120 MINUTE\n        GROUP BY ALL\n    ")
    con.execute(f"\n        CREATE TEMP TABLE press_int AS\n        WITH r AS (\n          SELECT patientid::INTEGER AS patientid, givenat::TIMESTAMP AS givenat, infusionid::VARCHAR AS infusionid\n          FROM read_csv('{PHARMA_GLOB}', union_by_name=true, header=true, columns={CSV_COLUMNS}, ignore_errors=true)\n          WHERE pharmaid IN {PRESSOR_IDS} AND patientid IN (SELECT patientid FROM both_patient)\n            AND givenat IS NOT NULL AND givendose > 0 AND route IN ('cv-inf','iv-inf')\n            AND (recordstatus & 2) = 0 AND (recordstatus & 32) = 0\n        ), x AS (SELECT *, LEAD(givenat) OVER (PARTITION BY patientid, infusionid ORDER BY givenat) AS next_at FROM r)\n        SELECT patientid, givenat AS start_at,\n               CASE WHEN next_at IS NOT NULL AND next_at <= givenat + INTERVAL 15 MINUTE THEN next_at ELSE givenat + INTERVAL 10 MINUTE END AS stop_at\n        FROM x\n    ")
    con.execute('\n        CREATE TEMP TABLE real_combined AS\n        SELECT s.*, CASE WHEN EXISTS (SELECT 1 FROM press_int p WHERE p.patientid=s.patientid AND p.start_at <= s.givenat AND s.givenat < p.stop_at) THEN 1 ELSE 0 END AS pressor\n        FROM real_summary s WHERE pre_n >= 3 AND post_n >= 3\n    ')
    con.execute('\n        CREATE TEMP TABLE fake_cand AS\n        SELECT r.rid, r.patientid, r.givenat + off.h * INTERVAL 1 HOUR AS fake_at, r.pressor\n        FROM real_combined r CROSS JOIN (VALUES (-16),(-12),(-8),(8),(12),(16)) off(h)\n        WHERE NOT EXISTS (SELECT 1 FROM eligible x WHERE x.patientid = r.patientid\n                          AND x.givenat BETWEEN r.givenat + off.h * INTERVAL 1 HOUR - INTERVAL 4 HOUR\n                                            AND r.givenat + off.h * INTERVAL 1 HOUR + INTERVAL 4 HOUR)\n    ')
    con.execute(f"\n        CREATE TEMP TABLE fake_summary AS\n        SELECT f.rid, f.patientid, f.fake_at, f.pressor,\n               CASE WHEN EXISTS (SELECT 1 FROM press_int p WHERE p.patientid=f.patientid AND p.start_at <= f.fake_at AND f.fake_at < p.stop_at) THEN 1 ELSE 0 END AS fake_pressor,\n               AVG(m.vm5) FILTER (WHERE m.datetime >= f.fake_at - INTERVAL 30 MINUTE AND m.datetime < f.fake_at) AS pre_mean,\n               REGR_SLOPE(m.vm5, EPOCH(m.datetime - f.fake_at)) FILTER (WHERE m.datetime >= f.fake_at - INTERVAL 30 MINUTE AND m.datetime < f.fake_at) AS pre_slope,\n               AVG(m.vm2) FILTER (WHERE m.datetime >= f.fake_at - INTERVAL 30 MINUTE AND m.datetime < f.fake_at) AS pre_temp,\n               COUNT(*) FILTER (WHERE m.datetime >= f.fake_at - INTERVAL 30 MINUTE AND m.datetime < f.fake_at) AS pre_n,\n               COUNT(*) FILTER (WHERE m.datetime >= f.fake_at AND m.datetime <= f.fake_at + INTERVAL 120 MINUTE) AS post_n\n        FROM fake_cand f JOIN read_parquet('{MAP_GLOB}') m\n          ON m.patientid = f.patientid AND m.vm5 IS NOT NULL\n         AND m.datetime BETWEEN f.fake_at - INTERVAL 30 MINUTE AND f.fake_at + INTERVAL 120 MINUTE\n        GROUP BY ALL\n    ")
    con.execute('\n        CREATE TEMP TABLE fake_best AS\n        SELECT * FROM (\n          SELECT f.*, ROW_NUMBER() OVER (PARTITION BY f.rid ORDER BY\n                 ABS(f.pre_mean - r.pre_mean) / 5.0 + ABS(f.pre_slope - r.pre_slope) * 60.0 / 2.0\n                 + COALESCE(ABS(f.pre_temp - r.pre_temp) / 1.0, 0.0) + CASE WHEN f.fake_pressor <> r.pressor THEN 10.0 ELSE 0.0 END) AS k\n          FROM fake_summary f JOIN real_combined r USING (rid)\n          WHERE f.pre_n >= 3 AND f.post_n >= 3\n        ) WHERE k = 1\n    ')
    sched_vals = '(5),(10),(15),(20),(25),(30),(40),(50),(60),(75),(90),(105),(120),(135),(150),(165),(180)'
    for name, src in [('real', 'SELECT rid, patientid, givenat AS t0 FROM real_combined'), ('sham', 'SELECT rid, patientid, fake_at AS t0 FROM fake_best')]:
        con.execute(f"\n            CREATE TEMP TABLE cantais_{name} AS\n            WITH idx AS ({src}),\n            sched AS (SELECT * FROM idx CROSS JOIN (VALUES {sched_vals}) s(minute)),\n            pts AS (\n              SELECT s.rid, s.minute,\n                     arg_min(m.vm5, ABS(EPOCH(m.datetime - (s.t0 + s.minute * INTERVAL 1 MINUTE)))) AS v\n              FROM sched s JOIN read_parquet('{MAP_GLOB}') m\n                ON m.patientid = s.patientid AND m.vm5 IS NOT NULL\n               AND m.datetime BETWEEN s.t0 + s.minute * INTERVAL 1 MINUTE - INTERVAL 3 MINUTE\n                                  AND s.t0 + s.minute * INTERVAL 1 MINUTE + INTERVAL 3 MINUTE\n              GROUP BY ALL),\n            base AS (\n              SELECT i.rid, i.patientid, arg_max(m.vm5, m.datetime) AS baseline\n              FROM idx i JOIN read_parquet('{MAP_GLOB}') m\n                ON m.patientid = i.patientid AND m.vm5 IS NOT NULL\n               AND m.datetime < i.t0 AND m.datetime >= i.t0 - INTERVAL 60 MINUTE\n              GROUP BY ALL)\n            SELECT b.rid, b.patientid, COUNT(p.v) AS points,\n                   MAX(CASE WHEN p.v <= b.baseline * 0.85 THEN 1 ELSE 0 END) AS event\n            FROM base b LEFT JOIN pts p ON p.rid = b.rid GROUP BY ALL\n        ")
        con.execute(f"COPY (SELECT '{name}' AS grp, * FROM cantais_{name} WHERE points >= 10) TO '{(OUT / f'cantais_{name}.parquet').as_posix()}' (FORMAT parquet)")
    print('real/sham:', con.execute('SELECT (SELECT COUNT(*) FROM real_combined), (SELECT COUNT(*) FROM fake_best)').fetchall(), flush=True)
    parts = []
    for stratum, fname in [('nonpressor', 'hirid_metamizole_crossover_analysis.json'), ('pressor', 'hirid_metamizole_pressor_analysis.json')]:
        x = event_rows(RES / fname)
        drug_col = 'current_drug' if 'current_drug' in x else 'drug'
        at_col = 'current_at' if 'current_at' in x else 'givenat'
        parts.append(pd.DataFrame({'patientid': x.patientid.astype(int), 't0': pd.to_datetime(x[at_col]), 'grp': 'hirid_' + stratum + '_' + x[drug_col].astype(str)}))
    xo = pd.concat(parts, ignore_index=True)
    con.register('xo_df', xo)
    con.execute("\n        CREATE TEMP TABLE idx AS\n        SELECT 'real' AS grp, rid AS src, patientid, givenat AS t0 FROM real_combined\n        UNION ALL SELECT 'sham', rid, patientid, fake_at FROM fake_best\n        UNION ALL SELECT grp, -1, patientid, t0 FROM xo_df\n    ")
    con.execute('CREATE TEMP TABLE idx2 AS SELECT ROW_NUMBER() OVER () AS eid, * FROM idx')
    con.execute(f"\n        COPY (\n          SELECT i.eid, i.grp, i.src, i.patientid,\n                 CAST(FLOOR(EPOCH(m.datetime - i.t0) / 120.0) * 2 AS INTEGER) AS bin,\n                 AVG(m.vm5) AS map, COUNT(*) AS n\n          FROM idx2 i JOIN read_parquet('{MAP_GLOB}') m\n            ON m.patientid = i.patientid AND m.vm5 IS NOT NULL\n           AND m.datetime >= i.t0 - INTERVAL 60 MINUTE AND m.datetime < i.t0 + INTERVAL 120 MINUTE\n          GROUP BY ALL\n        ) TO '{(OUT / 'hirid_bins.parquet').as_posix()}' (FORMAT parquet)\n    ")
    print('hirid bins written', flush=True)

def decode_raw(raw: str, val: str, offset: int):
    if raw and raw.startswith('0x') and (len(raw) == 482):
        try:
            vals = struct.unpack('<60f', bytes.fromhex(raw[2:]))
            return [(offset + 60 * i, v) for i, v in enumerate(vals) if v != 0.0 and np.isfinite(v)]
        except (ValueError, struct.error):
            pass
    try:
        v = float(val)
        return [(offset, v)] if np.isfinite(v) else []
    except (TypeError, ValueError):
        return []

def sicdb() -> None:
    parts = []
    for stratum, fname in [('nonpressor', 'sicdb_metamizole_nopressor_analysis.json'), ('pressor', 'sicdb_metamizole_pressor_analysis.json')]:
        x = event_rows(RES / fname)
        parts.append(pd.DataFrame({'case': x.CaseID.astype(int), 't0': x.Offset.astype(float), 'grp': 'sicdb_' + stratum + '_' + x.drug.astype(str)}))
    ev = pd.concat(parts, ignore_index=True)
    nb = 90
    sums = np.zeros((len(ev), nb))
    cnts = np.zeros((len(ev), nb), dtype=np.int32)
    by_case: dict[int, tuple[list[float], list[int]]] = {}
    for i, (c, t) in enumerate(zip(ev.case, ev.t0)):
        by_case.setdefault(int(c), ([], []))
    for c, g in ev.groupby('case'):
        g = g.sort_values('t0')
        by_case[int(c)] = (g.t0.tolist(), g.index.tolist())
    rows = 0
    with gzip.open(FLOAT_H, 'rt', newline='', encoding='utf-8', errors='replace') as fh:
        rd = csv.reader(fh)
        header = next(rd)
        ic, idd, io, iv, ir = (header.index(k) for k in ('CaseID', 'DataID', 'Offset', 'Val', 'rawdata'))
        for row in rd:
            rows += 1
            if rows % 20000000 == 0:
                print('rows', rows, flush=True)
            if row[idd] != '703':
                continue
            case = int(row[ic])
            hit = by_case.get(case)
            if hit is None:
                continue
            starts, idxs = hit
            base = int(row[io])
            lo = bisect.bisect_left(starts, base - 7200 - 3600)
            hi = bisect.bisect_right(starts, base + 3600 + 3600)
            if lo >= hi:
                continue
            for pt, pv in decode_raw(row[ir], row[iv], base):
                a = bisect.bisect_left(starts, pt - 7200 + 1e-09)
                b = bisect.bisect_right(starts, pt + 3600)
                for k in range(a, b):
                    dt = pt - starts[k]
                    if -3600 <= dt < 7200:
                        j = int((dt + 3600) // 120)
                        e = idxs[k]
                        sums[e, j] += pv
                        cnts[e, j] += 1
    recs = []
    for e in range(len(ev)):
        nz = np.nonzero(cnts[e])[0]
        for j in nz:
            recs.append((e, ev.grp.iat[e], int(ev.case.iat[e]), int(j * 2 - 60), sums[e, j] / cnts[e, j], int(cnts[e, j])))
    pd.DataFrame(recs, columns=['eid', 'grp', 'patientid', 'bin', 'map', 'n']).to_parquet(OUT / 'sicdb_bins.parquet')
    print('sicdb bins written', len(recs), flush=True)
if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('hirid', 'all'):
        hirid()
    if what in ('sicdb', 'all'):
        sicdb()
