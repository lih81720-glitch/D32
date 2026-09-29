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
OUT = RES / 'revision'
OUT.mkdir(parents=True, exist_ok=True)
HIRID_ROOT = HIRID_ROOT
PHARMA_GLOB = str(HIRID_ROOT / 'raw_stage' / 'pharma_records' / 'csv' / 'part-*.csv').replace('\\', '/')
MAP_GLOB = HIRID_MAP_GLOB
CSV_COLUMNS = "{patientid: 'INTEGER', pharmaid: 'INTEGER', givenat: 'TIMESTAMP', enteredentryat: 'TIMESTAMP', givendose: 'DOUBLE', cumulativedose: 'DOUBLE', fluidamount_calc: 'DOUBLE', cumulfluidamount_calc: 'DOUBLE', doseunit: 'VARCHAR', route: 'VARCHAR', infusionid: 'VARCHAR', typeid: 'INTEGER', subtypeid: 'DOUBLE', recordstatus: 'INTEGER'}"
TARGET_IDS = (1000605, 1000489, 1000490)
PRESSOR_IDS = (1000462, 1000656, 1000657, 1000658, 71, 1000750, 1000649, 1000650, 1000655, 112, 113)
SEDATIVE_BOLUS = (1000700, 1001054, 251, 1001052, 1000691, 1000239, 245, 390, 442, 1000400, 1000857)
BP_LOWERING = (1000763, 1001047, 1000252, 1000695, 1000867, 1000243, 1000868, 1001048, 1001141, 1001142, 117, 1000596, 1000436, 94, 242, 1000477)
OFFSETS_H = (-28, -26, -24, -22, -20, 20, 22, 24, 26, 28)
SIC_ROOT = SICDB_ROOT
MET_ID, PARA_ID = (1610, 1420)
NE_ID, VASO_ID, EPI_ID = (1562, 1550, 1502)
SIC_COMED = (1480, 1495, 1499, 1549, 1520, 1896, 1521, 1546, 1547, 1556, 1578, 1671, 1640, 1494, 1400, 1712, 1747)

def event_rows(path: Path) -> pd.DataFrame:
    d = json.loads(path.read_text(encoding='utf-8'))
    return pd.DataFrame(d.get('event_rows') or d.get('analysis', {}).get('event_rows', []))

def hirid() -> None:
    import duckdb
    con = duckdb.connect()
    con.execute("SET memory_limit='3GB'")
    con.execute('SET threads=4')
    con.execute('SET preserve_insertion_order=false')
    con.execute(f"SET temp_directory='{DUCKDB_TEMP.as_posix()}'")
    con.execute(f"\n        CREATE TEMP TABLE pharma AS\n        SELECT patientid::INTEGER AS patientid, pharmaid::INTEGER AS pharmaid, givenat::TIMESTAMP AS givenat,\n               givendose::DOUBLE AS givendose, doseunit, route, infusionid::VARCHAR AS infusionid, recordstatus::INTEGER AS recordstatus\n        FROM read_csv('{PHARMA_GLOB}', union_by_name=true, header=true, columns={CSV_COLUMNS}, ignore_errors=true)\n        WHERE pharmaid IN {TARGET_IDS + PRESSOR_IDS + SEDATIVE_BOLUS + BP_LOWERING} AND givenat IS NOT NULL\n    ")
    con.execute(f"\n        CREATE TEMP TABLE eligible AS SELECT * FROM pharma\n        WHERE pharmaid IN {TARGET_IDS} AND route IN ('iv-inj','iv-inf','cv-inj','cv-inf') AND givendose > 0\n          AND (recordstatus & 2) = 0 AND (recordstatus & 32) = 0\n    ")
    parts = []
    for stratum, fname in [('nonpressor', 'hirid_metamizole_crossover_analysis.json'), ('pressor', 'hirid_metamizole_pressor_analysis.json')]:
        x = event_rows(RES / fname)
        parts.append(pd.DataFrame({'key': [f'hirid_{stratum}:{i}' for i in range(len(x))], 'patientid': x.patientid.astype(int), 't0': pd.to_datetime(x.current_at), 'drug': x.current_drug.astype(str)}))
    xo = pd.concat(parts, ignore_index=True)
    con.register('xo_df', xo)
    con.execute('CREATE TEMP TABLE xo AS SELECT * FROM xo_df')
    con.execute(f"\n        COPY (SELECT x.key, CAST(FLOOR(EPOCH(m.datetime - x.t0) / 120.0) * 2 AS INTEGER) AS bin, AVG(m.vm5) AS v, COUNT(*) AS n\n              FROM xo x JOIN read_parquet('{MAP_GLOB}') m\n                ON m.patientid = x.patientid AND m.vm5 IS NOT NULL\n               AND m.datetime >= x.t0 - INTERVAL 60 MINUTE AND m.datetime < x.t0 + INTERVAL 180 MINUTE\n              GROUP BY ALL) TO '{(OUT / 'hirid_xo_bins.parquet').as_posix()}' (FORMAT parquet)\n    ")
    con.execute(f"\n        COPY (SELECT x.key, MAX(e.givendose) AS dose, ANY_VALUE(e.doseunit) AS doseunit, STRING_AGG(DISTINCT e.route, ',') AS route,\n                     MIN(e.pharmaid) AS pharmaid, COUNT(*) AS n_rows\n              FROM xo x JOIN eligible e ON e.patientid = x.patientid AND e.givenat = x.t0\n              GROUP BY x.key) TO '{(OUT / 'hirid_xo_dose.parquet').as_posix()}' (FORMAT parquet)\n    ")
    con.execute(f"\n        COPY (SELECT x.key,\n                     MAX(CASE WHEN p.pharmaid IN {SEDATIVE_BOLUS} AND p.givenat >= x.t0 - INTERVAL 30 MINUTE AND p.givenat < x.t0 THEN 1 ELSE 0 END) AS sed_pre,\n                     MAX(CASE WHEN p.pharmaid IN {SEDATIVE_BOLUS} AND p.givenat >= x.t0 AND p.givenat <= x.t0 + INTERVAL 120 MINUTE THEN 1 ELSE 0 END) AS sed_post,\n                     MAX(CASE WHEN p.pharmaid IN {BP_LOWERING} AND p.givenat >= x.t0 - INTERVAL 30 MINUTE AND p.givenat < x.t0 THEN 1 ELSE 0 END) AS bpl_pre,\n                     MAX(CASE WHEN p.pharmaid IN {BP_LOWERING} AND p.givenat >= x.t0 AND p.givenat <= x.t0 + INTERVAL 120 MINUTE THEN 1 ELSE 0 END) AS bpl_post\n              FROM xo x LEFT JOIN pharma p ON p.patientid = x.patientid AND p.pharmaid IN {SEDATIVE_BOLUS + BP_LOWERING}\n               AND p.givendose > 0 AND (p.recordstatus & 2) = 0 AND (p.recordstatus & 32) = 0\n               AND p.givenat BETWEEN x.t0 - INTERVAL 30 MINUTE AND x.t0 + INTERVAL 120 MINUTE\n              GROUP BY x.key) TO '{(OUT / 'hirid_xo_comed.parquet').as_posix()}' (FORMAT parquet)\n    ")
    print('hirid crossover done', flush=True)
    con.execute('\n        CREATE TEMP TABLE both_patient AS SELECT patientid FROM eligible GROUP BY patientid\n        HAVING BOOL_OR(pharmaid = 1000605) AND BOOL_OR(pharmaid IN (1000489, 1000490))\n    ')
    con.execute('CREATE TEMP TABLE eb AS SELECT e.rowid AS rid, e.* FROM eligible e JOIN both_patient USING (patientid)')
    con.execute('\n        CREATE TEMP TABLE real_iso AS\n        SELECT r.rid, r.patientid, r.pharmaid, r.givenat AS t0 FROM eb r\n        WHERE NOT EXISTS (SELECT 1 FROM eligible x WHERE x.patientid = r.patientid AND x.givenat <> r.givenat\n                          AND x.givenat >= r.givenat - INTERVAL 120 MINUTE AND x.givenat <= r.givenat + INTERVAL 180 MINUTE)\n    ')
    con.execute("\n        CREATE TEMP TABLE press_int AS\n        WITH r AS (SELECT patientid, givenat, infusionid FROM pharma\n                   WHERE pharmaid IN (1000462, 1000656, 1000657, 1000658, 71, 1000750, 1000649, 1000650, 1000655, 112, 113)\n                     AND patientid IN (SELECT patientid FROM both_patient) AND givendose > 0 AND route IN ('cv-inf','iv-inf')\n                     AND (recordstatus & 2) = 0 AND (recordstatus & 32) = 0),\n             x AS (SELECT *, LEAD(givenat) OVER (PARTITION BY patientid, infusionid ORDER BY givenat) AS next_at FROM r)\n        SELECT patientid, givenat AS start_at,\n               CASE WHEN next_at IS NOT NULL AND next_at <= givenat + INTERVAL 15 MINUTE THEN next_at ELSE givenat + INTERVAL 10 MINUTE END AS stop_at\n        FROM x\n    ")
    con.execute('\n        CREATE TEMP TABLE cand AS\n        SELECT r.rid, r.patientid, r.t0 + off.h * INTERVAL 1 HOUR AS t0, off.h AS offset_h FROM real_iso r\n        CROSS JOIN (VALUES (-28),(-26),(-24),(-22),(-20),(20),(22),(24),(26),(28)) off(h)\n        WHERE NOT EXISTS (SELECT 1 FROM eligible x WHERE x.patientid = r.patientid\n                          AND x.givenat >= r.t0 + off.h * INTERVAL 1 HOUR - INTERVAL 120 MINUTE\n                          AND x.givenat <= r.t0 + off.h * INTERVAL 1 HOUR + INTERVAL 180 MINUTE)\n    ')
    con.execute("\n        CREATE TEMP TABLE idx AS\n        SELECT 'real' AS grp, rid, patientid, t0, 0 AS offset_h FROM real_iso\n        UNION ALL SELECT 'cand', rid, patientid, t0, offset_h FROM cand\n    ")
    con.execute('CREATE TEMP TABLE idx2 AS SELECT ROW_NUMBER() OVER () AS iid, * FROM idx')
    con.execute(f"\n        CREATE TEMP TABLE summ AS\n        SELECT i.iid, i.grp, i.rid, i.patientid, i.t0, i.offset_h,\n               AVG(m.vm5) FILTER (WHERE m.datetime >= i.t0 - INTERVAL 30 MINUTE AND m.datetime < i.t0) AS pre_mean,\n               REGR_SLOPE(m.vm5, EPOCH(m.datetime - i.t0)) FILTER (WHERE m.datetime >= i.t0 - INTERVAL 30 MINUTE AND m.datetime < i.t0) AS pre_slope,\n               ARG_MAX(m.vm2, m.datetime) FILTER (WHERE m.vm2 IS NOT NULL AND m.datetime < i.t0) AS pre_temp,\n               COUNT(m.vm5) FILTER (WHERE m.datetime >= i.t0 - INTERVAL 30 MINUTE AND m.datetime < i.t0) AS pre_n,\n               COUNT(m.vm5) FILTER (WHERE m.datetime >= i.t0 + INTERVAL 30 MINUTE AND m.datetime < i.t0 + INTERVAL 120 MINUTE) AS post_n\n        FROM idx2 i JOIN read_parquet('{MAP_GLOB}') m\n          ON m.patientid = i.patientid AND m.datetime >= i.t0 - INTERVAL 240 MINUTE AND m.datetime < i.t0 + INTERVAL 120 MINUTE\n         AND (m.vm5 IS NOT NULL OR m.vm2 IS NOT NULL)\n        GROUP BY ALL\n    ")
    con.execute('\n        CREATE TEMP TABLE summ2 AS\n        SELECT s.*, CASE WHEN EXISTS (SELECT 1 FROM press_int p WHERE p.patientid = s.patientid AND p.start_at <= s.t0 AND s.t0 < p.stop_at) THEN 1 ELSE 0 END AS pressor,\n               (EXTRACT(hour FROM s.t0) * 60 + EXTRACT(minute FROM s.t0)) AS clock_min\n        FROM summ s WHERE s.pre_n >= 3 AND s.post_n >= 3\n    ')
    con.execute("\n        CREATE TEMP TABLE best AS\n        SELECT * FROM (\n          SELECT c.iid, c.rid, c.patientid, c.t0, c.offset_h, c.pre_mean, c.pressor,\n                 ROW_NUMBER() OVER (PARTITION BY c.rid ORDER BY\n                   ABS(c.pre_mean - r.pre_mean) / 5.0 + ABS(c.pre_slope - r.pre_slope) * 60.0 / 2.0\n                   + COALESCE(ABS(c.pre_temp - r.pre_temp), 0.0)\n                   + CASE WHEN c.pressor <> r.pressor THEN 10.0 ELSE 0.0 END\n                   + LEAST(ABS(c.clock_min - r.clock_min), 1440 - ABS(c.clock_min - r.clock_min)) / 120.0) AS k\n          FROM summ2 c JOIN summ2 r ON r.rid = c.rid AND r.grp = 'real'\n          WHERE c.grp = 'cand') WHERE k = 1\n    ")
    con.execute("\n        CREATE TEMP TABLE sham_idx AS\n        SELECT r.iid, 'real' AS grp, r.rid, r.patientid, r.t0, e.pharmaid, r.pressor, r.pre_mean, r.pre_slope, r.pre_temp, r.clock_min,\n               CASE WHEN b.rid IS NULL THEN 0 ELSE 1 END AS matched\n        FROM summ2 r JOIN eb e ON e.rid = r.rid LEFT JOIN best b ON b.rid = r.rid WHERE r.grp = 'real'\n        UNION ALL\n        SELECT b.iid, 'sham', b.rid, b.patientid, b.t0, e.pharmaid, b.pressor, b.pre_mean, NULL, NULL, NULL, 1\n        FROM best b JOIN eb e ON e.rid = b.rid\n    ")
    con.execute(f"COPY sham_idx TO '{(OUT / 'hirid_sham_index.parquet').as_posix()}' (FORMAT parquet)")
    con.execute(f"\n        COPY (SELECT s.iid, CAST(FLOOR(EPOCH(m.datetime - s.t0) / 120.0) * 2 AS INTEGER) AS bin, AVG(m.vm5) AS v, COUNT(*) AS n\n              FROM sham_idx s JOIN read_parquet('{MAP_GLOB}') m\n                ON m.patientid = s.patientid AND m.vm5 IS NOT NULL\n               AND m.datetime >= s.t0 - INTERVAL 60 MINUTE AND m.datetime < s.t0 + INTERVAL 180 MINUTE\n              GROUP BY ALL) TO '{(OUT / 'hirid_sham_bins.parquet').as_posix()}' (FORMAT parquet)\n    ")
    sched = '(5),(10),(15),(20),(25),(30),(40),(50),(60),(75),(90),(105),(120),(135),(150),(165),(180)'
    con.execute(f"\n        COPY (\n          WITH s AS (SELECT iid, patientid, t0 FROM sham_idx WHERE matched = 1),\n          sc AS (SELECT * FROM s CROSS JOIN (VALUES {sched}) q(minute)),\n          pts AS (SELECT sc.iid, sc.minute, arg_min(m.vm5, ABS(EPOCH(m.datetime - (sc.t0 + sc.minute * INTERVAL 1 MINUTE)))) AS v\n                  FROM sc JOIN read_parquet('{MAP_GLOB}') m ON m.patientid = sc.patientid AND m.vm5 IS NOT NULL\n                   AND m.datetime BETWEEN sc.t0 + sc.minute * INTERVAL 1 MINUTE - INTERVAL 3 MINUTE AND sc.t0 + sc.minute * INTERVAL 1 MINUTE + INTERVAL 3 MINUTE\n                  GROUP BY ALL),\n          base AS (SELECT s.iid, arg_max(m.vm5, m.datetime) AS baseline FROM s JOIN read_parquet('{MAP_GLOB}') m\n                   ON m.patientid = s.patientid AND m.vm5 IS NOT NULL AND m.datetime < s.t0 AND m.datetime >= s.t0 - INTERVAL 60 MINUTE GROUP BY ALL)\n          SELECT b.iid, COUNT(p.v) AS points, MAX(CASE WHEN p.v <= b.baseline * 0.85 THEN 1 ELSE 0 END) AS event\n          FROM base b LEFT JOIN pts p ON p.iid = b.iid GROUP BY ALL\n        ) TO '{(OUT / 'hirid_sham_cantais.parquet').as_posix()}' (FORMAT parquet)\n    ")
    print('hirid sham done', con.execute('SELECT grp, matched, COUNT(*) FROM sham_idx GROUP BY ALL ORDER BY ALL').fetchall(), flush=True)

def decode(raw: str, val: str, offset: int):
    if raw and raw.startswith('0x') and (len(raw) == 482):
        try:
            v = np.frombuffer(bytes.fromhex(raw[2:]), dtype='<f4').astype(float)
            t = offset + 60 * np.arange(60)
            ok = (v != 0) & np.isfinite(v)
            return (t[ok], v[ok])
        except ValueError:
            pass
    try:
        x = float(val)
        return (np.array([offset]), np.array([x])) if np.isfinite(x) else (np.array([]), np.array([]))
    except (TypeError, ValueError):
        return (np.array([]), np.array([]))

def sicdb() -> None:
    med = pd.read_csv(SIC_ROOT / 'medication.csv.gz', usecols=['CaseID', 'PatientID', 'DrugID', 'Offset', 'OffsetDrugEnd', 'IsSingleDose', 'Amount', 'AmountPerMinute'])
    med = med[med.DrugID.isin((MET_ID, PARA_ID, NE_ID, VASO_ID, EPI_ID) + SIC_COMED)].copy()
    tgt = med[med.DrugID.isin((MET_ID, PARA_ID))].sort_values(['CaseID', 'Offset'])
    press = med[med.DrugID.isin((NE_ID, VASO_ID, EPI_ID))]
    comed = med[med.DrugID.isin(SIC_COMED) & (med.IsSingleDose == 1)]
    parts = []
    for stratum, fname in [('nonpressor', 'sicdb_metamizole_nopressor_analysis.json'), ('pressor', 'sicdb_metamizole_pressor_analysis.json')]:
        x = event_rows(RES / fname)
        parts.append(pd.DataFrame({'key': [f'sicdb_{stratum}:{i}' for i in range(len(x))], 'case': x.CaseID.astype(int), 't0': x.Offset.astype(float), 'drug': x.drug.astype(str), 'DrugID': x.DrugID.astype(int)}))
    xo = pd.concat(parts, ignore_index=True)
    d = xo.merge(tgt[['CaseID', 'DrugID', 'Offset', 'OffsetDrugEnd', 'Amount', 'IsSingleDose']].drop_duplicates(['CaseID', 'DrugID', 'Offset']), left_on=['case', 'DrugID', 't0'], right_on=['CaseID', 'DrugID', 'Offset'], how='left')
    d['duration_min'] = (d.OffsetDrugEnd - d.Offset) / 60
    d[['key', 'Amount', 'duration_min', 'IsSingleDose']].to_parquet(OUT / 'sicdb_xo_dose.parquet')
    cm = comed.groupby('CaseID').Offset.apply(lambda s: np.sort(s.values)).to_dict()
    flags = []
    for k, c, t in zip(xo.key, xo.case, xo.t0):
        o = cm.get(c)
        pre = post = 0
        if o is not None:
            pre = int(((o >= t - 1800) & (o < t)).any())
            post = int(((o >= t) & (o <= t + 7200)).any())
        flags.append((k, pre, post))
    pd.DataFrame(flags, columns=['key', 'comed_pre', 'comed_post']).to_parquet(OUT / 'sicdb_xo_comed.parquet')
    both = tgt.groupby('CaseID').DrugID.agg(lambda s: {MET_ID, PARA_ID}.issubset(set(s)))
    tb = tgt[tgt.CaseID.isin(both[both].index)]
    times = tb.groupby('CaseID').Offset.apply(lambda s: np.unique(s.values)).to_dict()

    def isolated(case, t):
        o = times[case]
        lo, hi = (np.searchsorted(o, t - 7200, 'left'), np.searchsorted(o, t + 10800, 'right'))
        return hi - lo == 0
    real, cand = ([], [])
    for r in tb.drop_duplicates(['CaseID', 'Offset']).itertuples():
        o = times[r.CaseID]
        lo, hi = (np.searchsorted(o, r.Offset - 7200, 'left'), np.searchsorted(o, r.Offset + 10800, 'right'))
        if hi - lo != 1:
            continue
        real.append((r.CaseID, float(r.Offset), int(r.DrugID)))
    real = pd.DataFrame(real, columns=['case', 't0', 'DrugID'])
    real['rid'] = np.arange(len(real))
    for r in real.itertuples():
        for h in OFFSETS_H:
            t = r.t0 + h * 3600
            if t >= 0 and isolated(r.case, t):
                cand.append((r.case, t, r.rid, h))
    cand = pd.DataFrame(cand, columns=['case', 't0', 'rid', 'offset_h'])
    idx = pd.concat([real.assign(grp='real', offset_h=0), cand.assign(grp='cand', DrugID=-1)], ignore_index=True)
    idx['iid'] = np.arange(len(idx))
    pint = {c: g[['Offset', 'OffsetDrugEnd']].values.astype(float) for c, g in press.groupby('CaseID')}
    idx['pressor'] = 0
    for c, g in idx.groupby('case'):
        iv_ = pint.get(c)
        if iv_ is None:
            continue
        T = g.t0.values[None, :]
        idx.loc[g.index, 'pressor'] = ((iv_[:, :1] <= T) & (T < iv_[:, 1:2])).any(0).astype(int)
    print('sicdb indices', len(real), len(cand), flush=True)
    N = len(idx)
    NX = len(xo)
    pre = np.zeros((N, 5))
    b10s = np.zeros((N, 24), np.float32)
    b10n = np.zeros((N, 24), np.int32)
    nadir = np.full(N, np.inf)
    tlast = np.full(N, -np.inf)
    tval = np.full(N, np.nan)
    x2s = np.zeros((NX, 120), np.float32)
    x2n = np.zeros((NX, 120), np.int32)

    def build(df, case_col):
        m = {}
        for c, g in df.groupby(case_col):
            g = g.sort_values('t0')
            m[int(c)] = (g.t0.values, g.index.values)
        return m
    im = build(idx, 'case')
    xm = build(xo, 'case')
    rows = 0
    with gzip.open(SIC_ROOT / 'data_float_h.csv.gz', 'rt', newline='', encoding='utf-8', errors='replace') as fh:
        rd = csv.reader(fh)
        h = next(rd)
        ic, idd, io, iv, ir = (h.index(k) for k in ('CaseID', 'DataID', 'Offset', 'Val', 'rawdata'))
        for row in rd:
            rows += 1
            if rows % 20000000 == 0:
                print('rows', rows, flush=True)
            did = row[idd]
            if did != '703' and did != '709':
                continue
            case = int(row[ic])
            a = im.get(case)
            b = xm.get(case) if did == '703' else None
            if a is None and b is None:
                continue
            base = int(row[io])
            tt, vv = decode(row[ir], row[iv], base)
            if len(tt) == 0:
                continue
            if did == '709':
                if a is None:
                    continue
                st, ii = a
                lo, hi = (np.searchsorted(st, tt.min(), 'right'), np.searchsorted(st, tt.max() + 14400, 'right'))
                for k in range(lo, hi):
                    j = ii[k]
                    t0 = st[k]
                    ok = (tt < t0) & (tt >= t0 - 14400)
                    if ok.any():
                        tm = tt[ok].max()
                        if tm > tlast[j]:
                            tlast[j] = tm
                            tval[j] = vv[ok][np.argmax(tt[ok])]
                continue
            if a is not None:
                st, ii = a
                lo, hi = (np.searchsorted(st, tt.min() - 10800, 'right'), np.searchsorted(st, tt.max() + 3600, 'right'))
                for k in range(lo, hi):
                    j = ii[k]
                    dt = tt - st[k]
                    w = (dt >= -3600) & (dt < 10800)
                    if not w.any():
                        continue
                    d_, v_ = (dt[w], vv[w])
                    bi = ((d_ + 3600) // 600).astype(int)
                    np.add.at(b10s[j], bi, v_)
                    np.add.at(b10n[j], bi, 1)
                    p = (d_ >= -1800) & (d_ < 0)
                    if p.any():
                        tm = d_[p] / 60.0
                        pv = v_[p]
                        pre[j] += (len(pv), tm.sum(), pv.sum(), (tm * pv).sum(), (tm * tm).sum())
                    q = (d_ >= 0) & (d_ < 7200)
                    if q.any():
                        nadir[j] = min(nadir[j], v_[q].min())
            if b is not None:
                st, ii = b
                lo, hi = (np.searchsorted(st, tt.min() - 10800, 'right'), np.searchsorted(st, tt.max() + 3600, 'right'))
                for k in range(lo, hi):
                    j = ii[k]
                    dt = tt - st[k]
                    w = (dt >= -3600) & (dt < 10800)
                    if w.any():
                        bi = ((dt[w] + 3600) // 120).astype(int)
                        np.add.at(x2s[j], bi, vv[w])
                        np.add.at(x2n[j], bi, 1)
    n, St, Sv, Stv, Stt = pre.T
    with np.errstate(invalid='ignore', divide='ignore'):
        idx['pre_mean'] = Sv / n
        idx['pre_slope'] = (n * Stv - St * Sv) / (n * Stt - St ** 2)
    idx['pre_n'] = n
    idx['pre_temp'] = tval
    idx['nadir'] = np.where(np.isfinite(nadir), nadir, np.nan)
    idx['clock_min'] = 0.0
    for j in range(24):
        with np.errstate(invalid='ignore', divide='ignore'):
            idx[f'b{j * 10 - 60}'] = np.where(b10n[:, j] > 0, b10s[:, j] / np.maximum(b10n[:, j], 1), np.nan)
    idx.to_parquet(OUT / 'sicdb_sham_index_raw.parquet')
    recs = [(xo.key.iat[e], int(j * 2 - 60), float(x2s[e, j] / x2n[e, j]), int(x2n[e, j])) for e in range(NX) for j in np.nonzero(x2n[e])[0]]
    pd.DataFrame(recs, columns=['key', 'bin', 'v', 'n']).to_parquet(OUT / 'sicdb_xo_bins.parquet')
    print('sicdb done', flush=True)
if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('hirid', 'all'):
        hirid()
    if what in ('sicdb', 'all'):
        sicdb()
