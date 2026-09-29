# D32

Reproducibility code for a within-patient comparison of intravenous metamizole and paracetamol in adult intensive care. The analyses use HiRID 1.1.1 and SICdb 1.0.8 and evaluate changes in mean arterial pressure, vasopressor dose, and temperature after drug administration.

## Data access

HiRID and SICdb are credentialed clinical datasets and are not included in this repository. Obtain each dataset from its official distributor and comply with its data-use agreement.

The scripts expect:

- HiRID raw pharma records under `raw_stage/pharma_records/csv/part-*.csv`.
- HiRID processed two-minute records as Parquet files containing `patientid`, `datetime`, `vm5` (MAP), and `vm2` (temperature).
- SICdb 1.0.8 CSV or CSV.GZ files in the original release layout.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

$env:D32_HIRID_ROOT = "D:\data\hirid-1.1.1"
$env:D32_HIRID_MAP_GLOB = "D:/data/hirid-processed/merged_stage/part-*.parquet"
$env:D32_SICDB_ROOT = "D:\data\sicdb-1.0.8"
$env:D32_RESULTS_DIR = "$PWD\results"
$env:D32_DUCKDB_TEMP = "$PWD\.duckdb_tmp"
```

The defaults point to `data/hirid`, `data/sicdb`, `results`, and `.duckdb_tmp` inside the repository. Data and generated outputs are excluded by `.gitignore`.

## Analysis workflow

Run the scripts from the repository root.

1. Feasibility and exposure audits:

```powershell
python hirid_metamizole_audit.py
python hirid_metamizole_exposure_audit.py
python hirid_metamizole_crossover_sample_audit.py
python hirid_metamizole_crossover_washout_sample_audit.py
```

2. Primary HiRID analyses:

```powershell
python hirid_metamizole_crossover_analysis.py
python hirid_metamizole_pressor_analysis.py
python hirid_metamizole_pressor_balance_audit.py
```

3. SICdb replication:

```powershell
python sicdb_metamizole_pressor_analysis.py
python sicdb_metamizole_nopressor_analysis.py
```

4. Identifiable-model reruns and follow-up analyses:

```powershell
python metamizole_followup.py
python metamizole_clean_rerun.py
python revision_extract.py all
python revision_analysis.py
python revision_robustness.py
python sicdb_raw_robustness.py
python sham_measurement_audit.py
python make_figure_data.py all
```

The extraction scripts process the large source tables in batches or with DuckDB projections. They do not require loading the full HiRID or SICdb databases into memory.

## Main design choices

- Drug administrations are compared within the same patient.
- The primary continuous outcome is mean MAP from 30 to 120 minutes after administration minus mean MAP during the preceding 30 minutes.
- Analyses are stratified by vasopressor use at administration.
- The main models use patient fixed effects with patient-clustered standard errors.
- Sham administration times and pre-administration placebo windows quantify natural MAP fluctuation and regression to the mean.
- Sensitivity analyses vary the minimum interval between antipyretic administrations and exclude concurrent interventions.

## Reproducibility notes

Generated event tables and result files can contain patient-level information and must remain in an approved secure environment. The repository contains analysis code only. Exact numerical reproduction also depends on the authorized dataset versions and the upstream HiRID preprocessing used to construct the two-minute Parquet files.

## License

Code is released under the MIT License. Dataset licenses and data-use agreements remain controlling for HiRID and SICdb.
