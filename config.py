import os
from pathlib import Path
REPOSITORY_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = Path(os.environ.get('D32_RESULTS_DIR', REPOSITORY_ROOT / 'results')).resolve()
HIRID_ROOT = Path(os.environ.get('D32_HIRID_ROOT', REPOSITORY_ROOT / 'data' / 'hirid')).resolve()
SICDB_ROOT = Path(os.environ.get('D32_SICDB_ROOT', REPOSITORY_ROOT / 'data' / 'sicdb')).resolve()
HIRID_MAP_GLOB = os.environ.get('D32_HIRID_MAP_GLOB', str(HIRID_ROOT / 'processed' / 'merged_stage' / 'part-*.parquet')).replace('\\', '/')
DUCKDB_TEMP = Path(os.environ.get('D32_DUCKDB_TEMP', REPOSITORY_ROOT / '.duckdb_tmp')).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DUCKDB_TEMP.mkdir(parents=True, exist_ok=True)
