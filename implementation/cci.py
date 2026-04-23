import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "dataset" / "pfas_golden.parquet"
OUTPUTS_DIR = ROOT / "outputs" / "index"

THRESHOLDS = {"PFOS": 40, "PFOA": 40, "PFHxS": 100, "PFNA": 100, "PFDA": 100, "PFHpA": 100, "PFBS": 100}

def calculate_cci():
    if not DATA_PATH.exists(): raise FileNotFoundError(f"Missing {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    
    pivoted = df.pivot_table(index=['lat', 'lon', 'year'], columns='substance', values='value', aggfunc='max')
    
    norm_values = pd.DataFrame(index=pivoted.index)
    from tqdm import tqdm
    for sub, thresh in tqdm(THRESHOLDS.items(), desc="  Normalising compounds", leave=False):
        if sub in pivoted.columns:
            norm_values[sub] = (pivoted[sub] / thresh)**2
        else:
            norm_values[sub] = 0
            
    cci = norm_values.sum(axis=1)**0.5
    results = pd.DataFrame({'cci': cci})
    
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUTS_DIR / "pfas_cci.csv")
    log.info(f"CCI results saved to {OUTPUTS_DIR}/pfas_cci.csv")

if __name__ == "__main__":
    calculate_cci()
