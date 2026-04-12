"""
implementation/train.py
======================
Trains ML models for PFAS Risk.
"""

import logging
import pickle
import json
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "dataset" / "pfas_features.parquet"
MODELS_DIR = ROOT / "outputs" / "models"
RESULTS_DIR = ROOT / "outputs"

FEATURE_COLS = [
    "substance_ord", "is_long_chain", "carbon_chain_length", "is_sulfonyl",
    "is_aquatic", "is_soil_based", "is_wastewater",
    "year_normalized", "month",
    "spatial_density_50km", "mean_log_value_50km",
    "elevation_m", "land_use_class", "annual_precip_mm",
    "soil_permeability", "population_density_log",
    "dist_to_airport_km", "dist_to_military_km", "dist_to_wastewater_km",
]

def train():
    if not FEATURES_PATH.exists(): raise FileNotFoundError(f"Missing {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Classification
    log.info("Training Classification Model...")
    clf_df = df.dropna(subset=["above_100_ng_l"]).copy()
    X = clf_df[FEATURE_COLS]
    y = clf_df["above_100_ng_l"].values
    
    clf = LGBMClassifier(n_estimators=100, num_leaves=31, learning_rate=0.05, verbose=-1)
    clf.fit(X, y)
    with open(MODELS_DIR / "lgbm_exceedance_v1.pkl", "wb") as f: pickle.dump(clf, f)
    
    # 2. Regression
    log.info("Training Regression Model...")
    reg_df = df.dropna(subset=["log_value"]).copy()
    X_reg = reg_df[FEATURE_COLS]
    y_reg = reg_df["log_value"].values
    
    reg = LGBMRegressor(n_estimators=100, num_leaves=31, learning_rate=0.05, verbose=-1)
    reg.fit(X_reg, y_reg)
    with open(MODELS_DIR / "lgbm_regression_v1.pkl", "wb") as f: pickle.dump(reg, f)
    
    # Save Schema
    with open(MODELS_DIR / "feature_schema.json", "w") as f: json.dump(FEATURE_COLS, f)
    
    log.info(f"Models saved to {MODELS_DIR}")

if __name__ == "__main__":
    train()
