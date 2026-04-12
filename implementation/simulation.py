"""
implementation/simulation.py
==========================
PFAS Scenario Simulation Engine (Production Grade)
Handles perturbation of features and delta calculation for the "What-If Lab".
"""
import logging
import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "outputs" / "models"

class SimulationEngine:
    def __init__(self):
        self.model_path = MODELS_DIR / "lgbm_exceedance_v1.pkl"
        self.schema_path = MODELS_DIR / "feature_schema.json"
        
        if not self.model_path.exists():
            raise FileNotFoundError("Models not found. Run main.py first.")
            
        with open(self.model_path, "rb") as f: self.model = pickle.load(f)
        with open(self.schema_path, "r") as f: self.schema = json.load(f)

    def run_lab_scenario(self, base_features: pd.DataFrame, mods: dict):
        """
        Runs a simulation based on specific modifiers from the 'What-If Lab'.
        mods: {
            'industrial_increase': float (0-1),
            'airport_proximity_km': float,
            'rainfall_factor': float (0.5-2.0),
            'filtration_efficiency': float (0-1)
        }
        """
        df_sim = base_features.copy()
        
        # 1. Industrial Discharge Scenario
        if 'industrial_increase' in mods:
            # We simulate this by increasing spatial density markers
            df_sim['spatial_density_50km'] *= (1 + mods['industrial_increase'])
            
        # 2. Airport Runoff Scenario
        if 'airport_proximity_km' in mods:
            df_sim['dist_to_airport_km'] = mods['airport_proximity_km']
            
        # 3. Rainfall Transport
        if 'rainfall_factor' in mods:
            df_sim['annual_precip_mm'] *= mods['rainfall_factor']
            
        # 4. Cleanup Intervention
        if 'filtration_efficiency' in mods:
            # We simulate this by reducing the neighborhood mean
            df_sim['mean_log_value_50km'] *= (1 - mods['filtration_efficiency'])

        # Predict
        X = df_sim[self.schema]
        prob = self.model.predict_proba(X)[:, 1]
        
        return prob
