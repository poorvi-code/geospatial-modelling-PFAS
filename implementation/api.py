"""
implementation/api.py
=====================
PFAS Global Inference Engine
-----------------------------
Local-only — no external API calls. All inference is done against
locally trained LightGBM models + KD-trees.

PFASPredictor.predict(lat, lon, substance) →  dict with:
  - exceedance_prob        (calibrated probability ≥ 100 ng/L)
  - predicted_value_ngl    (back-transformed regression estimate)
  - dist_to_nearest_sample_km
  - dist_to_airport_km
  - shap_values            (dict feature → float)
  - feature_vector         (dict — used by SimulationEngine)
  - confidence_level       (HIGH / MEDIUM / LOW / EXTRAPOLATION)
  - confidence_note        (plain-English note)
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.spatial import KDTree

log = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "outputs" / "models"
KD_DIR     = ROOT / "outputs" / "kdtrees"
EARTH_R    = 6371.0

SUBSTANCES = ["PFOS", "PFOA", "PFHXS", "PFNA", "PFDA", "PFHPA", "PFBS"]

SUBSTANCE_ORD = {
    "PFBS": 0, "PFHPA": 1, "PFHXS": 2, "PFNA": 3, "PFDA": 4, "PFOA": 5, "PFOS": 6
}
CARBON_CHAIN = {
    "PFBS": 4, "PFHPA": 7, "PFHXS": 6, "PFNA": 9, "PFDA": 10, "PFOA": 8, "PFOS": 8
}
LONG_CHAIN = {"PFOS", "PFOA", "PFNA", "PFDA"}
SULFONYL   = {"PFOS", "PFHXS", "PFBS"}


class PFASPredictor:
    def __init__(self):
        # Try calibrated model first, fall back to uncalibrated
        clf_path     = MODELS_DIR / "lgbm_calibrated.pkl"
        clf_legacy   = MODELS_DIR / "lgbm_exceedance_v1.pkl"
        reg_path     = MODELS_DIR / "lgbm_regressor.pkl"
        reg_legacy   = MODELS_DIR / "lgbm_regression_v1.pkl"
        schema_path  = MODELS_DIR / "feature_schema.json"

        with open(clf_path if clf_path.exists() else clf_legacy, "rb") as f:
            self.clf = pickle.load(f)
        with open(reg_path if reg_path.exists() else reg_legacy, "rb") as f:
            self.reg = pickle.load(f)
        with open(schema_path, "r") as f:
            self.schema: list = json.load(f)

        with open(KD_DIR / "training_points.pkl", "rb") as f:
            self.tree_train = pickle.load(f)
        self.train_vals = np.load(KD_DIR / "training_log_values.npy")

        airport_pkl = KD_DIR / "airports.pkl"
        self.tree_air = None
        if airport_pkl.exists():
            with open(airport_pkl, "rb") as f:
                self.tree_air = pickle.load(f)

        # SHAP explainer — use the underlying estimator if calibrated
        base_clf = self.clf
        if hasattr(self.clf, "calibrated_classifiers_") and len(self.clf.calibrated_classifiers_) > 0:
            # For ensemble=False, the first one is the calibrated wrapper
            base_clf = self.clf.calibrated_classifiers_[0].estimator

        # Unpack wrappers like FrozenEstimator
        while hasattr(base_clf, "estimator"):
            # If the inner model is what we want (LightGBM), stop unpacking
            if "lightgbm" in str(type(base_clf)).lower():
                break
            base_clf = base_clf.estimator

        try:
            self.explainer = shap.TreeExplainer(base_clf, feature_perturbation="tree_path_dependent")
        except Exception:
            self.explainer = shap.TreeExplainer(base_clf)

        log.info("PFASPredictor initialised.")

    def build_feature_frame(
        self,
        lat: float,
        lon: float,
        substance: str = "PFOS",
        year: int = 2024,
        media_type: str = "surface water",
    ) -> tuple:
        """
        Build feature vector for any (lat, lon) on Earth.
        Returns (X: DataFrame, nearest_km: float, airport_km: float)
        """
        pt = np.deg2rad([[lat, lon]])

        d_tr, _ = self.tree_train.query(pt, k=1)
        nearest_km = float(d_tr[0]) * EARTH_R

        airport_km = -1.0
        if self.tree_air is not None:
            d_air, _ = self.tree_air.query(pt, k=1)
            airport_km = float(d_air[0]) * EARTH_R

        # Spatial lag (50 km)
        idx_50 = self.tree_train.query_ball_point(pt[0], r=50.0 / EARTH_R)
        mean_log_50   = float(np.mean(self.train_vals[idx_50])) if idx_50 else 0.0
        density_50    = len(idx_50)

        sub_upper = substance.upper().strip()
        sub_ord   = SUBSTANCE_ORD.get(sub_upper, 6)
        is_long   = int(sub_upper in LONG_CHAIN)
        c_chain   = CARBON_CHAIN.get(sub_upper, 8)
        is_sulf   = int(sub_upper in SULFONYL)

        m = media_type.lower()
        is_aquatic    = int(any(k in m for k in ["water", "groundwater", "sea", "drink", "surface"]))
        is_soil       = int(any(k in m for k in ["soil", "sediment"]))
        is_wastewater = int(any(k in m for k in ["waste", "leach"]))

        yr_norm  = (year - 2001) / 23.0
        post2018 = int(year >= 2018)

        feat = {f: -1.0 for f in self.schema}
        feat.update({
            "substance_ord":             sub_ord,
            "is_long_chain":             is_long,
            "carbon_chain_length":       c_chain,
            "is_sulfonyl":               is_sulf,
            "is_aquatic":                is_aquatic,
            "is_soil_based":             is_soil,
            "is_wastewater":             is_wastewater,
            "year_normalized":           yr_norm,
            "is_post_2018":              post2018,
            "month":                     6,  # Default to mid-year for sensitivity
            "spatial_density_50km":      density_50,
            "mean_log_value_50km":       mean_log_50,
            "nearest_training_point_km": nearest_km,
            "dist_to_airport_km":        airport_km,
        })

        X = pd.DataFrame([feat])[self.schema]
        return X, nearest_km, airport_km

    def _confidence(self, nearest_km: float) -> tuple:
        if nearest_km < 100:
            return "HIGH",          "Location is within 100 km of training data."
        if nearest_km < 500:
            return "MEDIUM",        "Location is 100–500 km from training data. Use with caution."
        if nearest_km < 2000:
            return "LOW",           "Location is 500–2000 km from training data. Rough estimate only."
        return "EXTRAPOLATION",     "Location is >2000 km from training data. Treat as speculative."

    def predict(
        self,
        lat: float,
        lon: float,
        substance: str = "PFOS",
        year: int = 2024,
        media_type: str = "surface water",
    ) -> dict:
        if substance.upper() == "GENERAL":
            # For General PFAS, we aggregate results across all compounds
            sub_results = [
                self.predict(lat, lon, sub, year, media_type)
                for sub in SUBSTANCES
            ]
            
            # Probability: Probability that *at least one* compound exceeds 100 ng/L
            # Since compounds often co-occur, we take the max probability as the base
            # and add a small contribution from others if they are significant.
            # A more robust aggregation for exceedance is 1 - prod(1-p_i)
            # which assumes independence (conservative/risk-averse).
            total_prob = 1.0 - np.prod([1.0 - r["exceedance_prob"] for r in sub_results])
            
            # Concentration: Simple sum of estimated concentrations
            total_conc = sum(r["predicted_value_ngl"] for r in sub_results)
            
            # Log prediction: log(total_conc + 1)
            total_log = np.log1p(total_conc)
            
            # SHAP: Average SHAP values (heuristic for importance in General score)
            all_shap = {}
            for r in sub_results:
                for k, v in r["shap_values"].items():
                    all_shap[k] = all_shap.get(k, 0.0) + v
            for k in all_shap:
                all_shap[k] /= len(sub_results)
                
            # Basic info from the first result (spatial distance is the same)
            base = sub_results[0]
            
            return {
                "exceedance_prob":           float(total_prob),
                "predicted_value_ngl":       float(total_conc),
                "log_prediction":            float(total_log),
                "dist_to_nearest_sample_km": base["dist_to_nearest_sample_km"],
                "dist_to_airport_km":        base["dist_to_airport_km"],
                "confidence_level":          base["confidence_level"],
                "confidence_note":           base["confidence_note"],
                "shap_values":               all_shap,
                "feature_vector":            base["feature_vector"],
                "substance":                 "GENERAL",
            }

        X, nearest_km, airport_km = self.build_feature_frame(lat, lon, substance, year, media_type)

        prob = float(self.clf.predict_proba(X)[0, 1])
        log_val = float(self.reg.predict(X)[0])
        conc = float(np.expm1(log_val))

        # SHAP
        try:
            raw_shap = self.explainer.shap_values(X)
            if isinstance(raw_shap, list):
                local_shap = np.array(raw_shap[1][0])
            else:
                local_shap = np.array(raw_shap[0])
        except Exception:
            local_shap = np.zeros(len(self.schema))

        conf_level, conf_note = self._confidence(nearest_km)

        return {
            "exceedance_prob":           prob,
            "predicted_value_ngl":       conc,
            "log_prediction":            log_val,
            "dist_to_nearest_sample_km": nearest_km,
            "dist_to_airport_km":        airport_km,
            "confidence_level":          conf_level,
            "confidence_note":           conf_note,
            "shap_values":               dict(zip(self.schema, local_shap.tolist())),
            "feature_vector":            X.to_dict(orient="records")[0],
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = PFASPredictor()
    r = p.predict(51.5, -0.1, substance="PFOS")
    print(f"Exceedance prob : {r['exceedance_prob']:.3f}")
    print(f"Concentration   : {r['predicted_value_ngl']:.1f} ng/L")
    print(f"Confidence      : {r['confidence_level']} — {r['confidence_note']}")
