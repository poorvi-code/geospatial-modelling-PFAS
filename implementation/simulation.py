"""
implementation/simulation.py
============================
Gamified PFAS What-If Simulation Engine
----------------------------------------
Provides:
  - Preset scenario packs (Industrial Disaster, Airport Expansion, Cleanup Drive, etc.)
  - Continuous sliders for fine-grained control
  - Monte Carlo uncertainty (1 000 runs → p5/p50/p95 bands)
  - Risk Score (0–100) with named Levels (Safe Zone → Critical Alert)
  - Plain-English narrative for every result
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "outputs" / "models"


# Risk level taxonomy
RISK_LEVELS = [
    (0,  20,  "Safe Zone",       "#22c55e",
     "Contamination signals are very low. This area shows no major PFAS pressure."),
    (20, 40,  "Watch Zone",      "#eab308",
     "Some background contamination is present. Routine monitoring is advisable."),
    (40, 60,  "Caution Zone",    "#f97316",
     "Elevated risk detected. This area warrants detailed investigation and sampling."),
    (60, 80,  "High Alert",      "#ef4444",
     "High contamination probability. Regulatory action is strongly recommended."),
    (80, 101, "Critical Alert",  "#991b1b",
     "Extreme risk. Immediate intervention and public health advisory may be required."),
]


# Preset scenario packs
SCENARIO_PRESETS = {
    "baseline": {
        "label": "Baseline",
        "description": "Current conditions — no modifications applied.",
        "mods": {},
    },
    "industrial_spill": {
        "label": "Industrial Spill",
        "description": "Simulates a major industrial discharge event. Spatial contamination density surges.",
        "mods": {"spatial_density_boost": 3.0, "mean_log_value_boost": 1.5},
    },
    "airport_expansion": {
        "label": "Airport Expansion",
        "description": "A new large runway nearby increases AFFF foam usage risk.",
        "mods": {"airport_distance_km": 2.0},
    },
    "new_wastewater_plant": {
        "label": "New Wastewater Plant",
        "description": "A wastewater treatment plant is built 5 km away. PFAS leaching risk rises.",
        "mods": {"spatial_density_boost": 1.5, "mean_log_value_boost": 0.8},
    },
    "cleanup_basic": {
        "label": "Basic Cleanup",
        "description": "Standard remediation reduces local PFAS concentration by ~30%.",
        "mods": {"mean_log_value_reduction": 0.30},
    },
    "cleanup_advanced": {
        "label": "Advanced Treatment",
        "description": "State-of-the-art granular activated carbon filtration. ~70% reduction.",
        "mods": {"mean_log_value_reduction": 0.70},
    },
    "regulation_ban": {
        "label": "PFAS Use Ban",
        "description": "Regulatory ban on long-chain PFAS (simulates post-2018 effect). Contamination trends downward.",
        "mods": {"mean_log_value_reduction": 0.45, "is_post_2018_override": 1},
    },
    "worst_case": {
        "label": "Worst Case Combination",
        "description": "Industrial spill + nearby airport + no cleanup. Upper-bound risk estimate.",
        "mods": {
            "spatial_density_boost": 4.0,
            "mean_log_value_boost": 2.0,
            "airport_distance_km": 1.0,
        },
    },
}


@dataclass
class SimResult:
    scenario_label:    str
    base_prob:         float
    scenario_prob:     float
    base_score:        float
    scenario_score:    float
    risk_level:        str
    risk_color:        str
    risk_narrative:    str
    delta_pts:         float           # probability delta in percentage points
    mc_p5:             float = 0.0
    mc_p50:            float = 0.0
    mc_p95:            float = 0.0
    feature_deltas:    Dict[str, float] = field(default_factory=dict)
    plain_explanation: str = ""


def _prob_to_score(prob: float) -> float:
    """Map exceedance probability [0, 1] → risk score [0, 100]."""
    return float(np.clip(prob * 100.0, 0.0, 100.0))


def _score_to_level(score: float):
    for lo, hi, label, color, narrative in RISK_LEVELS:
        if lo <= score < hi:
            return label, color, narrative
    return RISK_LEVELS[-1][2], RISK_LEVELS[-1][3], RISK_LEVELS[-1][4]


def _build_plain_explanation(result: SimResult) -> str:
    direction = "increased" if result.delta_pts > 0 else "decreased"
    magnitude = abs(result.delta_pts)
    if magnitude < 2:
        change_desc = "barely changed"
    elif magnitude < 10:
        change_desc = f"{direction} slightly (by {magnitude:.1f} percentage points)"
    elif magnitude < 25:
        change_desc = f"{direction} noticeably (by {magnitude:.1f} percentage points)"
    else:
        change_desc = f"{direction} dramatically (by {magnitude:.1f} percentage points)"

    return (
        f"Under the {result.scenario_label} scenario, the estimated contamination probability "
        f"has {change_desc} from {result.base_prob*100:.1f}% to {result.scenario_prob*100:.1f}%. "
        f"The risk level is now {result.risk_level}. {result.risk_narrative}"
    )


class SimulationEngine:
    def __init__(self):
        clf_path    = MODELS_DIR / "lgbm_calibrated.pkl"
        legacy_path = MODELS_DIR / "lgbm_exceedance_v1.pkl"
        schema_path = MODELS_DIR / "feature_schema.json"

        if not (clf_path.exists() or legacy_path.exists()):
            raise FileNotFoundError(
                "No trained model found. Run `python main.py` first to build the pipeline."
            )

        model_file = clf_path if clf_path.exists() else legacy_path
        with open(model_file, "rb") as f:
            self.model = pickle.load(f)
        with open(schema_path, "r") as f:
            self.schema: list = json.load(f)

        log.info(f"SimulationEngine loaded model from {model_file.name}")

    
    # Core prediction helpers
    

    def _predict_prob(self, X: pd.DataFrame) -> float:
        X_aligned = X.reindex(columns=self.schema, fill_value=-1).fillna(-1)
        prob = self.model.predict_proba(X_aligned)[0, 1]
        return float(np.clip(prob, 0.0, 1.0))

    def _apply_mods(self, base: pd.DataFrame, mods: dict) -> pd.DataFrame:
        df = base.copy()
        if "spatial_density_boost" in mods:
            k = mods["spatial_density_boost"]
            if "spatial_density_50km" in df.columns:
                df["spatial_density_50km"] = df["spatial_density_50km"] * k
        if "mean_log_value_boost" in mods:
            delta = mods["mean_log_value_boost"]
            if "mean_log_value_50km" in df.columns:
                df["mean_log_value_50km"] = df["mean_log_value_50km"] + delta
        if "mean_log_value_reduction" in mods:
            frac = mods["mean_log_value_reduction"]
            if "mean_log_value_50km" in df.columns:
                df["mean_log_value_50km"] = df["mean_log_value_50km"] * (1 - frac)
        if "airport_distance_km" in mods:
            if "dist_to_airport_km" in df.columns:
                df["dist_to_airport_km"] = mods["airport_distance_km"]
        if "is_post_2018_override" in mods:
            if "is_post_2018" in df.columns:
                df["is_post_2018"] = int(mods["is_post_2018_override"])
        # Custom slider overrides
        for col in ["spatial_density_50km", "mean_log_value_50km",
                    "dist_to_airport_km", "nearest_training_point_km",
                    "year_normalized", "is_post_2018"]:
            key = f"override_{col}"
            if key in mods and col in df.columns:
                df[col] = mods[key]
        return df

    
    # Public API
    

    def run_preset(self, base_features: pd.DataFrame, preset_key: str) -> SimResult:
        """Run a named preset scenario against a base feature vector."""
        preset = SCENARIO_PRESETS.get(preset_key, SCENARIO_PRESETS["baseline"])
        return self.run_custom(base_features, preset["mods"], label=preset["label"])

    def run_custom(
        self,
        base_features: pd.DataFrame,
        mods: dict,
        label: str = "Custom Scenario",
    ) -> SimResult:
        """Run a custom scenario with explicit modifier dict."""
        base_prob     = self._predict_prob(base_features)
        mod_features  = self._apply_mods(base_features, mods)
        scenario_prob = self._predict_prob(mod_features)

        base_score     = _prob_to_score(base_prob)
        scenario_score = _prob_to_score(scenario_prob)
        risk_level, risk_color, risk_narrative = _score_to_level(scenario_score)

        # Feature delta dict (for waterfall chart in dashboard)
        feature_deltas = {}
        for col in self.schema:
            b_val = float(base_features[col].iloc[0]) if col in base_features.columns else 0.0
            s_val = float(mod_features[col].iloc[0])  if col in mod_features.columns  else 0.0
            if abs(b_val - s_val) > 1e-6:
                feature_deltas[col] = round(s_val - b_val, 4)

        result = SimResult(
            scenario_label  = label,
            base_prob       = base_prob,
            scenario_prob   = scenario_prob,
            base_score      = base_score,
            scenario_score  = scenario_score,
            risk_level      = risk_level,
            risk_color      = risk_color,
            risk_narrative  = risk_narrative,
            delta_pts       = (scenario_prob - base_prob) * 100.0,
            feature_deltas  = feature_deltas,
        )
        result.plain_explanation = _build_plain_explanation(result)
        return result

    def run_monte_carlo(
        self,
        base_features: pd.DataFrame,
        mods: dict,
        n_runs: int = 1000,
        noise_std: float = 0.15,
    ) -> SimResult:
        """Monte Carlo uncertainty estimation (1 000 perturbed predictions)."""
        result = self.run_custom(base_features, mods)
        mod_features = self._apply_mods(base_features, mods)
        probs = []

        rng = np.random.default_rng(seed=42)
        for _ in range(n_runs):
            perturbed = mod_features.copy()
            for col in self.schema:
                if col in perturbed.columns:
                    v = float(perturbed[col].iloc[0])
                    if v > 0:
                        perturbed[col] = v * (1 + rng.normal(0, noise_std))
                    else:
                        perturbed[col] = v + rng.normal(0, abs(v) * noise_std + 0.1)
            probs.append(self._predict_prob(perturbed))

        probs_arr = np.array(probs)
        result.mc_p5  = float(np.percentile(probs_arr, 5)  * 100)
        result.mc_p50 = float(np.percentile(probs_arr, 50) * 100)
        result.mc_p95 = float(np.percentile(probs_arr, 95) * 100)
        return result

    @staticmethod
    def get_presets() -> dict:
        return SCENARIO_PRESETS
