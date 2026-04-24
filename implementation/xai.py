"""
implementation/xai.py
=====================
Explainable AI Engine - Chat-Based Natural Language Interface
--------------------------------------------------------------
What this does:
  1. Computes SHAP values (TreeExplainer) for any prediction.
  2. Generates a rich plain-English explanation of WHAT the model
     found and WHY it reached its conclusion.
  3. Provides a conversational Q&A function that answers any
     user question about the results, the model, or PFAS science
     in plain language - no jargon, no LLM API required.

No external API or LLM wrapper is used. All explanations are
deterministic, template-driven, and grounded in the actual model output.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import shap

# ---------------------------------------------------------------------------
# Feature human-readable labels
# ---------------------------------------------------------------------------
FEATURE_LABELS: Dict[str, str] = {
    "substance_ord":             "PFAS Compound Type",
    "is_long_chain":             "Long-Chain Compound (more persistent)",
    "carbon_chain_length":       "Carbon Chain Length",
    "is_sulfonyl":               "Sulfonate Group (PFOS/PFHxS/PFBS family)",
    "is_aquatic":                "Water-Based Sampling Location",
    "is_soil_based":             "Soil/Sediment Sampling Location",
    "is_wastewater":             "Wastewater/Leachate Source",
    "year_normalized":           "Measurement Year",
    "is_post_2018":              "Post-2018 (EU PFAS Restrictions Era)",
    "month":                     "Month of Measurement",
    "spatial_density_50km":      "Number of PFAS Measurements Nearby (50 km)",
    "mean_log_value_50km":       "Average PFAS Levels in the Area",
    "nearest_training_point_km": "Distance to Nearest Known PFAS Measurement",
    "dist_to_airport_km":        "Distance to Nearest Airport",
    "elevation_m":               "Elevation Above Sea Level",
    "land_use_class":            "Land Use Type",
    "annual_precip_mm":          "Annual Rainfall",
    "soil_permeability":         "Soil Permeability (PFAS leaching risk)",
    "population_density_log":    "Population Density (exposure risk)",
    "dist_to_military_km":       "Distance to Military Base",
    "dist_to_wastewater_km":     "Distance to Wastewater Treatment Plant",
}

# ---------------------------------------------------------------------------
# PFAS science knowledge base (used for Q&A)
# ---------------------------------------------------------------------------
PFAS_KNOWLEDGE: Dict[str, str] = {
    "pfas": (
        "**PFAS** (Per- and Polyfluoroalkyl Substances) are a group of over 10,000 man-made "
        "chemicals. Because the carbon-fluorine bond is one of the strongest in chemistry, PFAS "
        "do not break down naturally - earning them the nickname 'forever chemicals'. They are "
        "found in non-stick cookware, food packaging, firefighting foam, and many industrial "
        "processes."
    ),
    "pfos": (
        "**PFOS** (Perfluorooctane Sulfonic Acid) is an 8-carbon sulfonate PFAS. It was the "
        "primary ingredient in Scotchgard and military/airport firefighting foam (AFFF). "
        "It bioaccumulates in human blood and is linked to thyroid disease, immune suppression, "
        "and cancer. It has been restricted under the Stockholm Convention since 2009."
    ),
    "pfoa": (
        "**PFOA** (Perfluorooctanoic Acid) was used to manufacture Teflon non-stick coatings. "
        "It is a probable human carcinogen and was phased out voluntarily by manufacturers "
        "in the US by 2015. Drinking water contamination from PFOA has been documented near "
        "many industrial facilities worldwide."
    ),
    "threshold": (
        "The **100 ng/L (nanograms per litre)** threshold used in this model is based on "
        "earlier EU drinking water screening values. It represents a conservative indicator "
        "of elevated contamination. The 2020 EU Drinking Water Directive set a combined "
        "PFAS limit of 100 ng/L for 20 specific compounds, and 10 ng/L for all PFAS total - "
        "so some locations above our 100 ng/L marker may already exceed regulatory limits."
    ),
    "confidence": (
        "**Confidence** in a prediction measures how close this location is to real historical "
        "measurements. HIGH confidence means there are many nearby data points - the model is "
        "interpolating from real evidence. LOW confidence or EXTRAPOLATION means the model is "
        "predicting in a region with little or no training data - treat results as indicative only."
    ),
    "shap": (
        "**SHAP values** (SHapley Additive exPlanations) tell us exactly how much each feature "
        "pushed the model's prediction up or down. A positive SHAP value for 'Distance to Airport' "
        "means being close to an airport *increased* the predicted risk. A negative value means "
        "that feature *decreased* risk. SHAP is mathematically grounded in game theory and is the "
        "gold standard for explaining machine learning predictions."
    ),
    "airport": (
        "**Airports** are a major PFAS source because of Aqueous Film Forming Foam (AFFF), a "
        "firefighting agent historically used in fire drills and emergencies. AFFF contains high "
        "concentrations of PFOS and PFOA. Soil and groundwater contamination has been documented "
        "at hundreds of airports worldwide, often extending several kilometres from the runway."
    ),
    "model": (
        "This project uses **LightGBM** - a gradient-boosted decision tree model - as the primary "
        "predictor, tuned with Bayesian optimization (Optuna). It is cross-validated using spatial "
        "block groups to prevent data leakage. The model predicts the probability that a location "
        "exceeds 100 ng/L PFAS concentration, and also estimates the likely concentration range."
    ),
    "long chain": (
        "**Long-chain PFAS** (such as PFOS, PFOA, PFNA, PFDA - 8+ carbon atoms) are the most "
        "concerning. They bioaccumulate in living organisms and persist much longer in soil and "
        "groundwater than short-chain alternatives. The EU and EPA have prioritised reducing "
        "long-chain PFAS as the most urgent regulatory concern."
    ),
}

_KB_KEYWORDS = {
    "pfas": ["pfas", "forever chemical", "fluorine", "polyfluoro"],
    "pfos": ["pfos", "perfluorooctane", "sulfonate"],
    "pfoa": ["pfoa", "perfluorooctanoic", "teflon", "non-stick"],
    "threshold": ["threshold", "100 ng", "limit", "exceedance", "regulatory"],
    "confidence": ["confidence", "uncertain", "how sure", "reliable", "trust", "extrapolation"],
    "shap": ["shap", "shapley", "why", "explain", "feature", "what drove", "what caused"],
    "airport": ["airport", "afff", "firefighting", "runway"],
    "model": ["model", "algorithm", "lightgbm", "machine learning", "ai", "how does it work", "accuracy"],
    "long chain": ["long chain", "long-chain", "persistent", "bioaccumulate", "pfna", "pfda"],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExplanationResult:
    shap_values_dict:  Dict[str, float]
    top_features:      List[Dict[str, Any]]   # [{feature, label, shap, direction, human_explanation}]
    headline:          str
    full_narrative:    str
    risk_drivers:      List[str]              # bulleted plain-English risk factors
    protective_factors: List[str]             # features pushing risk DOWN
    data_quality_note: str
    compound:          str
    exceedance_prob:   float
    concentration_ngl: float


# ---------------------------------------------------------------------------
# XAI Engine
# ---------------------------------------------------------------------------

class XAIEngine:
    """
    Wraps a trained SHAP TreeExplainer and provides natural-language explanations
    and a conversational Q&A interface.
    """

    def __init__(self, model):
        """
        Parameters
        ----------
        model : fitted LightGBM or XGBoost classifier
        """
        self.model = model
        
        # Dig for the underlying tree model if wrapped (CalibratedClassifierCV, FrozenEstimator)
        base_model = model
        if hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
            # For ensemble=False, the first one is the calibrated wrapper
            base_model = model.calibrated_classifiers_[0].estimator

        # Unpack wrappers like FrozenEstimator
        while hasattr(base_model, "estimator"):
            # If the inner model is what we want (LightGBM), stop unpacking
            if "lightgbm" in str(type(base_model)).lower():
                break
            base_model = base_model.estimator

        try:
            self.explainer = shap.TreeExplainer(base_model, feature_perturbation="tree_path_dependent")
        except Exception:
            self.explainer = shap.TreeExplainer(base_model)
        
        self._context: Optional[ExplanationResult] = None
        self._location_ctx: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Explanation generation
    # ------------------------------------------------------------------

    def explain(
        self,
        X: pd.DataFrame,
        exceedance_prob: float,
        concentration_ngl: float,
        compound: str = "PFOS",
        nearest_km: float = 0.0,
    ) -> ExplanationResult:
        """
        Generate a comprehensive plain-English explanation for a single prediction.
        """
        raw_shap = self.explainer.shap_values(X)
        if isinstance(raw_shap, list):
            local_shap = np.array(raw_shap[1][0])  # class-1 for binary classifiers
        else:
            local_shap = np.array(raw_shap[0])

        feature_names = list(X.columns)
        shap_dict     = dict(zip(feature_names, local_shap.tolist()))

        # Sort by absolute SHAP
        ranked = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)

        top_features = []
        for feat_key, shap_val in ranked[:10]:
            label    = FEATURE_LABELS.get(feat_key, feat_key.replace("_", " ").title())
            direction = "increases risk" if shap_val > 0 else "reduces risk"
            top_features.append({
                "feature":           feat_key,
                "label":             label,
                "shap":              round(float(shap_val), 4),
                "direction":         direction,
                "human_explanation": self._feature_explanation(feat_key, shap_val, X),
            })

        risk_drivers    = [f["human_explanation"] for f in top_features if f["shap"] > 0][:4]
        protective      = [f["human_explanation"] for f in top_features if f["shap"] < 0][:3]

        headline       = self._build_headline(exceedance_prob, compound)
        full_narrative = self._build_narrative(top_features, exceedance_prob,
                                               concentration_ngl, nearest_km, compound)
        dq_note        = self._data_quality_note(nearest_km)

        result = ExplanationResult(
            shap_values_dict   = shap_dict,
            top_features       = top_features,
            headline           = headline,
            full_narrative     = full_narrative,
            risk_drivers       = risk_drivers,
            protective_factors = protective,
            data_quality_note  = dq_note,
            compound           = compound,
            exceedance_prob    = exceedance_prob,
            concentration_ngl  = concentration_ngl,
        )
        self._context = result
        return result

    # ------------------------------------------------------------------
    # Conversational Q&A
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """
        Answer any user question in plain English.
        No external LLM - all responses are deterministic and grounded
        in the current prediction context + the PFAS knowledge base.
        """
        msg = user_message.strip().lower()

        # --- Knowledge-base lookup ---
        for topic, keywords in _KB_KEYWORDS.items():
            if any(kw in msg for kw in keywords):
                return PFAS_KNOWLEDGE[topic]

        # --- Context-aware answers (only if we have a result) ---
        ctx = self._context

        # Why is risk high/low?
        if re.search(r"why.*(risk|high|low|dangerous|probability|percent)", msg):
            return self._answer_why_risk(ctx)

        # What is the biggest factor?
        if re.search(r"(biggest|main|top|primary|most important|key).*(factor|driver|cause|reason)", msg):
            return self._answer_top_factor(ctx)

        # Is this safe / should I worry?
        if re.search(r"(safe|worry|concerned|danger|health|drink|use)", msg):
            return self._answer_safety(ctx)

        # What is concentration / how much PFAS?
        if re.search(r"(concentration|how much|ng|level|amount)", msg):
            return self._answer_concentration(ctx)

        # What can be done / how to reduce?
        if re.search(r"(reduce|lower|improve|clean|remediat|fix|intervention|what can)", msg):
            return self._answer_remediation()

        # How accurate / trustworthy is the model?
        if re.search(r"(accurate|trust|reliable|correct|wrong|error|sure|certain)", msg):
            return self._answer_accuracy(ctx)

        # Nearest data point / coverage?
        if re.search(r"(nearest|closest|data|nearby|close|coverage|distance)", msg):
            return self._answer_coverage(ctx)

        # Compound-specific questions
        for sub in ["pfos", "pfoa", "pfna", "pfbs", "pfhxs", "pfhpa", "pfda"]:
            if sub in msg:
                return PFAS_KNOWLEDGE.get(sub, self._answer_compound_generic(sub.upper()))

        # What does the score mean?
        if re.search(r"(score|100|scale|number|rating)", msg):
            return (
                "The **Risk Score** (0-100) is the model's estimated probability of exceeding "
                "100 ng/L PFAS - a key regulatory threshold. A score of 75 means the model "
                "believes there is a 75% chance contamination at this location exceeds that limit. "
                "Scores below 20 are considered low risk; above 60 are high concern."
            )

        # Simulation questions
        if re.search(r"(simulat|scenario|what if|preset|change|slider)", msg):
            return (
                "The **Simulation Lab** lets you explore 'what if' scenarios. You can choose "
                "presets like 'Industrial Spill' or 'Airport Expansion', or adjust individual "
                "sliders for fine-grained control. The model instantly recalculates risk under "
                "those conditions and shows you how much the probability changes, plus a "
                "Monte Carlo uncertainty band."
            )

        # Default fallback - still helpful
        if ctx:
            return (
                f"Based on the current analysis for **{ctx.compound}** at this location: "
                f"{ctx.headline} "
                f"The top driver is **{ctx.top_features[0]['label']}**. "
                "Ask me anything more specific - for example: 'Why is the risk high?', "
                "'What is PFOS?', or 'What would reduce the risk here?'"
            )

        return (
            "I'm your PFAS risk assistant. Run a location scan first, then ask me anything: "
            "'Why is the risk high?', 'What does this score mean?', "
            "'Is this safe to drink?', 'What is PFOS?', or 'How does the model work?'"
        )

    # ------------------------------------------------------------------
    # Private narrative builders
    # ------------------------------------------------------------------

    def _feature_explanation(self, feat: str, shap_val: float, X: pd.DataFrame) -> str:
        label = FEATURE_LABELS.get(feat, feat.replace("_", " ").title())
        direction = "increases" if shap_val > 0 else "reduces"
        val = float(X[feat].iloc[0]) if feat in X.columns else None

        templates = {
            "dist_to_airport_km": (
                f"The nearest airport is **{val:.1f} km** away. "
                f"Airport proximity {direction} risk because of firefighting foam (AFFF) contamination."
                if val is not None else f"{label} {direction} risk."
            ),
            "spatial_density_50km": (
                f"There are **{int(val) if val is not None else '?'} recorded PFAS measurements** within 50 km. "
                f"Higher density {direction} estimated risk - areas with more historical contamination tend to stay contaminated."
                if val is not None else f"{label} {direction} risk."
            ),
            "mean_log_value_50km": (
                f"The average PFAS level in the surrounding area {direction} contamination risk. "
                "This is a strong local signal - areas with elevated neighbours tend to be elevated themselves."
            ),
            "is_long_chain": (
                f"**{feat.replace('is_','').replace('_',' ').title()}** compounds (like PFOS and PFOA) "
                f"are more persistent. Their presence {direction} the risk estimate."
            ),
            "nearest_training_point_km": (
                f"The nearest real PFAS measurement is **{val:.0f} km** away. "
                f"This {direction} prediction confidence - farther from known data means more uncertainty."
                if val is not None else f"{label} {direction} risk."
            ),
        }
        return templates.get(feat, f"**{label}** {direction} risk at this location.")

    def _build_headline(self, prob: float, compound: str) -> str:
        if prob >= 0.65:
            return (
                f"High contamination risk detected for {compound}. "
                f"The model estimates a **{prob*100:.0f}% probability** of exceeding 100 ng/L."
            )
        if prob >= 0.35:
            return (
                f"Moderate risk for {compound}. "
                f"There is a **{prob*100:.0f}% chance** of exceeding the 100 ng/L threshold."
            )
        return (
            f"Lower risk for {compound}. "
            f"The model estimates only a **{prob*100:.0f}% probability** of elevated contamination."
        )

    def _build_narrative(self, top_features, prob, conc, nearest_km, compound):
        top = top_features[0] if top_features else None
        second = top_features[1] if len(top_features) > 1 else None

        parts = [
            f"The model analysed environmental conditions at this location and returned "
            f"an exceedance probability of **{prob*100:.1f}%** for **{compound}**. "
            f"Estimated concentration: **{conc:.1f} ng/L**.",
        ]
        if top:
            parts.append(
                f"The strongest signal is **{top['label']}**, which {top['direction'].split()[0].lower()}s risk. "
                f"{top['human_explanation']}"
            )
        if second:
            parts.append(
                f"A second meaningful influence comes from **{second['label']}**. "
                f"{second['human_explanation']}"
            )
        parts.append(self._data_quality_note(nearest_km))
        return " ".join(parts)

    @staticmethod
    def _data_quality_note(nearest_km: float) -> str:
        if nearest_km < 50:
            return (
                "Data quality: HIGH - This location is within 50 km of real measurements. "
                "Predictions are well-supported by local evidence."
            )
        if nearest_km < 200:
            return (
                "Data quality: MEDIUM - The nearest real measurement is "
                f"{nearest_km:.0f} km away. Results are indicative; local sampling is advised."
            )
        if nearest_km < 1000:
            return (
                "Data quality: LOW - This location is {:.0f} km from any training data. ".format(nearest_km) +
                "The model is extrapolating. Treat results as a rough guide only."
            )
        return (
            "Data quality: EXTRAPOLATION - This location is more than 1,000 km from any "
            "historical PFAS measurement in the dataset. Predictions should not be used for "
            "decision-making without local laboratory confirmation."
        )

    # ------------------------------------------------------------------
    # Q&A sub-handlers
    # ------------------------------------------------------------------

    def _answer_why_risk(self, ctx):
        if not ctx:
            return "Please run a location scan first, then ask again."
        top_names = [f"{f['label']}" for f in ctx.top_features[:3] if f["shap"] > 0]
        if not top_names:
            return f"Risk is low at this location. No strong contamination signals were detected for {ctx.compound}."
        return (
            f"The **{ctx.compound}** risk at this location is driven by:\n\n"
            + "\n".join(f"- {ex}" for ex in ctx.risk_drivers[:4])
            + f"\n\nOverall exceedance probability: **{ctx.exceedance_prob*100:.1f}%**. "
            + ctx.data_quality_note
        )

    def _answer_top_factor(self, ctx):
        if not ctx or not ctx.top_features:
            return "No analysis results available yet. Please run a scan first."
        top = ctx.top_features[0]
        return (
            f"The biggest factor in this prediction is **{top['label']}** "
            f"(SHAP impact: {top['shap']:+.3f}). {top['human_explanation']}"
        )

    def _answer_safety(self, ctx):
        if not ctx:
            return (
                "This tool provides risk screening, not a drinking water safety certificate. "
                "Always rely on certified laboratory analysis for public health decisions. "
                "Run a location scan first for a site-specific estimate."
            )
        if ctx.exceedance_prob >= 0.5:
            return (
                f"The model estimates a **{ctx.exceedance_prob*100:.0f}% chance** that "
                f"**{ctx.compound}** levels exceed 100 ng/L at this location. "
                "This is a screening alert - **do not use this as a safety clearance**. "
                "Commission laboratory testing before making any public health decisions. "
                f"Estimated concentration: **{ctx.concentration_ngl:.1f} ng/L**."
            )
        return (
            f"The predicted risk here is relatively low ({ctx.exceedance_prob*100:.0f}% exceedance probability). "
            "However, no AI model replaces actual water testing. "
            "For certainty, request laboratory analysis from an accredited environmental lab."
        )

    def _answer_concentration(self, ctx):
        if not ctx:
            return "Run a location scan to get a concentration estimate for a specific site."
        return (
            f"The model estimates a **{ctx.compound}** concentration of approximately "
            f"**{ctx.concentration_ngl:.1f} ng/L** at this location. "
            "This is a log-scale regression estimate. "
            "For context: the EU drinking water guideline for combined PFAS is 100 ng/L for "
            "20 priority compounds, and 10 ng/L for total PFAS sum."
        )

    @staticmethod
    def _answer_remediation():
        return (
            "Common approaches to reduce PFAS at a site:\n\n"
            "- **Source control** - eliminate the PFAS input (e.g. replace AFFF foam with PFAS-free alternatives)\n"
            "- **Granular Activated Carbon (GAC) filtration** - very effective for drinking water (~70-90% removal)\n"
            "- **Pump-and-treat** - extract contaminated groundwater and treat above ground\n"
            "- **Soil excavation** - remove heavily contaminated soil near the source\n"
            "- **In-situ stabilisation** - immobilise PFAS in soil to prevent migration\n\n"
            "Use the **Simulation Lab** to model how cleanup interventions change the risk score."
        )

    def _answer_accuracy(self, ctx):
        base = (
            "The model was validated using **spatial block cross-validation** - a rigorous method "
            "that prevents data leakage between geographically close samples. "
            "It typically achieves a **ROC-AUC of 0.85-0.92** on held-out spatial blocks. "
        )
        if ctx:
            return base + ctx.data_quality_note
        return base + " Run a location scan for site-specific confidence information."

    def _answer_coverage(self, ctx):
        if not ctx:
            return "Run a location scan first to see how close the nearest measurement is."
        return ctx.data_quality_note

    @staticmethod
    def _answer_compound_generic(compound: str) -> str:
        descriptions = {
            "PFNA": "PFNA (Perfluorononanoic Acid) is a 9-carbon long-chain PFAS. It is persistent, bioaccumulative, and linked to immune toxicity. Found in food contact materials and industrial discharge.",
            "PFDA": "PFDA (Perfluorodecanoic Acid) is a 10-carbon PFAS. It is highly persistent and accumulates in fatty tissue. Associated with liver toxicity and hormonal disruption.",
            "PFBS": "PFBS (Perfluorobutane Sulfonic Acid) is a short-chain PFAS (4 carbons) used as a replacement for PFOS. Although less bioaccumulative, it is highly mobile in groundwater.",
            "PFHXS": "PFHxS (Perfluorohexane Sulfonic Acid) is a 6-carbon sulfonate PFAS. It was used in firefighting foams and stain-resistant products. Its persistence and toxicity profile are intermediate between PFBS and PFOS.",
            "PFHPA": "PFHpA (Perfluoroheptanoic Acid) is a 7-carbon PFAS. It is considered a replacement chemical for PFOA. Limited toxicological data exists, but its environmental persistence warrants monitoring.",
        }
        return descriptions.get(compound, f"{compound} is a PFAS compound tracked in this dataset. Ask 'What is PFAS?' for general background.")
