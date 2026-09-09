"""
implementation/xai.py
=====================
Explainable AI Engine - Chat-Based Natural Language Interface
--------------------------------------------------------------
What this does:
  1. Computes SHAP values (TreeExplainer) for any prediction.
  2. Generates a rich plain-English explanation of WHAT the model
     found and WHY it reached its conclusion.
  3. Provides a conversational Q&A function that intelligently answers
     any user question about the results, the model, or PFAS science
     in plain language — no jargon, no LLM API required.
     The engine uses multi-intent classification and semantic fallback
     to handle open-ended questions gracefully.

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
# Expanded PFAS science knowledge base
# ---------------------------------------------------------------------------
PFAS_KNOWLEDGE: Dict[str, str] = {
    # ── Dashboard and map UI ──
    "map_icon": (
        "The Overview map is an offline geographic coordinate view:\n\n"
        "• Green points are sampled historical PFAS measurements from the local training dataset.\n"
        "• Orange points are detected hotspots. Darker orange means a stronger local Gi* hotspot statistic.\n"
        "• Country outlines, country labels, and major-city labels provide familiar geographic context. The horizontal axis is longitude and the vertical axis is latitude. You can zoom, pan, and hover over points.\n\n"
        "It indicates evidence coverage and detected spatial clustering; it is not a regulatory contamination map "
        "or a substitute for local laboratory sampling."
    ),
    "offline_map": (
        "The current map is different from the previous version in an important way: it works without internet access.\n\n"
        "Previous version: a Folium map inside an iframe using CartoDB basemap tiles. It could show roads, borders, "
        "and place labels, but those tiles were downloaded from the internet and could fail or appear blank offline.\n\n"
        "Current version: a Plotly map built entirely from local country boundaries, PFAS measurements, and the "
        "locally generated hotspot GeoJSON. It has no basemap-tile, geocoding, or external-font request. "
        "It supports zoom, pan, hover, country outlines, country labels, and major-city labels, but intentionally does not show street-level roads or labels."
    ),
    "project": (
        "This PFAS Geospatial Intelligence Platform is a research-screening dashboard, not a regulatory decision system.\n\n"
        "1. Data pipeline: cleans PFAS measurements and builds geographic proximity features.\n"
        "2. Scanner: estimates the chance of exceeding 100 ng/L and an expected concentration for chosen coordinates, compound, year, and medium.\n"
        "3. Analysis: uses SHAP values to show which input features increased or reduced the prediction.\n"
        "4. Simulation Lab: changes selected features to compare contamination or remediation scenarios.\n"
        "5. Explorer and Overview: summarize the locally stored measurement and hotspot data.\n\n"
        "Use it to prioritize investigation and then confirm conclusions with accredited laboratory analysis."
    ),

    # ── General PFAS science ──
    "pfas": (
        "PFAS (Per- and Polyfluoroalkyl Substances) are a family of over 12,000 man-made chemicals "
        "defined by the exceptionally strong carbon-fluorine (C-F) bond — one of the strongest in all "
        "of chemistry. This bond makes PFAS essentially indestructible by natural processes, earning them "
        "the nickname 'forever chemicals'.\n\n"
        "They are found in non-stick cookware (Teflon), food packaging, waterproof clothing (Gore-Tex), "
        "firefighting foam (AFFF), cosmetics, dental floss, and hundreds of industrial processes. "
        "PFAS have been manufactured globally since the 1950s and are now detectable in the blood of "
        "virtually every person on Earth, in Arctic ice, and in deep ocean sediments.\n\n"
        "Health concerns include: thyroid disruption, immune system suppression, elevated cholesterol, "
        "liver damage, kidney cancer, testicular cancer, and reduced vaccine effectiveness in children. "
        "The EU and US EPA have significantly tightened regulatory limits since 2020."
    ),

    # ── Individual compounds ──
    "pfos": (
        "PFOS (Perfluorooctane Sulfonic Acid) is an 8-carbon sulfonate PFAS and was the world's most "
        "widely used industrial fluorosurfactant until the 2000s. Primary uses: Scotchgard fabric protector, "
        "military/airport firefighting foam (AFFF), metal plating baths, and photographic film coatings.\n\n"
        "Health impacts: thyroid hormone disruption, immune suppression, liver damage, elevated cholesterol, "
        "and links to kidney and bladder cancer. It bioaccumulates in blood serum and has a human half-life "
        "of approximately 5.4 years.\n\n"
        "Regulation: Listed as a persistent organic pollutant (POP) under the Stockholm Convention in 2009. "
        "The EU Drinking Water Directive (2020) sets a combined limit of 0.1 µg/L for PFOS + PFOA + "
        "PFHxS + PFNA + PFDA."
    ),
    "pfoa": (
        "PFOA (Perfluorooctanoic Acid) is an 8-carbon carboxylate PFAS, best known as the 'Teflon chemical'. "
        "It was manufactured by DuPont (now Chemours) and used as a processing aid in the production of "
        "polytetrafluoroethylene (PTFE). Contamination near DuPont's Washington Works plant in West Virginia "
        "exposed tens of thousands of residents for decades.\n\n"
        "Health impacts: classified as a possible/probable human carcinogen. Linked to kidney cancer, "
        "testicular cancer, thyroid disease, ulcerative colitis, pregnancy-induced hypertension, and elevated "
        "cholesterol. Blood half-life: approximately 3.5 years.\n\n"
        "Regulation: voluntarily phased out by US manufacturers by 2015. The EU includes PFOA in Annex I "
        "of the Stockholm Convention. US EPA MCL goal: zero (as of 2023 proposed rules)."
    ),
    "pfhxs": (
        "PFHxS (Perfluorohexane Sulfonic Acid) is a 6-carbon sulfonate PFAS — a shorter-chain analogue of "
        "PFOS. It was introduced as a PFOS replacement and is commonly found in firefighting foams, stain "
        "repellents, and metal plating.\n\n"
        "Health impacts: similar endocrine disruption to PFOS, but generally considered less bioaccumulative "
        "due to its shorter chain length. Human half-life: approximately 7–8 years, making it MORE persistent "
        "in blood than PFOS. Linked to thyroid hormone disruption.\n\n"
        "Regulation: included in the EU Drinking Water Directive 2020 priority list. Increasingly regulated "
        "alongside PFOS and PFOA."
    ),
    "pfna": (
        "PFNA (Perfluorononanoic Acid) is a 9-carbon long-chain PFAS carboxylate. Found in food contact "
        "materials, floor waxes, and industrial effluents. It is one of the most potent PFAS in terms of "
        "bioaccumulation.\n\n"
        "Health impacts: immune toxicity (reduces vaccine response in children), developmental toxicity, "
        "liver damage, and endocrine disruption. Detected in human breast milk, blood, and liver tissue.\n\n"
        "Regulation: classified as a persistent bioaccumulative toxic (PBT) substance under EU REACH. "
        "Included in the EU Drinking Water Directive priority group."
    ),
    "pfda": (
        "PFDA (Perfluorodecanoic Acid) is a 10-carbon long-chain PFAS — one of the heaviest commonly "
        "monitored PFAS compounds. Extremely persistent in both the environment and living organisms.\n\n"
        "Health impacts: strongly linked to liver toxicity and fatty liver disease, hormonal (oestrogen) "
        "disruption, and developmental effects in animal studies. Accumulates in fatty tissue over time.\n\n"
        "Due to its large molecular size, PFDA tends to bind more tightly to soil particles and moves "
        "more slowly through aquifers than shorter-chain compounds."
    ),
    "pfbs": (
        "PFBS (Perfluorobutane Sulfonic Acid) is a short-chain 4-carbon PFAS, introduced as a 'safer' "
        "replacement for PFOS after the latter's phase-out. Used in carpet treatments, waterproof coatings, "
        "and some firefighting foams.\n\n"
        "Health impacts: while less bioaccumulative than PFOS, PFBS is highly mobile in groundwater "
        "due to its low soil binding affinity. Detected increasingly in drinking water wells downhill from "
        "industrial sites. US EPA classified it as having liver toxicity and developmental effects.\n\n"
        "The assumption that short-chain PFAS are 'safe' replacements has been increasingly challenged by "
        "new toxicology data."
    ),
    "pfhpa": (
        "PFHpA (Perfluoroheptanoic Acid) is a 7-carbon PFAS carboxylate, positioned between the short-chain "
        "(≤6C) and long-chain (≥8C) categories. It is increasingly used as a replacement for PFOA.\n\n"
        "Toxicological data is more limited than for longer-chain compounds, but animal studies show "
        "liver toxicity and developmental effects. It is more mobile in the environment than PFOA "
        "and can travel further through soil and groundwater systems."
    ),

    # ── Technical / model concepts ──
    "threshold": (
        "Regulatory thresholds for PFAS vary by jurisdiction and compound:\n\n"
        "• EU Drinking Water Directive (2020): 100 ng/L combined for 20 specific PFAS, and 10 ng/L "
        "for sum of all PFAS detected\n"
        "• US EPA MCL (2024): 4 ng/L for PFOS and PFOA individually; 10 ng/L for PFNA, PFHxS, and HFPO-DA\n"
        "• UK DWI: 100 ng/L for individual PFAS and 500 ng/L total PFAS (under review)\n\n"
        "This model uses 100 ng/L as its primary exceedance threshold — a conservative value aligned "
        "with earlier EU guidance. Many locations flagged as 'high risk' by this model may already "
        "be in regulatory breach under the newer 10 ng/L EU standard."
    ),
    "confidence": (
        "Prediction confidence in this model reflects how close the location is to real historical "
        "measurements in the training dataset.\n\n"
        "• HIGH confidence (< 50 km): The prediction is interpolating from nearby real measurements. "
        "Results are well-supported.\n"
        "• MEDIUM confidence (50–200 km): Some spatial extrapolation is occurring. Results are indicative; "
        "local sampling is advisable.\n"
        "• LOW confidence (200–1000 km): The model is extrapolating significantly. Treat as a rough guide.\n"
        "• EXTRAPOLATION (> 1000 km): Far from any training data. Results should not be used for decisions "
        "without laboratory confirmation.\n\n"
        "Confidence does NOT mean the prediction is wrong — it indicates how much real-world evidence "
        "supports the estimate."
    ),
    "shap": (
        "SHAP (SHapley Additive exPlanations) values quantify exactly how much each input feature "
        "pushed the model's prediction up or down from the average.\n\n"
        "• A POSITIVE SHAP value: that feature increased the predicted risk at this location\n"
        "• A NEGATIVE SHAP value: that feature decreased the predicted risk\n"
        "• Magnitude: the larger the absolute value, the stronger the influence\n\n"
        "SHAP is grounded in cooperative game theory (Shapley values). Unlike simpler methods, it "
        "guarantees consistency — the feature contributions always sum to the total model output. "
        "This means you can trust the orange bars in the chart to be the actual reasons the model "
        "reached its conclusion, not a simplified approximation."
    ),
    "airport": (
        "Airports are a major PFAS contamination source because of Aqueous Film Forming Foam (AFFF), "
        "a firefighting agent used in fire drills, emergency training, and real incidents. AFFF "
        "historically contained high concentrations of PFOS and PFOA.\n\n"
        "Contamination patterns: PFAS from AFFF leach into soil and groundwater, often extending "
        "2–10 km from the runway threshold. Plumes can travel significant distances in certain "
        "aquifer conditions. Hundreds of military and civilian airports worldwide have confirmed "
        "PFAS groundwater contamination.\n\n"
        "In this model, distance to the nearest airport (dist_to_airport_km) is one of the most "
        "influential features for sites near major airports."
    ),
    "model": (
        "This dashboard uses LightGBM — a gradient-boosted decision tree algorithm — as its primary "
        "predictor. The model was:\n\n"
        "• Tuned with Bayesian hyperparameter optimisation (Optuna) over 150+ trials\n"
        "• Validated using spatial block cross-validation (groups by geography to prevent data leakage)\n"
        "• Calibrated using Platt sigmoid scaling so that the predicted probabilities are reliable (e.g. 70% "
        "probability really does correspond to ~70% exceedance rate in held-out data)\n"
        "• Trained on a curated 'golden' dataset of real PFAS measurements across Europe\n\n"
        "Typical performance: ROC-AUC of 0.85–0.92 on held-out spatial blocks. The model predicts "
        "exceedance probability (> 100 ng/L) and also estimates the likely concentration range."
    ),
    "long chain": (
        "Long-chain PFAS (8+ carbon atoms) are the most regulated and most concerning:\n\n"
        "Long-chain carboxylates (PFCAs): PFOA (8C), PFNA (9C), PFDA (10C), PFUnDA (11C)\n"
        "Long-chain sulfonates (PFSAs): PFOS (8C), PFHxS (6C — borderline)\n\n"
        "Why they're problematic:\n"
        "• They bioaccumulate in living organisms (concentrate up the food chain)\n"
        "• They persist for decades in soil and groundwater\n"
        "• They have longer human blood half-lives (3–8+ years)\n"
        "• They are associated with cancer, immune dysfunction, and developmental toxicity\n\n"
        "Short-chain replacements (≤6C) are more mobile in groundwater and may not be 'safer' — "
        "they're simply less studied."
    ),
    "general": (
        "The GENERAL (Total) compound selection provides an aggregate risk assessment across all "
        "monitored PFAS compounds (PFOS, PFOA, PFHxS, PFNA, PFDA, PFHpA, PFBS).\n\n"
        "The Estimated Concentration is the sum of all individual predictions, while the Risk "
        "Probability reflects the likelihood that at least one compound exceeds 100 ng/L. "
        "This is the most comprehensive screening indicator for a site's overall chemical burden "
        "and is the recommended starting point for any new site assessment."
    ),
    "health": (
        "PFAS health effects are well-documented in epidemiological and animal studies:\n\n"
        "Immune system: Reduced vaccine efficacy in children is one of the most robust findings. "
        "PFAS suppress antibody response to vaccines at blood levels now common in many populations.\n\n"
        "Endocrine: Thyroid hormone disruption, altered sex hormone levels, and delayed puberty.\n\n"
        "Metabolic: Elevated total cholesterol and LDL (bad cholesterol); associations with "
        "obesity and insulin resistance.\n\n"
        "Cancer: Kidney cancer and testicular cancer have the strongest evidence for PFOA and PFOS. "
        "The IARC classifies PFOA as Group 1 (carcinogenic to humans, 2023).\n\n"
        "Reproductive: Reduced fertility, pregnancy complications, and reduced birth weight.\n\n"
        "Liver: Non-alcoholic fatty liver disease (NAFLD) at elevated exposures.\n\n"
        "No safe level of PFAS exposure has been established — the science points to harm "
        "at concentrations previously considered safe."
    ),
    "military": (
        "Military bases are significant PFAS contamination sources, primarily from the extensive "
        "use of AFFF firefighting foam during training exercises and emergency responses.\n\n"
        "AFFF has been used at military airfields, fire training areas, and crash rescue facilities "
        "for decades. Unlike civilian airports, military fire training historically used far higher "
        "foam volumes, often with minimal containment.\n\n"
        "In the US, over 700 military installations have confirmed PFAS groundwater contamination. "
        "Similar patterns exist near NATO bases across Europe. The model accounts for proximity to "
        "military bases (dist_to_military_km) as a risk factor."
    ),
    "wastewater": (
        "Wastewater treatment plants (WWTPs) are both a sink and a secondary source of PFAS contamination:\n\n"
        "As a sink: PFAS from industrial discharges, households, and stormwater enter WWTPs. "
        "Standard treatment processes (activated sludge, trickling filters) are largely ineffective "
        "at removing PFAS — removal rates are typically 0–30% for most compounds.\n\n"
        "As a source: PFAS that adsorb to sewage sludge remain in biosolids. When biosolids are "
        "applied to agricultural land as fertiliser, PFAS leach into soil and groundwater. "
        "WWTP effluents discharge PFAS directly into rivers.\n\n"
        "Distance to the nearest WWTP (dist_to_wastewater_km) is included in this model as a "
        "risk factor, particularly for surface water and soil predictions."
    ),
    "soil": (
        "PFAS behaviour in soil depends on compound properties and soil characteristics:\n\n"
        "Soil binding: Long-chain PFAS bind strongly to organic matter and clay particles. "
        "Short-chain PFAS are more weakly bound and more likely to leach into groundwater.\n\n"
        "Soil permeability: High permeability (sandy soils) allows faster PFAS migration to "
        "groundwater. Low permeability (clay soils) can trap PFAS near the surface but also "
        "concentrate them.\n\n"
        "Sources in soil: AFFF application, biosolid amendments, atmospheric deposition, "
        "industrial spills, and landfill leachate.\n\n"
        "The model includes soil_permeability and land_use_class as features that "
        "influence predictions for soil and groundwater media types."
    ),
    "elevation": (
        "Elevation influences PFAS contamination patterns in several ways:\n\n"
        "• Higher elevation areas tend to be uphill of industrial zones, reducing downstream "
        "contamination exposure\n"
        "• Low-lying areas (valleys, flood plains) accumulate PFAS from upstream sources via "
        "surface water flow and sediment deposition\n"
        "• Elevation affects rainfall patterns (more precipitation at altitude can increase "
        "PFAS transport via stormwater)\n"
        "• Coastal and estuarine areas at near-zero elevation often show elevated PFAS from "
        "port activities and tidal mixing\n\n"
        "In this model, elevation_m is a moderate predictor, working in combination with "
        "precipitation and land use."
    ),
    "rainfall": (
        "Annual precipitation (annual_precip_mm) influences PFAS transport in the environment:\n\n"
        "High rainfall areas: PFAS are flushed more rapidly through soil into groundwater and "
        "surface water. Higher runoff volumes also transport PFAS from contaminated land to "
        "water bodies.\n\n"
        "Low rainfall areas: PFAS may concentrate at depth or accumulate in surface soils "
        "without sufficient water to dilute or transport them.\n\n"
        "Seasonal effects: Peak contamination in streams often occurs during the first flush "
        "after dry periods, when accumulated PFAS wash from catchment surfaces."
    ),
    "population": (
        "Population density (population_density_log in the model) is a proxy for several "
        "PFAS exposure pathways:\n\n"
        "• More people → more PFAS-containing consumer products → more PFAS entering "
        "wastewater streams\n"
        "• Urban areas have higher road surface PFAS loadings from tyre and brake dust "
        "(some PFAS used in fluoropolymer lubricants)\n"
        "• Dense urban areas typically have more industrial zones, commercial activities, "
        "and vehicle-related contamination\n"
        "• Population density also increases the public health significance of any PFAS "
        "contamination detected"
    ),
    "remediation": (
        "PFAS remediation is an active research field. Current approaches range from "
        "containment to destruction:\n\n"
        "DRINKING WATER TREATMENT:\n"
        "• Granular Activated Carbon (GAC): 70–95% PFAS removal; well-established\n"
        "• Ion Exchange Resins (IEX): single-use or regenerable; >99% removal possible\n"
        "• Reverse Osmosis (RO): >95% removal but generates concentrated PFAS waste\n\n"
        "GROUNDWATER & SOIL:\n"
        "• Pump-and-treat: extract and treat contaminated groundwater above ground\n"
        "• Soil excavation: remove heavily contaminated near-source soil\n"
        "• In-situ stabilisation: immobilise PFAS using activated carbon amendment\n"
        "• Monitored Natural Attenuation (MNA): for low-level, stable plumes\n\n"
        "DESTRUCTION TECHNOLOGIES (emerging):\n"
        "• Supercritical water oxidation (SCWO): breaks C-F bonds at very high temp/pressure\n"
        "• Electrochemical oxidation: electrically induced PFAS mineralisation\n"
        "• Sonochemical treatment: ultrasound-driven PFAS decomposition\n\n"
        "Use the Simulation Lab to model how cleanup interventions change the risk score."
    ),
    "score": (
        "The Risk Score (0–100%) is the model's estimated probability that PFAS concentration "
        "at this location exceeds the 100 ng/L screening threshold.\n\n"
        "Risk level interpretation:\n"
        "• 0–20%: Safe Zone — very low contamination signals\n"
        "• 20–40%: Watch Zone — background contamination; routine monitoring advisable\n"
        "• 40–60%: Caution Zone — elevated risk; detailed investigation warranted\n"
        "• 60–80%: High Alert — high contamination probability; regulatory action recommended\n"
        "• 80–100%: Critical Alert — extreme risk; immediate intervention may be required\n\n"
        "The score is a probability estimate, not a measured concentration. Always commission "
        "laboratory testing before making public health or regulatory decisions."
    ),
    "simulation": (
        "The Simulation Lab lets researchers explore counterfactual 'what if' scenarios:\n\n"
        "• Select multiple scenario presets (e.g. Industrial Spill + Airport Expansion) to "
        "compare their combined effects side-by-side\n"
        "• Fine-tune parameters with sliders: industrial intensity, airport distance, "
        "and cleanup intervention efficiency\n"
        "• The comparison bar chart shows risk scores for each scenario vs baseline\n"
        "• The outcome narrative explains the probability change in plain English\n\n"
        "All simulations use the feature vector from the last Scanner run as the baseline. "
        "Run a Scanner prediction first, then switch to Simulation to explore scenarios."
    ),
    "accuracy": (
        "Model performance metrics:\n\n"
        "• ROC-AUC: typically 0.85–0.92 on held-out spatial blocks — meaning the model "
        "correctly ranks contaminated vs clean sites 85–92% of the time\n"
        "• Spatial block cross-validation: training/test splits by geographic region to "
        "prevent data leakage between nearby correlated samples\n"
        "• Calibration: Platt sigmoid scaling ensures predicted probabilities are reliable "
        "(a 70% prediction corresponds to ~70% real-world exceedance rate)\n"
        "• Concentration RMSE: logarithmic error of ~0.45 log-units\n\n"
        "Limitations:\n"
        "• Performance degrades in areas far from training data (see confidence levels)\n"
        "• The model cannot detect point-source contamination events that haven't been "
        "sampled in the training data\n"
        "• Predictions are probabilistic — individual site outcomes may differ"
    ),
}

_KB_KEYWORDS = {
    "pfas": ["pfas", "pfa", "forever chemical", "fluorine", "polyfluoro", "chemicals", "pfaa", "fluorinated"],
    "map_icon": [
        "home icon", "map pin", "green dot", "circle marker", "cluster", "bubble",
        "heatmap", "legend", "icons mean", "symbols", "map mean", "pin on the map",
        "what does the map show",
    ],
    "offline_map": [
        "offline map", "map offline", "without internet", "no internet", "map different",
        "previous map", "old map", "map tiles", "basemap", "cartodb",
    ],
    "project": [
        "this project", "this platform", "this dashboard", "this application", "this app",
        "project do", "platform do", "dashboard do", "project work", "platform work",
        "explain the project", "explain this dashboard", "about the project",
    ],
    "pfos": ["pfos", "perfluorooctane sulfon", "sulfonate", "firefighting foam", "afff"],
    "pfoa": ["pfoa", "perfluorooctanoic", "teflon", "non-stick", "carcinogen", "cancer", "dupont"],
    "pfhxs": ["pfhxs", "pfhxs", "perfluorohexane"],
    "pfna": ["pfna", "perfluorononan", "nonanoic"],
    "pfda": ["pfda", "perfluorodecan", "decanoic"],
    "pfbs": ["pfbs", "perfluorobutane", "short chain replacement"],
    "pfhpa": ["pfhpa", "pfhpa", "perfluoroheptanoic", "heptanoic"],
    "threshold": ["threshold", "100 ng", "10 ng", "limit", "exceedance", "regulatory", "guideline", "standard"],
    "confidence": ["confidence", "uncertain", "how sure", "reliable", "trust", "extrapolation", "how confident", "how close"],
    "shap": ["shap", "shapley", "feature importance", "what drove", "what caused", "attribution", "contribution"],
    "airport": ["airport", "afff", "firefighting", "runway", "aviation"],
    "model": ["model", "algorithm", "lightgbm", "machine learning", "ai", "how does it work", "gradient boost"],
    "long chain": ["long chain", "long-chain", "persistent", "bioaccumulate", "pfna", "pfda", "8 carbon"],
    "general": ["general", "total pfas", "all compounds", "aggregate", "combined score", "total score"],
    "health": ["health", "cancer", "thyroid", "immune", "vaccine", "blood", "toxic", "harmf", "disease", "safe to eat", "safe to swim", "effect on", "impact on"],
    "military": ["military", "base", "army", "navy", "airforce", "defense", "defence", "nato", "miltary"],
    "wastewater": ["wastewater", "sewage", "wwtp", "treatment plant", "biosolid", "sludge", "sewage"],
    "soil": ["soil", "permeab", "sediment", "land use", "ground", "subsurface", "clay", "sandy"],
    "elevation": ["elevation", "altitude", "hill", "valley", "topograph"],
    "rainfall": ["rainfall", "precipitation", "rain", "climate", "weather", "annual precip"],
    "population": ["population", "urban", "density", "city", "people", "inhabited"],
    "remediation": ["remediat", "cleanup", "clean up", "remove", "treatment", "gac", "carbon filter", "ion exchange", "reverse osmosis", "pump and treat", "reduce risk"],
    "score": ["score", "100", "what does the score mean", "risk score", "percentage", "rating", "scale"],
    "simulation": ["simulat", "scenario", "what if", "preset", "lab", "playground"],
    "accuracy": ["accurate", "trust", "reliable", "correct", "wrong", "error", "sure", "certain", "validation", "performance", "auc"],
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
    and a conversational Q&A interface that handles any question intelligently.
    """

    def __init__(self, model):
        self.model = model

        # Dig for the underlying tree model if wrapped (CalibratedClassifierCV, FrozenEstimator)
        base_model = model
        if hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
            base_model = model.calibrated_classifiers_[0].estimator

        # Unpack wrappers like FrozenEstimator
        while hasattr(base_model, "estimator"):
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
        """Generate a comprehensive plain-English explanation for a single prediction."""
        raw_shap = self.explainer.shap_values(X)
        if isinstance(raw_shap, list):
            local_shap = np.array(raw_shap[1][0])
        else:
            local_shap = np.array(raw_shap[0])

        feature_names = list(X.columns)
        shap_dict     = dict(zip(feature_names, local_shap.tolist()))

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
    # Conversational Q&A — handles any question intelligently
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """
        Answer any user question in plain English.
        Uses multi-intent classification with semantic fallback.
        """
        msg = user_message.strip().lower()
        ctx = self._context

        # Resolve specific multi-factor and action questions before shorter
        # overlapping phrases such as "factor" and "should I".
        if re.search(r"(all|every|full|complete).*(factor|driver|reason|influence|breakdown)", msg):
            return self._answer_all_factors(ctx)

        if re.search(r"(next step|what should i do|what do i do next|recommend.*action|action.*recommend|what action)", msg):
            return self._answer_next_steps(ctx)

        if re.search(r"(map.*(different|previous|old|offline|internet)|(?:different|previous|old|offline|internet).*map)", msg):
            return PFAS_KNOWLEDGE["offline_map"]

        # Site-specific intents come before broad knowledge-base keywords.
        # This prevents words such as "health", "model", or "safe" from
        # masking a more useful answer about the active scanner result.
        if re.search(r"why.*(risk|high|low|dangerous|probability|percent|elevated|so bad)", msg):
            return self._answer_why_risk(ctx)

        if re.search(r"(biggest|main|top|primary|most important|key|dominant|strongest).*(factor|driver|cause|reason|predictor|influence)", msg):
            return self._answer_top_factor(ctx)

        if re.search(r"(safe|worry|concerned|danger|drink|swim|eat|use|should i|can i|okay to|risk to)", msg):
            return self._answer_safety(ctx)

        if re.search(r"(accurate|trust|reliable|correct|wrong|error|sure|certain|validation|performance|how good)", msg):
            return self._answer_accuracy(ctx)

        if re.search(r"(nearest|closest|data|nearby|close|coverage|distance|training)", msg):
            return self._answer_coverage(ctx)

        if re.search(r"(concentration|how much|ng|level|amount|estimate|detected|measure)", msg):
            return self._answer_concentration(ctx)

        if re.search(r"(protective|what is protecting|what reduce|what lower|what help|what is good|positive)", msg):
            return self._answer_protective_factors(ctx)

        # ── Step 1: Knowledge-base lookup (exact keyword match) ──
        for topic, keywords in _KB_KEYWORDS.items():
            if any(kw in msg for kw in keywords):
                return PFAS_KNOWLEDGE[topic]

        # ── Step 2: Context-aware answers ──
        if re.search(r"why.*(risk|high|low|dangerous|probability|percent|elevated|so bad)", msg):
            return self._answer_why_risk(ctx)

        if re.search(r"(biggest|main|top|primary|most important|key|dominant|strongest).*(factor|driver|cause|reason|predictor|influence)", msg):
            return self._answer_top_factor(ctx)

        if re.search(r"(safe|worry|concerned|danger|drink|swim|eat|use|should i|can i|okay to|risk to)", msg):
            return self._answer_safety(ctx)

        if re.search(r"(concentration|how much|ng|level|amount|estimate|detected|measure)", msg):
            return self._answer_concentration(ctx)

        if re.search(r"(reduce|lower|improve|clean|remediat|fix|intervention|what can|mitigate|decrease|help)", msg):
            return PFAS_KNOWLEDGE["remediation"]

        if re.search(r"(accurate|trust|reliable|correct|wrong|error|sure|certain|validation|performance|how good)", msg):
            return self._answer_accuracy(ctx)

        if re.search(r"(nearest|closest|data|nearby|close|coverage|distance|training)", msg):
            return self._answer_coverage(ctx)

        if re.search(r"(compound|chemical|substance|which pfas|type of pfas|what pfas)", msg):
            if ctx:
                return self._answer_compound_context(ctx)
            return PFAS_KNOWLEDGE["pfas"]

        if re.search(r"(protective|what is protecting|what reduce|what lower|what help|what is good|positive)", msg):
            return self._answer_protective_factors(ctx)

        if re.search(r"(all factor|all driver|full list|every factor|list the|breakdown|full breakdown|all reason)", msg):
            return self._answer_all_factors(ctx)

        if re.search(r"(compare|versus|vs|difference|baseline|scenario|change)", msg):
            return PFAS_KNOWLEDGE["simulation"]

        if re.search(r"(what is this site|this location|this place|this area|describe this|tell me about this)", msg):
            return self._answer_site_overview(ctx)

        if re.search(r"(next step|what should i do|recommend|action|suggestion|investigate)", msg):
            return self._answer_next_steps(ctx)

        # ── Step 3: Generic "what does / explain / describe / how" ──
        if re.search(r"(what does|tell me about|explain|describe|how|what is|what are|meaning of)", msg):
            # Try to extract a compound name
            for compound_kw, topic in [
                ("pfos", "pfos"), ("pfoa", "pfoa"), ("pfhxs", "pfhxs"),
                ("pfna", "pfna"), ("pfda", "pfda"), ("pfbs", "pfbs"), ("pfhpa", "pfhpa"),
            ]:
                if compound_kw in msg:
                    return PFAS_KNOWLEDGE[topic]
            if ctx:
                return self._answer_open_question(msg, ctx)
            return PFAS_KNOWLEDGE["pfas"]

        # ── Step 4: Context-aware general fallback ──
        if ctx:
            return (
                f"Based on the active site analysis ({ctx.compound}):\n\n"
                f"{ctx.headline}\n\n"
                f"Top risk driver: {ctx.top_features[0]['label']} (SHAP: {ctx.top_features[0]['shap']:+.3f}).\n\n"
                f"{ctx.data_quality_note}\n\n"
                "Try asking: 'Why is the risk high?', 'What are all the factors?', "
                "'Is this safe to drink?', 'How can I reduce the risk?', or about any specific "
                "PFAS compound (e.g. 'What is PFOS?')."
            )

        # ── Step 5: Welcome fallback ──
        return (
            "Hello! I'm your PFAS risk assistant. I can help you with:\n\n"
            "• Site-specific analysis (run a Scanner scan first)\n"
            "• PFAS compound explanations (PFOS, PFOA, PFHxS, PFNA, PFDA, PFBS, PFHpA)\n"
            "• Health effects, regulatory thresholds, and exposure risks\n"
            "• How the model works, SHAP values, and prediction confidence\n"
            "• Remediation and cleanup options\n"
            "• Scenario simulation guidance\n\n"
            "Try asking: 'What is PFOA?', 'What are the health effects?', "
            "'How does the model work?', or 'What is the threshold?' — or run a site "
            "scan first and ask 'Why is the risk high?'"
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
                f"The nearest airport is {val:.1f} km away. "
                f"Airport proximity {direction} risk because of firefighting foam (AFFF) contamination."
                if val is not None else f"{label} {direction} risk."
            ),
            "spatial_density_50km": (
                f"There are {int(val) if val is not None else '?'} recorded PFAS measurements within 50 km. "
                f"Higher density {direction} estimated risk — areas with more historical contamination tend to stay contaminated."
                if val is not None else f"{label} {direction} risk."
            ),
            "mean_log_value_50km": (
                f"The average PFAS level in the surrounding area {direction} contamination risk. "
                "This is a strong local signal — areas with elevated neighbours tend to be elevated themselves."
            ),
            "is_long_chain": (
                f"{'Long' if val and val > 0.5 else 'Short'}-chain compounds like "
                f"{'PFOS and PFOA' if val and val > 0.5 else 'PFBS and PFHpA'} "
                f"{direction} the risk estimate due to their {'high' if val and val > 0.5 else 'lower'} persistence."
            ),
            "nearest_training_point_km": (
                f"The nearest real PFAS measurement is {val:.0f} km away. "
                f"This {direction} prediction confidence — farther from known data means more uncertainty."
                if val is not None else f"{label} {direction} risk."
            ),
            "dist_to_military_km": (
                f"The nearest military base is {val:.1f} km away. "
                f"Military facilities {direction} risk due to extensive AFFF usage during training exercises."
                if val is not None else f"{label} {direction} risk."
            ),
            "dist_to_wastewater_km": (
                f"The nearest wastewater treatment plant is {val:.1f} km away. "
                f"Wastewater plants {direction} risk as they concentrate PFAS from effluents and biosolids."
                if val is not None else f"{label} {direction} risk."
            ),
            "soil_permeability": (
                f"Soil permeability at this location {direction} risk — "
                f"{'permeable soils allow PFAS to leach more readily into groundwater' if shap_val > 0 else 'less permeable soils slow PFAS migration'}."
            ),
            "population_density_log": (
                f"Population density {direction} risk, reflecting "
                f"{'higher PFAS loading from consumer products and industrial activities' if shap_val > 0 else 'lower industrial and consumer PFAS inputs'}."
            ),
            "annual_precip_mm": (
                f"Annual rainfall {direction} PFAS transport risk — "
                f"{'higher rainfall flushes more PFAS through soil into waterways' if shap_val > 0 else 'lower rainfall reduces PFAS leaching and transport'}."
            ),
        }
        return templates.get(feat, f"{label} {direction} risk at this location.")

    def _build_headline(self, prob: float, compound: str) -> str:
        display_sub = "Total PFAS" if compound == "GENERAL" else compound
        if prob >= 0.65:
            return (
                f"High contamination risk detected for {display_sub}. "
                f"The model estimates a {prob*100:.0f}% probability of exceeding 100 ng/L."
            )
        if prob >= 0.35:
            return (
                f"Moderate risk for {display_sub}. "
                f"There is a {prob*100:.0f}% chance of exceeding the 100 ng/L threshold."
            )
        return (
            f"Lower risk for {display_sub}. "
            f"The model estimates only a {prob*100:.0f}% probability of elevated contamination."
        )

    def _build_narrative(self, top_features, prob, conc, nearest_km, compound):
        top = top_features[0] if top_features else None
        second = top_features[1] if len(top_features) > 1 else None

        parts = [
            f"The model analysed environmental conditions at this location and returned "
            f"an exceedance probability of {prob*100:.1f}% for {compound}. "
            f"Estimated concentration: {conc:.1f} ng/L.",
        ]
        if top:
            parts.append(
                f"The strongest signal is {top['label']}, which {top['direction'].split()[0].lower()}s risk. "
                f"{top['human_explanation']}"
            )
        if second:
            parts.append(
                f"A second meaningful influence comes from {second['label']}. "
                f"{second['human_explanation']}"
            )
        parts.append(self._data_quality_note(nearest_km))
        return " ".join(parts)

    @staticmethod
    def _data_quality_note(nearest_km: float) -> str:
        if nearest_km < 50:
            return (
                "Data quality: HIGH — This location is within 50 km of real measurements. "
                "Predictions are well-supported by local evidence."
            )
        if nearest_km < 200:
            return (
                "Data quality: MEDIUM — The nearest real measurement is "
                f"{nearest_km:.0f} km away. Results are indicative; local sampling is advised."
            )
        if nearest_km < 1000:
            return (
                "Data quality: LOW — This location is {:.0f} km from any training data. ".format(nearest_km) +
                "The model is extrapolating. Treat results as a rough guide only."
            )
        return (
            "Data quality: EXTRAPOLATION — This location is more than 1,000 km from any "
            "historical PFAS measurement in the dataset. Predictions should not be used for "
            "decision-making without local laboratory confirmation."
        )

    # ------------------------------------------------------------------
    # Q&A sub-handlers
    # ------------------------------------------------------------------

    def _answer_why_risk(self, ctx):
        if not ctx:
            return (
                "Please run a location scan first, then ask again. "
                "Once a scan is complete, I can explain exactly which environmental factors "
                "drove the risk prediction at your chosen site."
            )
        top_names = [f"{f['label']}" for f in ctx.top_features[:3] if f["shap"] > 0]
        if not top_names:
            return (
                f"Risk is low at this location. No strong contamination signals were detected "
                f"for {ctx.compound}. The main protective factors are:\n\n"
                + "\n".join(f"• {ex}" for ex in ctx.protective_factors[:3])
            )
        return (
            f"The {ctx.compound} risk at this location is driven by:\n\n"
            + "\n".join(f"• {ex}" for ex in ctx.risk_drivers[:4])
            + f"\n\nOverall exceedance probability: {ctx.exceedance_prob*100:.1f}%.\n"
            + ctx.data_quality_note
        )

    def _answer_all_factors(self, ctx):
        if not ctx:
            return "Run a location scan first. I'll then show you all factors ranked by their influence."
        lines = []
        for f in ctx.top_features[:8]:
            arrow = "▲" if f["shap"] > 0 else "▼"
            lines.append(f"{arrow} {f['label']} (SHAP: {f['shap']:+.3f}) — {f['direction']}")
        return (
            f"All influential factors for the {ctx.compound} prediction "
            f"(ranked by impact magnitude):\n\n"
            + "\n".join(lines)
            + f"\n\nExceedance probability: {ctx.exceedance_prob*100:.1f}%\n"
            + ctx.data_quality_note
        )

    def _answer_protective_factors(self, ctx):
        if not ctx:
            return "Run a location scan first. I'll then show you what factors are reducing risk."
        if not ctx.protective_factors:
            return (
                f"No significant protective factors were identified for this location. "
                f"All major features are neutral or contributing positively to the {ctx.exceedance_prob*100:.1f}% risk estimate."
            )
        return (
            f"Protective factors reducing risk at this location:\n\n"
            + "\n".join(f"• {p}" for p in ctx.protective_factors)
            + f"\n\nDespite these, the overall exceedance probability is {ctx.exceedance_prob*100:.1f}%."
        )

    def _answer_open_question(self, user_message: str, ctx):
        if not ctx:
            return "Run a location scan first so I can answer with site-specific context."

        msg = user_message.lower()
        if any(term in msg for term in ["risk", "probability", "chance", "score", "exceed"]):
            level = "high concern" if ctx.exceedance_prob >= 0.65 else "moderate concern" if ctx.exceedance_prob >= 0.35 else "low concern"
            return (
                f"The current exceedance probability for {ctx.compound} is {ctx.exceedance_prob * 100:.1f}%, "
                f"placing this site in the '{level}' category.\n\n"
                f"In practical terms: there is a {ctx.exceedance_prob * 100:.0f}% chance that "
                f"PFAS levels at this location exceed the 100 ng/L regulatory screening threshold.\n\n"
                f"{ctx.data_quality_note}"
            )

        if any(term in msg for term in ["factor", "driver", "why", "cause", "reason", "what"]):
            return self._answer_top_factor(ctx)

        if any(term in msg for term in ["protect", "reduce", "mitig", "cleanup", "remedi", "lower"]):
            return PFAS_KNOWLEDGE["remediation"]

        if any(term in msg for term in ["concentration", "amount", "ng", "level", "estimate"]):
            return self._answer_concentration(ctx)

        return (
            f"From the active site context: {ctx.headline} "
            f"The strongest predictor is {ctx.top_features[0]['label']} (SHAP: {ctx.top_features[0]['shap']:+.3f}). "
            f"{ctx.data_quality_note}"
        )

    def _answer_site_overview(self, ctx):
        if not ctx:
            return "No site scan has been run yet. Please use the Scanner page to analyse a location first."
        risk_label = "high" if ctx.exceedance_prob >= 0.65 else "moderate" if ctx.exceedance_prob >= 0.35 else "low"
        return (
            f"Site overview for the active scan ({ctx.compound}):\n\n"
            f"• Risk level: {risk_label.upper()} — {ctx.exceedance_prob*100:.1f}% exceedance probability\n"
            f"• Estimated concentration: {ctx.concentration_ngl:.1f} ng/L\n"
            f"• Top risk driver: {ctx.top_features[0]['label']}\n"
            f"• {ctx.data_quality_note}\n\n"
            f"Full summary: {ctx.headline}"
        )

    def _answer_next_steps(self, ctx):
        if not ctx:
            return (
                "General next steps for PFAS investigation:\n\n"
                "1. Run a Scanner prediction for your target location\n"
                "2. Review the SHAP attribution to understand what's driving the risk\n"
                "3. Use the Simulation Lab to model remediation or scenario changes\n"
                "4. Commission certified laboratory sampling for regulatory decisions\n"
                "5. Consult your national environmental authority if levels exceed thresholds"
            )
        if ctx.exceedance_prob >= 0.65:
            return (
                f"Given the HIGH risk prediction ({ctx.exceedance_prob*100:.0f}%) for {ctx.compound}, "
                "recommended next steps:\n\n"
                "1. Commission certified laboratory PFAS analysis immediately\n"
                "2. Notify your local environmental regulator if confirmed above threshold\n"
                "3. Identify and investigate potential source areas (especially if near airport, military base, or industrial site)\n"
                "4. Consider interim precautionary measures (alternative water supply if drinking water affected)\n"
                "5. Use the Simulation Lab to model remediation options and their expected risk reduction"
            )
        if ctx.exceedance_prob >= 0.35:
            return (
                f"Given the MODERATE risk prediction ({ctx.exceedance_prob*100:.0f}%) for {ctx.compound}, "
                "recommended next steps:\n\n"
                "1. Schedule targeted PFAS sampling at this location\n"
                "2. Investigate the dominant risk driver: {ctx.top_features[0]['label']}\n"
                "3. Check nearby potential sources (use the Explorer map)\n"
                "4. Monitor with annual sampling to detect any trend\n"
                "5. Use the Simulation Lab to test 'what if' cleanup scenarios"
            )
        return (
            f"The risk prediction is relatively LOW ({ctx.exceedance_prob*100:.0f}%) for {ctx.compound}. "
            "Recommended steps:\n\n"
            "1. Include this site in routine monitoring programmes\n"
            "2. Re-assess if land use changes nearby (new industrial activity, airport works, etc.)\n"
            "3. Use the Simulation Lab to test worst-case scenarios for contingency planning"
        )

    def _answer_top_factor(self, ctx):
        if not ctx or not ctx.top_features:
            return "No analysis results available yet. Please run a scan first."
        top = ctx.top_features[0]
        second = ctx.top_features[1] if len(ctx.top_features) > 1 else None
        resp = (
            f"The biggest factor in this prediction is: {top['label']}\n\n"
            f"Impact: {top['shap']:+.3f} (SHAP) — {top['direction']}\n\n"
            f"{top['human_explanation']}"
        )
        if second:
            resp += f"\n\nSecond most influential: {second['label']} (SHAP: {second['shap']:+.3f}) — {second['human_explanation']}"
        return resp

    def _answer_safety(self, ctx):
        if not ctx:
            return (
                "This tool provides probabilistic risk screening — not a drinking water safety certificate. "
                "Always rely on certified laboratory analysis and guidance from your public health authority "
                "for decisions about water, food, or land use safety. Run a location scan first "
                "for a site-specific risk estimate."
            )
        if ctx.exceedance_prob >= 0.65:
            return (
                f"⚠️ The model estimates a {ctx.exceedance_prob*100:.0f}% chance that "
                f"{ctx.compound} levels exceed 100 ng/L at this location. "
                f"Estimated concentration: {ctx.concentration_ngl:.1f} ng/L.\n\n"
                "This is a HIGH RISK screening result. Do NOT treat this as a safety clearance. "
                "Key actions:\n"
                "• Do not rely on untested surface or groundwater at this location\n"
                "• Commission laboratory testing from an accredited environmental laboratory\n"
                "• Contact your national environmental or public health authority\n"
                "• Note that even laboratory-tested results require expert interpretation"
            )
        if ctx.exceedance_prob >= 0.35:
            return (
                f"The predicted risk is moderate ({ctx.exceedance_prob*100:.0f}% exceedance probability) "
                f"for {ctx.compound}. Estimated concentration: {ctx.concentration_ngl:.1f} ng/L.\n\n"
                "This warrants precautionary investigation. Commission laboratory testing before "
                "making any decisions about water use or land management. "
                "This AI model provides screening guidance only."
            )
        return (
            f"The predicted risk is relatively low ({ctx.exceedance_prob*100:.0f}% exceedance probability) "
            f"for {ctx.compound} at this location. Estimated concentration: {ctx.concentration_ngl:.1f} ng/L.\n\n"
            "While this is a lower-risk result, no AI model replaces actual water testing. "
            "For regulatory certainty or public health decisions, always request certified laboratory analysis. "
            f"{ctx.data_quality_note}"
        )

    def _answer_concentration(self, ctx):
        if not ctx:
            return "Run a location scan to get a concentration estimate for a specific site."
        return (
            f"Estimated {ctx.compound} concentration: approximately {ctx.concentration_ngl:.1f} ng/L.\n\n"
            "This is a log-scale regression estimate from the model — actual laboratory values will vary.\n\n"
            "For context:\n"
            "• EU limit (20 priority PFAS combined): 100 ng/L\n"
            "• EU total PFAS limit: 10 ng/L\n"
            "• US EPA MCL (PFOS + PFOA): 4 ng/L each\n\n"
            f"{ctx.data_quality_note}"
        )

    def _answer_accuracy(self, ctx):
        base = PFAS_KNOWLEDGE["accuracy"]
        if ctx:
            return base + f"\n\nFor this specific site: {ctx.data_quality_note}"
        return base

    def _answer_coverage(self, ctx):
        if not ctx:
            return "Run a location scan first to see how close the nearest measurement is."
        return ctx.data_quality_note

    def _answer_compound_context(self, ctx):
        compound_map = {
            "PFOS": "pfos", "PFOA": "pfoa", "PFHxS": "pfhxs",
            "PFNA": "pfna", "PFDA": "pfda", "PFBS": "pfbs",
            "PFHpA": "pfhpa", "GENERAL": "general",
        }
        topic = compound_map.get(ctx.compound, "pfas")
        return PFAS_KNOWLEDGE.get(topic, PFAS_KNOWLEDGE["pfas"])
