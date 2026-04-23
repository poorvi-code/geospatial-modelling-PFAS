# PFAS Geospatial Intelligence Platform
---

## What This Does

PFAS (per- and polyfluoroalkyl substances) are man-made chemicals found in soil, groundwater and drinking water worldwide. This platform:

1. **Merges** three real-world PFAS datasets into a single cleaned, research-ready dataset (369 K+ measurements)
2. **Trains** state-of-the-art gradient-boosted models (LightGBM + XGBoost) to predict contamination risk at any location
3. **Explains** every prediction using SHAP values, converted into plain English
4. **Simulates** "what if" scenarios - what happens if a new airport opens nearby? What if a cleanup removes 70% of PFAS?
5. **Visualises** everything through a research-grade dashboard any non-expert can navigate

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (cleaning → training → hotspots → CCI)
python main.py

# 3. Launch the dashboard
streamlit run implementation/dashboard.py
```

---

## Dashboard Tabs

| Tab | What it shows |
|-----|---------------|
| **Overview** | Europe-wide contamination heatmap + KPI metrics |
| **Risk Scanner** | Enter any location → get a risk score, concentration estimate, and plain-English explanation |
| **Simulation Lab** | Choose scenario presets (Industrial Spill, Airport Expansion, Cleanup Drive…) and see how risk changes |
| **AI Explain** | SHAP waterfall chart + live chat assistant — ask anything in plain language |
| **Data Explorer** | Temporal trends, compound comparison, country breakdown |

---

## Dataset Sources 

| File | Description |
|------|-------------|
| `dataset/pfas_raw.csv` | Primary PFAS measurements CSV (CNRS PDH raw export) |
| `dataset/pdh_data.parquet` | CNRS PFAS Data Hub bulk download (local copy) |
| `dataset/pfas_contamination.shp` | PFAS contamination shapefile |
| `dataset/airports.csv` | OurAirports global airport database (proximity features) |

---

## Machine Learning Stack

| Component | Technology |
|-----------|-----------|
| Primary classifier | LightGBM (Optuna-tuned, 80 trials, spatial block CV) |
| Secondary comparison | XGBoost + Random Forest |
| Class imbalance | ADASYN oversampling |
| Calibration | Isotonic regression (CalibratedClassifierCV) |
| Explainability | SHAP TreeExplainer |
| Tracking | MLflow |

**Model target:** Binary exceedance of 100 ng/L PFAS concentration  
**Validation:** 5-fold Spatial Block GroupKFold (prevents geographic leakage)  
**Typical CV ROC-AUC:** 0.85–0.92 depending on dataset coverage

---

## The 7 PFAS Compounds Modelled

| Compound | Carbon Chain | Type | Priority |
|----------|-------------|------|----------|
| PFOS | C8 sulfonate | Long-chain | High |
| PFOA | C8 carboxylate | Long-chain | High |
| PFNA | C9 carboxylate | Long-chain | Medium |
| PFDA | C10 carboxylate | Long-chain | Medium |
| PFHxS | C6 sulfonate | Short-chain | Watch |
| PFHpA | C7 carboxylate | Short-chain | Watch |
| PFBS | C4 sulfonate | Short-chain |  Monitor |

---


