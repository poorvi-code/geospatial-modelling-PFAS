# PFAS Geospatial Decision-Support System

A modular Python system for global PFAS contamination modeling, hotspot detection, and risk assessment.

## Project Structure
- `dataset/`: Input raw data (`pfas_raw.csv`, `airports.csv`).
- `outputs/`: All generated results, models, and spatial data.
- `implementation/`: Core Python modules for the full pipeline.
- `main.py`: Orchestrator to run everything in order.

## How to Run from Scratch

### 1. Requirements
Ensure you have Python 3.10+ installed.

```bash
# Install dependencies
pip install pandas geopandas pyarrow scipy numpy scikit-learn requests tqdm joblib lightgbm esda libpysal streamlit pydeck
```

### 2. Run the Full Pipeline
This will clean the data, build features, train models, and identify hotspots.

```bash
python main.py
```

### 3. Launch the Dashboard
View the interactive risk map and perform point queries.

```bash
streamlit run implementation/dashboard.py
```

## Modules
1. **`clean.py`**: Ingestion, target creation, and spatial feature engineering.
2. **`train.py`**: LightGBM training for exceedance and concentration.
3. **`hotspot.py`**: Local spatial autocorrelation (Getis-Ord Gi*) and heatmap generation.
4. **`cci.py`**: Composite Contamination Index calculation across 7 compounds.
5. **`api.py`**: Low-latency global inference engine.