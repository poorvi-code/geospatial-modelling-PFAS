import logging
import streamlit as st
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import folium
import geopandas as gpd
from pathlib import Path
from folium.plugins import HeatMap
from geopy.geocoders import Nominatim

# Paths
ROOT         = Path(__file__).resolve().parent.parent.parent
GOLDEN_PATH  = ROOT / "dataset" / "pfas_golden.parquet"
HOTSPOT_PATH = ROOT / "outputs" / "spatial" / "pfas_hotspots.geojson"

# Models availability
try:
    import sys
    sys.path.insert(0, str(ROOT / "implementation"))
    from api import PFASPredictor
    from simulation import SimulationEngine, SCENARIO_PRESETS
    from xai import XAIEngine
    _MODELS_AVAILABLE = True
except Exception as _e:
    _MODELS_AVAILABLE = False
    _LOAD_ERROR = str(_e)

EMERALD_SEQUENCE = ["#10b981", "#34d399", "#6ee7b7", "#a7f3d0", "#d1fae5"]

@st.cache_data(show_spinner=False)
def load_summary():
    if not GOLDEN_PATH.exists():
        return None
    schema = pq.read_schema(GOLDEN_PATH).names
    cols   = [c for c in ["country","substance","year","above_100_ng_l","source_system","lat","lon"] if c in schema]
    df = pd.read_parquet(GOLDEN_PATH, columns=cols)
    year_s = pd.to_numeric(df.get("year", pd.Series()), errors="coerce")
    exc_s  = pd.to_numeric(df.get("above_100_ng_l", pd.Series()), errors="coerce")
    return {
        "rows":       len(df),
        "countries":  int(df["country"].replace("Unknown", np.nan).nunique()) if "country" in df else 0,
        "substances": int(df["substance"].replace("Unknown", np.nan).nunique()) if "substance" in df else 0,
        "year_min":   int(year_s.min()) if year_s.notna().any() else None,
        "year_max":   int(year_s.max()) if year_s.notna().any() else None,
        "exceedance": float(exc_s.mean() * 100) if exc_s.notna().any() else None,
        "top_countries":  df["country"].value_counts().head(10) if "country" in df else pd.Series(),
        "top_substances": df["substance"].value_counts().head(7)  if "substance" in df else pd.Series(),
        "source_mix":     df["source_system"].value_counts()      if "source_system" in df else pd.Series(),
        "df_preview":     df,
    }

@st.cache_data(show_spinner=False)
def load_trend_data():
    if not GOLDEN_PATH.exists():
        return pd.DataFrame()
    cols = ["year", "substance", "log_value", "above_100_ng_l"]
    schema = pq.read_schema(GOLDEN_PATH).names
    cols = [c for c in cols if c in schema]
    df = pd.read_parquet(GOLDEN_PATH, columns=cols)
    df["year"] = pd.to_numeric(df.get("year"), errors="coerce")
    return df.dropna(subset=["year"])

@st.cache_data(show_spinner=False)
def load_hotspots():
    if not HOTSPOT_PATH.exists():
        return None
    gdf = gpd.read_file(HOTSPOT_PATH)
    gdf["lat"] = gdf.geometry.y
    gdf["lon"] = gdf.geometry.x
    return gdf

@st.cache_resource(show_spinner=False)
def get_backend():
    if not _MODELS_AVAILABLE:
        return None, None, None
    try:
        predictor = PFASPredictor()
        sim       = SimulationEngine()
        xai       = XAIEngine(predictor.clf)
        return predictor, sim, xai
    except Exception as e:
        return None, None, str(e)

def get_plotly_layout(height=400):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
    )

def inject_tailwind():
    st.markdown('<script src="https://cdn.tailwindcss.com"></script>', unsafe_allow_html=True)
    st.markdown("""
    <style>
    /* Fix for Streamlit's container padding when using Tailwind */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    </style>
    """, unsafe_allow_html=True)

def card(title, body):
    st.markdown(f'''
    <div class="bg-gray-50 dark:bg-slate-900 border border-gray-200 dark:border-slate-800 rounded-lg p-6 mb-4">
        <div class="text-xs font-bold uppercase tracking-wider text-gray-500 dark:text-slate-400 mb-3">{title}</div>
        <div class="text-sm leading-relaxed text-gray-700 dark:text-slate-300 opacity-90">{body}</div>
    </div>
    ''', unsafe_allow_html=True)
