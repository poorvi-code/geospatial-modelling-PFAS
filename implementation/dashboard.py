"""
implementation/dashboard.py
===========================
PFAS Risk Intelligence — Redesigned Dashboard
----------------------------------------------
5 Tabs:
  1. 🌍 Overview        — KPIs, interactive map, coverage stats
  2. 🔍 Risk Scanner    — Point prediction with plain-English output
  3. 🎮 Simulation Lab  — Gamified what-if scenarios
  4. 🧠 AI Explain      — SHAP explanations + live chat assistant
  5. 📊 Data Explorer   — EDA charts (trends, substances, countries)

Design principles:
  - Dark premium theme (no jargon, no technical terms in UI)
  - All technical outputs translated into plain English
  - Chat-based XAI answers user questions conversationally
  - Maps as the primary visual anchor
  - Mobile-friendly wide layout with emoji status codes
"""

import json
import logging
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyarrow.parquet as pq
import streamlit as st
from folium.plugins import HeatMap
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Local module imports (graceful fallback if models not yet trained)
# ---------------------------------------------------------------------------
try:
    import sys, os
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from api import PFASPredictor
    from simulation import SimulationEngine, SCENARIO_PRESETS
    from xai import XAIEngine
    _MODELS_AVAILABLE = True
except Exception as _e:
    _MODELS_AVAILABLE = False
    _LOAD_ERROR = str(_e)

log = logging.getLogger(__name__)
ROOT         = Path(__file__).resolve().parent.parent
GOLDEN_PATH  = ROOT / "dataset" / "pfas_golden.parquet"
HOTSPOT_PATH = ROOT / "outputs" / "spatial" / "pfas_hotspots.geojson"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PFAS Risk Intelligence",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS  — premium dark theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    color: #e2e8f0;
}
.block-container { padding: 1.5rem 2rem 2rem; }

/* Hero */
.hero {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(236,72,153,0.10));
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px;
    padding: 2rem 2.4rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(8px);
}
.hero h1 { font-size: 2.2rem; font-weight: 700; color: #f1f5f9; margin: 0 0 0.5rem; }
.hero p  { color: #94a3b8; font-size: 1rem; line-height: 1.65; margin: 0; max-width: 70ch; }
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.2);
    color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* Metric cards */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.metric-card {
    flex: 1;
    background: rgba(30,41,59,0.8);
    border: 1px solid rgba(148,163,184,0.15);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: rgba(99,102,241,0.5); }
.metric-label { color: #64748b; font-size: 0.82rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.4rem; }
.metric-value { font-size: 2rem; font-weight: 700; color: #f1f5f9; line-height: 1; }
.metric-sub   { color: #64748b; font-size: 0.82rem; margin-top: 0.3rem; }

/* Cards */
.card {
    background: rgba(30,41,59,0.7);
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(6px);
}
.card-title { font-size: 1rem; font-weight: 600; color: #f1f5f9; margin-bottom: 0.6rem; }
.card-body  { color: #94a3b8; font-size: 0.93rem; line-height: 1.6; }

/* Risk badges */
.badge-safe     { background:#052e16; color:#4ade80; border:1px solid #166534; border-radius:8px; padding:0.6rem 1rem; font-weight:600; }
.badge-watch    { background:#422006; color:#fb923c; border:1px solid #9a3412; border-radius:8px; padding:0.6rem 1rem; font-weight:600; }
.badge-caution  { background:#431407; color:#fbbf24; border:1px solid #b45309; border-radius:8px; padding:0.6rem 1rem; font-weight:600; }
.badge-high     { background:#450a0a; color:#f87171; border:1px solid #991b1b; border-radius:8px; padding:0.6rem 1rem; font-weight:600; }
.badge-critical { background:#3b0764; color:#e879f9; border:1px solid #7e22ce; border-radius:8px; padding:0.6rem 1rem; font-weight:600; }

/* Chat */
.chat-user     { background:rgba(99,102,241,0.15); border:1px solid rgba(99,102,241,0.3); border-radius:12px 12px 4px 12px; padding:0.8rem 1rem; margin:0.5rem 0 0.5rem 3rem; color:#c7d2fe; }
.chat-ai       { background:rgba(30,41,59,0.9); border:1px solid rgba(148,163,184,0.15); border-radius:12px 12px 12px 4px; padding:0.8rem 1rem; margin:0.5rem 3rem 0.5rem 0; color:#e2e8f0; }
.chat-input-label { color:#64748b; font-size:0.85rem; margin-bottom:0.4rem; }

/* Gauge */
.gauge-wrap { display:flex; align-items:center; justify-content:center; }

/* Confidence banner */
.conf-high   { background:#052e16; border-left:4px solid #22c55e; padding:0.7rem 1rem; border-radius:0 8px 8px 0; color:#86efac; font-size:0.9rem; }
.conf-medium { background:#422006; border-left:4px solid #f97316; padding:0.7rem 1rem; border-radius:0 8px 8px 0; color:#fdba74; font-size:0.9rem; }
.conf-low    { background:#450a0a; border-left:4px solid #ef4444; padding:0.7rem 1rem; border-radius:0 8px 8px 0; color:#fca5a5; font-size:0.9rem; }

/* Plotly charts dark override */
[data-testid="stPlotlyChart"] { border-radius: 10px; overflow: hidden; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: rgba(15,23,42,0.6); border-radius: 10px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; color: #64748b; font-weight: 500; }
.stTabs [aria-selected="true"] { background: rgba(99,102,241,0.25) !important; color: #a5b4fc !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: rgba(15,23,42,0.95); border-right: 1px solid rgba(148,163,184,0.1); }
[data-testid="stSidebar"] .stMarkdown { color: #94a3b8; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white; border: none; border-radius: 10px;
    font-weight: 600; padding: 0.6rem 1.4rem;
    transition: all 0.2s;
}
.stButton > button:hover { opacity: 0.9; transform: translateY(-1px); box-shadow: 0 4px 16px rgba(99,102,241,0.4); }

/* Metrics widget */
[data-testid="stMetric"] { background: rgba(30,41,59,0.6); padding: 0.8rem 1rem; border-radius: 10px; border: 1px solid rgba(148,163,184,0.1); }
[data-testid="stMetricValue"] { color: #f1f5f9 !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; }

/* Info/Warning */
.stAlert { border-radius: 10px; }

/* Number input */
.stNumberInput input { background: rgba(30,41,59,0.8) !important; color: #e2e8f0 !important; border: 1px solid rgba(148,163,184,0.2) !important; border-radius: 8px !important; }

/* Selectbox */
.stSelectbox > div > div { background: rgba(30,41,59,0.8) !important; color: #e2e8f0 !important; border: 1px solid rgba(148,163,184,0.2) !important; border-radius: 8px !important; }

/* Score ring pulse */
@keyframes pulse-ring { 0%{transform:scale(1);opacity:1} 50%{transform:scale(1.05);opacity:0.7} 100%{transform:scale(1);opacity:1} }
.score-ring { animation: pulse-ring 2.5s ease-in-out infinite; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Plotly layout defaults (dark theme)
# ---------------------------------------------------------------------------
PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(15,23,42,0.0)",
    plot_bgcolor="rgba(30,41,59,0.4)",
    font=dict(family="Inter", color="#94a3b8"),
    margin=dict(l=10, r=10, t=36, b=10),
)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

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
    cols = [c for c in ["year", "substance", "log_value", "above_100_ng_l"] if True]
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


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _risk_badge(prob: float) -> str:
    score = prob * 100
    if score < 20:  return "badge-safe",     "🟢 Safe Zone"
    if score < 40:  return "badge-watch",    "🟡 Watch Zone"
    if score < 60:  return "badge-caution",  "🟠 Caution"
    if score < 80:  return "badge-high",     "🔴 High Alert"
    return               "badge-critical",   "🚨 Critical"


def _confidence_block(km: float):
    if km < 50:
        return "conf-high",   f"📍 High Confidence — {km:.0f} km from nearest real measurement"
    if km < 200:
        return "conf-medium", f"📊 Medium Confidence — {km:.0f} km from nearest measurement. Results are indicative."
    return "conf-low",        f"⚠️ Low Confidence — {km:.0f} km from nearest measurement. Treat as rough estimate only."


def _gauge_chart(value: float, title: str = "Risk Score", color: str = "#6366f1"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 15, "color": "#94a3b8"}},
        number={"font": {"size": 34, "color": "#f1f5f9"}, "suffix": "%"},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#334155", "tickfont": {"color": "#64748b"}},
            "bar":  {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(30,41,59,0.8)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  20],  "color": "rgba(34,197,94,0.12)"},
                {"range": [20, 40],  "color": "rgba(234,179,8,0.12)"},
                {"range": [40, 60],  "color": "rgba(249,115,22,0.12)"},
                {"range": [60, 80],  "color": "rgba(239,68,68,0.12)"},
                {"range": [80, 100], "color": "rgba(153,27,27,0.2)"},
            ],
            "threshold": {"line": {"color": "#fff", "width": 2}, "thickness": 0.9, "value": value},
        },
    ))
    fig.update_layout(**PLOTLY_DARK, height=260)
    return fig


def _delta_color(delta: float) -> str:
    if delta > 5:   return "#f87171"
    if delta < -5:  return "#4ade80"
    return "#fbbf24"


def _make_map_europe(hotspots_gdf=None, sample_df=None) -> folium.Map:
    m = folium.Map(
        location=[51.0, 10.0], zoom_start=4,
        tiles="CartoDB dark_matter",
        attr="PFAS Risk Intelligence",
    )
    if hotspots_gdf is not None and not hotspots_gdf.empty and "gi_zscore" in hotspots_gdf.columns:
        heat_data = [[r["lat"], r["lon"], max(float(r["gi_zscore"]), 0)]
                     for _, r in hotspots_gdf.iterrows()]
        HeatMap(heat_data, radius=18, blur=22,
                gradient={"0.2": "#1e3a5f", "0.5": "#6366f1", "0.8": "#f97316", "1.0": "#ef4444"},
                min_opacity=0.4).add_to(m)
    if sample_df is not None and not sample_df.empty:
        sample = sample_df.dropna(subset=["lat", "lon"]).sample(min(3000, len(sample_df)), random_state=42)
        for _, row in sample.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=3, weight=0, fill=True,
                fill_color="#6366f1", fill_opacity=0.35,
                tooltip=f"{row.get('substance','?')} | {row.get('country','?')}",
            ).add_to(m)
    return m


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
for _k in ["chat_history", "scan_result", "sim_result", "xai_result"]:
    if _k not in st.session_state:
        st.session_state[_k] = [] if _k == "chat_history" else None

# ---------------------------------------------------------------------------
# Load resources
# ---------------------------------------------------------------------------
summary    = load_summary()
trend_df   = load_trend_data()
hotspots   = load_hotspots()
predictor, sim_engine, xai_engine = get_backend()
geolocator = Nominatim(user_agent="pfas_risk_dashboard_v3")

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧪 PFAS Risk Intelligence")
    st.markdown("---")

    st.markdown("**🔎 Search a Location**")
    addr = st.text_input("City, region, or address", placeholder="e.g. Brussels, Belgium",
                         label_visibility="collapsed")
    if addr:
        try:
            loc = geolocator.geocode(addr)
            if loc:
                st.success(f"📍 {loc.address[:60]}...")
                st.session_state["preset_lat"] = loc.latitude
                st.session_state["preset_lon"] = loc.longitude
            else:
                st.warning("Location not found — try a different name.")
        except Exception:
            st.warning("Search unavailable. Enter coordinates manually.")

    st.markdown("---")
    st.markdown("**📖 How to use this dashboard**")
    st.markdown("""
1. **Overview** — See where contamination is concentrated on the map
2. **Risk Scanner** — Enter a location to get a risk assessment
3. **Simulation Lab** — Test 'what if' scenarios and see the outcome
4. **AI Explain** — Ask any question about the results in plain English
5. **Data Explorer** — Explore trends and patterns in the full dataset
""")

    st.markdown("---")
    if summary:
        st.markdown("**📦 Dataset**")
        st.markdown(f"- `{summary['rows']:,}` measurements")
        st.markdown(f"- `{summary['countries']}` countries")
        st.markdown(f"- `{summary['year_min']}` – `{summary['year_max']}`")
        if not _MODELS_AVAILABLE:
            st.error("⚠️ Models not trained yet. Run `python main.py` first.")
        else:
            st.success("✅ Models ready")


# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="hero-badge">Research Tool · PFAS Contamination Risk</div>
  <h1>🧪 PFAS Risk Intelligence Platform</h1>
  <p>
    Where are "forever chemicals" most concentrated? What is the risk at your location?
    This platform combines 369 000+ real measurements with machine learning to screen any
    location on Earth — and explains every result in plain language.
  </p>
</div>
""", unsafe_allow_html=True)

# Global KPIs
c1, c2, c3, c4 = st.columns(4)
if summary:
    c1.metric("📊 Measurements", f"{summary['rows']:,}")
    c2.metric("🌍 Countries",    f"{summary['countries']}")
    c3.metric("🧬 Compounds",    f"{summary['substances']}")
    exc_txt = f"{summary['exceedance']:.1f}%" if summary["exceedance"] is not None else "N/A"
    c4.metric("⚠️ Above 100 ng/L", exc_txt)
else:
    for col, lbl in zip([c1,c2,c3,c4], ["Measurements","Countries","Compounds","Above 100 ng/L"]):
        col.metric(lbl, "No data")

st.markdown("---")

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "🌍 Overview",
    "🔍 Risk Scanner",
    "🎮 Simulation Lab",
    "🧠 AI Explain",
    "📊 Data Explorer",
])

# ===========================================================================
# TAB 1 — OVERVIEW
# ===========================================================================
with tabs[0]:
    col_map, col_stats = st.columns([1.8, 1])

    with col_map:
        st.markdown('<div class="card"><div class="card-title">🗺️ PFAS Contamination Map</div>'
                    '<div class="card-body">Darker/warmer zones have higher contamination density. '
                    'Purple dots are individual measurements. Click zoom to explore regions.</div></div>',
                    unsafe_allow_html=True)
        sample_for_map = summary["df_preview"] if summary else None
        europe_map = _make_map_europe(hotspots, sample_for_map)
        st_folium(europe_map, use_container_width=True, height=480)

    with col_stats:
        st.markdown('<div class="card"><div class="card-title">📈 Dataset Coverage</div></div>',
                    unsafe_allow_html=True)
        if summary and not summary["source_mix"].empty:
            fig_src = px.bar(
                summary["source_mix"].reset_index(),
                x="count", y="source_system" if "source_system" in summary["source_mix"].reset_index().columns else summary["source_mix"].reset_index().columns[0],
                orientation="h",
                color_discrete_sequence=["#6366f1"],
                title="Measurements by Source",
            )
            fig_src.update_layout(**PLOTLY_DARK, height=200, showlegend=False,
                                  xaxis_title="Records", yaxis_title="")
            st.plotly_chart(fig_src, use_container_width=True)

        if summary and not summary["top_substances"].empty:
            fig_sub = px.pie(
                values=summary["top_substances"].values,
                names=summary["top_substances"].index,
                title="PFAS Compounds",
                color_discrete_sequence=px.colors.sequential.Plasma_r,
            )
            fig_sub.update_traces(textinfo="label+percent")
            fig_sub.update_layout(**PLOTLY_DARK, height=260)
            st.plotly_chart(fig_sub, use_container_width=True)

    # Country bar chart
    if summary and not summary["top_countries"].empty:
        st.markdown("#### 🌏 Top 10 Countries by Measurement Count")
        fig_cty = px.bar(
            summary["top_countries"].reset_index().rename(columns={"index": "country", "country": "country", "count": "records"}),
            x="count" if "count" in summary["top_countries"].reset_index().columns else summary["top_countries"].reset_index().columns[-1],
            y="country" if "country" in summary["top_countries"].reset_index().columns else summary["top_countries"].reset_index().columns[0],
            orientation="h",
            color_discrete_sequence=["#8b5cf6"],
            title="",
        )
        fig_cty.update_layout(**PLOTLY_DARK, height=340, xaxis_title="Measurements", yaxis_title="")
        st.plotly_chart(fig_cty, use_container_width=True)


# ===========================================================================
# TAB 2 — RISK SCANNER
# ===========================================================================
with tabs[1]:
    if not _MODELS_AVAILABLE or predictor is None:
        st.error(
            f"⚠️ Models are not ready yet. Run `python main.py` from the project root to train them.\n\n"
            + (f"Error: {_LOAD_ERROR}" if not _MODELS_AVAILABLE else "")
        )
    else:
        col_form, col_result = st.columns([1, 1.4])

        with col_form:
            st.markdown('<div class="card"><div class="card-title">📍 Location & Parameters</div></div>',
                        unsafe_allow_html=True)
            scan_lat = st.number_input("Latitude",  value=float(st.session_state.get("preset_lat", 51.5)), format="%.5f", step=0.1, key="slat")
            scan_lon = st.number_input("Longitude", value=float(st.session_state.get("preset_lon", -0.12)), format="%.5f", step=0.1, key="slon")
            substance = st.selectbox("Which PFAS compound?",
                                     ["PFOS", "PFOA", "PFHXS", "PFNA", "PFDA", "PFHPA", "PFBS"],
                                     help="PFOS and PFOA are the most studied and regulated compounds.")
            run_btn = st.button("🔍 Run Risk Assessment", use_container_width=True)

            st.markdown('<div class="card"><div class="card-body">'
                        '💡 <strong>Tip:</strong> Use the address search in the sidebar to '
                        'auto-fill the coordinates for any city or region.'
                        '</div></div>', unsafe_allow_html=True)

        with col_result:
            if run_btn or st.session_state.scan_result:
                if run_btn:
                    with st.spinner("Analysing location..."):
                        try:
                            res = predictor.predict(scan_lat, scan_lon, substance=substance)
                            st.session_state.scan_result = res
                            st.session_state.scan_lat    = scan_lat
                            st.session_state.scan_lon    = scan_lon
                            # Generate XAI explanation
                            if xai_engine:
                                from api import PFASPredictor as _P
                                X_feat, _, _ = predictor.build_feature_frame(scan_lat, scan_lon, substance=substance)
                                xai_res = xai_engine.explain(
                                    X_feat,
                                    exceedance_prob    = res["exceedance_prob"],
                                    concentration_ngl  = res["predicted_value_ngl"],
                                    compound           = substance,
                                    nearest_km         = res["dist_to_nearest_sample_km"],
                                )
                                st.session_state.xai_result = xai_res
                                # Prime the chat assistant with context
                                st.session_state.chat_history = []
                        except Exception as e:
                            st.error(f"Prediction failed: {e}")
                            st.stop()

                res = st.session_state.scan_result
                if res:
                    prob  = res["exceedance_prob"]
                    conc  = res["predicted_value_ngl"]
                    score = prob * 100

                    badge_cls, badge_txt = _risk_badge(prob)
                    conf_cls, conf_txt   = _confidence_block(res["dist_to_nearest_sample_km"])

                    # Gauge
                    gauge_color = ("#22c55e" if score<20 else "#eab308" if score<40 else
                                   "#f97316" if score<60 else "#ef4444")
                    st.plotly_chart(_gauge_chart(score, "Exceedance Risk Score", gauge_color),
                                    use_container_width=True)

                    # Risk level badge
                    st.markdown(f'<p class="{badge_cls}">{badge_txt}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="{conf_cls}">{conf_txt}</p>', unsafe_allow_html=True)

                    # Metrics row
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Exceedance Probability", f"{prob*100:.1f}%")
                    m2.metric("Est. Concentration",     f"{conc:.1f} ng/L")
                    m3.metric("Nearest Data Point",     f"{res['dist_to_nearest_sample_km']:.0f} km")

                    # Plain language reading
                    xai_res = st.session_state.xai_result
                    if xai_res:
                        st.markdown(f'<div class="card"><div class="card-title">📝 What this means</div>'
                                    f'<div class="card-body">{xai_res.headline}</div></div>',
                                    unsafe_allow_html=True)
                        if xai_res.risk_drivers:
                            st.markdown("**🚩 Key risk signals at this location:**")
                            for d in xai_res.risk_drivers[:3]:
                                st.markdown(f"• {d}")
                        if xai_res.protective_factors:
                            st.markdown("**✅ Factors reducing risk:**")
                            for p in xai_res.protective_factors[:2]:
                                st.markdown(f"• {p}")
                    else:
                        st.info("Run a scan to see the full explanation.")

            else:
                st.markdown('<div class="card"><div class="card-title">What appears here</div>'
                            '<div class="card-body">After you click "Run Risk Assessment", this panel will show:'
                            '<br><br>• A 0–100 risk score with a visual gauge'
                            '<br>• Plain-English explanation of what drives the risk'
                            '<br>• Estimated PFAS concentration in ng/L'
                            '<br>• Confidence indicator based on nearby data density'
                            '</div></div>', unsafe_allow_html=True)


# ===========================================================================
# TAB 3 — SIMULATION LAB (Gamified)
# ===========================================================================
with tabs[2]:
    if not _MODELS_AVAILABLE or sim_engine is None:
        st.error("⚠️ Models not ready. Run `python main.py` first.")
    elif st.session_state.scan_result is None:
        st.info("🔍 Run a **Risk Scanner** analysis first — the Simulation Lab uses it as your baseline.")
    else:
        res = st.session_state.scan_result
        base_prob = res["exceedance_prob"]

        st.markdown(f"""
<div class="card">
  <div class="card-title">🎮 What-If Scenario Laboratory</div>
  <div class="card-body">
    <strong>Baseline risk: {base_prob*100:.1f}%</strong> — now explore how different pressures
    or interventions change that number. Pick a preset or tune the sliders yourself.
    Monte Carlo shows the uncertainty range across 1,000 simulations.
  </div>
</div>
""", unsafe_allow_html=True)

        col_preset, col_sliders, col_outcome = st.columns([1.1, 1.2, 1.2])

        with col_preset:
            st.markdown("#### 🃏 Scenario Presets")
            preset_options = {v["label"]: k for k, v in SCENARIO_PRESETS.items()}
            chosen_label   = st.radio(
                "Choose a scenario",
                list(preset_options.keys()),
                label_visibility="collapsed",
            )
            chosen_key = preset_options[chosen_label]
            preset_info = SCENARIO_PRESETS[chosen_key]
            st.markdown(f'<div class="card"><div class="card-body">{preset_info["description"]}</div></div>',
                        unsafe_allow_html=True)

        with col_sliders:
            st.markdown("#### 🎛️ Fine-tune Parameters")
            mc_runs = 500

            # Build overrides from sliders
            slider_mods = SCENARIO_PRESETS[chosen_key]["mods"].copy()

            ind_pct = st.slider("Industrial pressure", 0, 300, 100,
                                help="100 = no change.  200 = double the nearby contamination density.",
                                format="%d%%")
            slider_mods["spatial_density_boost"] = ind_pct / 100.0

            airport_km = st.slider("Distance to nearest airport (km)", 1, 300,
                                   int(np.clip(res.get("dist_to_airport_km", 50), 1, 300)),
                                   help="Move the airport closer or farther away.")
            slider_mods["airport_distance_km"] = float(airport_km)

            cleanup_pct = st.slider("Cleanup efficiency", 0, 90, 0,
                                    help="Higher = more PFAS removed from the neighbourhood.",
                                    format="%d%%")
            if cleanup_pct > 0:
                slider_mods["mean_log_value_reduction"] = cleanup_pct / 100.0

            post2018 = st.checkbox("Apply EU PFAS restrictions (post-2018 effect)", value=False)
            if post2018:
                slider_mods["is_post_2018_override"] = 1

            run_sim  = st.button("▶️ Run Simulation", use_container_width=True)
            run_mc   = st.button("🎲 + Monte Carlo Uncertainty", use_container_width=True)

        with col_outcome:
            st.markdown("#### 📊 Simulation Outcome")

            if run_sim or run_mc or st.session_state.sim_result:
                if run_sim or run_mc:
                    with st.spinner("Running simulation..."):
                        try:
                            X_feat, _, _ = predictor.build_feature_frame(
                                st.session_state.get("scan_lat", 51.5),
                                st.session_state.get("scan_lon", -0.12),
                            )
                            if run_mc:
                                sim_res = sim_engine.run_monte_carlo(X_feat, slider_mods, n_runs=mc_runs)
                            else:
                                sim_res = sim_engine.run_custom(X_feat, slider_mods, label=chosen_label)
                            st.session_state.sim_result = sim_res
                        except Exception as e:
                            st.error(f"Simulation failed: {e}")
                            st.stop()

                sim_res = st.session_state.sim_result
                if sim_res:
                    d = sim_res.delta_pts
                    dcol = _delta_color(d)

                    # Side-by-side gauges
                    g1, g2 = st.columns(2)
                    g1.plotly_chart(_gauge_chart(sim_res.base_score,     "Baseline",  "#6366f1"), use_container_width=True)
                    g2.plotly_chart(_gauge_chart(sim_res.scenario_score, "Scenario",  gauge_color if abs(d)<2 else ("#ef4444" if d>0 else "#22c55e")), use_container_width=True)

                    # Delta metric
                    st.metric("Change in risk", f"{d:+.1f} percentage points",
                              delta=f"{'↑ worse' if d>0 else '↓ better'}",
                              delta_color="inverse")

                    # Risk level
                    st.markdown(f'<p class="{sim_res.risk_color.replace("#","").replace(" ","")}">'
                                f'{sim_res.risk_level}</p>', unsafe_allow_html=True)

                    # MC band
                    if sim_res.mc_p50 > 0:
                        fig_mc = go.Figure()
                        fig_mc.add_trace(go.Bar(
                            x=["p5 (best)", "p50 (likely)", "p95 (worst)"],
                            y=[sim_res.mc_p5, sim_res.mc_p50, sim_res.mc_p95],
                            marker_color=["#22c55e", "#6366f1", "#ef4444"],
                            text=[f"{v:.1f}%" for v in [sim_res.mc_p5, sim_res.mc_p50, sim_res.mc_p95]],
                            textposition="outside",
                        ))
                        fig_mc.update_layout(**PLOTLY_DARK, title="Uncertainty Range (1 000 simulations)",
                                            height=220, yaxis_title="Risk (%)", xaxis_title="")
                        st.plotly_chart(fig_mc, use_container_width=True)

                    # Plain explanation
                    st.markdown(f'<div class="card"><div class="card-body">'
                                f'{sim_res.plain_explanation}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="card"><div class="card-body">'
                            'Choose a scenario or adjust the sliders, then click <strong>Run Simulation</strong>.'
                            '</div></div>', unsafe_allow_html=True)


# ===========================================================================
# TAB 4 — AI EXPLAIN (Chat)
# ===========================================================================
with tabs[3]:
    if not _MODELS_AVAILABLE or xai_engine is None:
        st.error("⚠️ Models not ready. Run `python main.py` first.")
    else:
        col_explain, col_chat = st.columns([1.1, 1])

        with col_explain:
            st.markdown("#### 🔬 Model Explanation")
            xai_res = st.session_state.xai_result

            if xai_res is None:
                st.info("Run a **Risk Scanner** assessment first to unlock the explanation view.")
            else:
                # Full narrative
                st.markdown(f'<div class="card"><div class="card-title">📝 Full Analysis</div>'
                            f'<div class="card-body">{xai_res.full_narrative}</div></div>',
                            unsafe_allow_html=True)

                # SHAP waterfall
                shap_df = pd.DataFrame(xai_res.top_features).head(8)
                if not shap_df.empty:
                    shap_df["color"] = shap_df["shap"].apply(
                        lambda v: "#ef4444" if v > 0 else "#22c55e"
                    )
                    fig_shap = go.Figure(go.Bar(
                        x=shap_df["shap"],
                        y=shap_df["label"],
                        orientation="h",
                        marker_color=shap_df["color"],
                        text=shap_df["direction"],
                        textposition="outside",
                    ))
                    fig_shap.update_layout(
                        **PLOTLY_DARK,
                        title="What factors drove this prediction?",
                        height=350,
                        xaxis_title="← reduces risk                increases risk →",
                        yaxis_title="",
                        xaxis=dict(zeroline=True, zerolinecolor="#334155", zerolinewidth=2),
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)

                # Data quality note
                st.markdown(f'<div class="card"><div class="card-body">{xai_res.data_quality_note}</div></div>',
                            unsafe_allow_html=True)

        with col_chat:
            st.markdown("#### 💬 Ask Me Anything")
            st.markdown(
                '<div class="card"><div class="card-body">'
                'I can answer questions about the results, PFAS science, the model, '
                'or any concerns you have — in plain English. No jargon.'
                '</div></div>',
                unsafe_allow_html=True
            )

            # Quick-question buttons
            st.markdown("**Quick questions:**")
            qcols = st.columns(2)
            quick_qs = [
                "Why is the risk high?",
                "What is PFOS?",
                "Is this safe to drink?",
                "How does the model work?",
                "What would reduce the risk?",
                "What does the score mean?",
            ]
            for i, q in enumerate(quick_qs):
                if qcols[i % 2].button(q, key=f"qq_{i}", use_container_width=True):
                    answer = xai_engine.chat(q)
                    st.session_state.chat_history.append(("user", q))
                    st.session_state.chat_history.append(("ai",   answer))

            # Display conversation
            chat_placeholder = st.container()
            with chat_placeholder:
                for role, msg in st.session_state.chat_history[-12:]:  # show last 6 exchanges
                    css = "chat-user" if role == "user" else "chat-ai"
                    icon = "🧑" if role == "user" else "🤖"
                    st.markdown(f'<div class="{css}">{icon} {msg}</div>', unsafe_allow_html=True)

            # Free-text input
            st.markdown("---")
            with st.form("chat_form", clear_on_submit=True):
                user_msg = st.text_input("Your question:", placeholder="e.g. Why is the risk not zero?",
                                         label_visibility="collapsed")
                send = st.form_submit_button("Send ✉️", use_container_width=True)
                if send and user_msg.strip():
                    answer = xai_engine.chat(user_msg)
                    st.session_state.chat_history.append(("user", user_msg))
                    st.session_state.chat_history.append(("ai",   answer))
                    st.rerun()

            if st.button("🗑️ Clear conversation", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()


# ===========================================================================
# TAB 5 — DATA EXPLORER
# ===========================================================================
with tabs[4]:
    if trend_df.empty:
        st.warning("Golden dataset not found. Run `python main.py` first.")
    else:
        st.markdown("#### 📅 PFAS Trends Over Time (2001–2024)")

        if "substance" in trend_df.columns and "log_value" in trend_df.columns:
            yearly = (
                trend_df.groupby(["year", "substance"])["log_value"]
                .median()
                .reset_index()
            )
            yearly.columns = ["Year", "Compound", "Median Log Concentration"]
            fig_trend = px.line(
                yearly, x="Year", y="Median Log Concentration", color="Compound",
                title="Median PFAS Concentration by Year (log scale — higher = more contaminated)",
                color_discrete_sequence=px.colors.qualitative.Bold,
                markers=True,
            )
            fig_trend.update_layout(**PLOTLY_DARK, height=380)
            st.plotly_chart(fig_trend, use_container_width=True)

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("#### 🧬 Compound Exceedance Rates")
            if "above_100_ng_l" in trend_df.columns and "substance" in trend_df.columns:
                exc_rate = (
                    trend_df.dropna(subset=["above_100_ng_l"])
                    .groupby("substance")["above_100_ng_l"]
                    .mean()
                    .reset_index()
                    .rename(columns={"substance": "Compound", "above_100_ng_l": "Exceedance Rate"})
                )
                exc_rate["Exceedance Rate (%)"] = (exc_rate["Exceedance Rate"] * 100).round(1)
                fig_exc = px.bar(
                    exc_rate.sort_values("Exceedance Rate (%)", ascending=False),
                    x="Compound", y="Exceedance Rate (%)",
                    color="Exceedance Rate (%)",
                    color_continuous_scale="Reds",
                    title="% of Measurements Exceeding 100 ng/L",
                )
                fig_exc.update_layout(**PLOTLY_DARK, height=320, showlegend=False,
                                      coloraxis_showscale=False)
                st.plotly_chart(fig_exc, use_container_width=True)

        with col_r:
            st.markdown("#### 📊 Concentration Distribution")
            if "log_value" in trend_df.columns and "substance" in trend_df.columns:
                sample_exp = trend_df.sample(min(15000, len(trend_df)), random_state=42)
                fig_violin = px.violin(
                    sample_exp, x="substance", y="log_value",
                    color="substance",
                    title="Spread of Measurements per Compound (log scale)",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    box=True,
                )
                fig_violin.update_layout(**PLOTLY_DARK, height=320, showlegend=False,
                                         xaxis_title="Compound", yaxis_title="log(concentration + 1)")
                st.plotly_chart(fig_violin, use_container_width=True)

        # Country table
        if summary and not summary["top_countries"].empty:
            st.markdown("#### 🌍 Countries with the Most Measurements")
            ctable = summary["top_countries"].reset_index()
            ctable.columns = ["Country", "Records"]
            ctable["Records"] = ctable["Records"].map("{:,}".format)
            st.dataframe(ctable, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='color:#475569;font-size:0.82rem;text-align:center;'>"
    "PFAS Risk Intelligence — Research Platform | "
    "Data: CNRS PDH, pfas_raw.csv, PFAS Contamination Shapefile | "
    "Model: LightGBM (Optuna-tuned) + XAI (SHAP) | "
    "⚠️ For research screening purposes only — not a regulatory determination."
    "</div>",
    unsafe_allow_html=True,
)
