import logging
import streamlit as st
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import folium
import geopandas as gpd
from pathlib import Path
from folium.plugins import HeatMap

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

# ---------------------------------------------------------------------------
# Cached data loaders — Parquet columns only, no per-row Python loops
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
        # Return only lat/lon for efficient map rendering, not the full frame
        "map_points": (
            df.dropna(subset=["lat", "lon"])[["lat", "lon"]]
              .sample(min(3000, len(df)), random_state=42)
              .values.tolist()          # plain list — JSON-serialisable & fast
            if "lat" in df.columns and "lon" in df.columns else []
        ),
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

# ---------------------------------------------------------------------------
# Backend — cached as a resource (not re-loaded on every rerun)
# ---------------------------------------------------------------------------

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

    /* ── XAI Chat Panel ────────────────────────────────────────────────── */
    .xai-chat-wrap {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        max-height: 340px;
        overflow-y: auto;
        padding-right: 4px;
        scrollbar-width: thin;
        scrollbar-color: #10b981 transparent;
    }
    .xai-bubble {
        padding: 0.55rem 0.8rem;
        border-radius: 10px;
        font-size: 0.78rem;
        line-height: 1.5;
        word-break: break-word;
        animation: fadeIn 0.2s ease;
    }
    .xai-bubble.user {
        background: #1e293b;
        border-left: 3px solid #6366f1;
        margin-left: 1rem;
        color: #e2e8f0;
    }
    .xai-bubble.ai {
        background: #0f2a1e;
        border-left: 3px solid #10b981;
        margin-right: 1rem;
        color: #d1fae5;
    }
    .xai-chip {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 999px;
        font-size: 0.72rem;
        color: #94a3b8;
        cursor: pointer;
        transition: background 0.15s, color 0.15s;
        margin: 2px;
    }
    .xai-chip:hover { background: #10b981; color: #fff; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }
    </style>
    """, unsafe_allow_html=True)

def card(title, body):
    st.markdown(f'''
    <div class="bg-gray-50 dark:bg-slate-900 border border-gray-200 dark:border-slate-800 rounded-lg p-6 mb-4">
        <div class="text-xs font-bold uppercase tracking-wider text-gray-500 dark:text-slate-400 mb-3">{title}</div>
        <div class="text-sm leading-relaxed text-gray-700 dark:text-slate-300 opacity-90">{body}</div>
    </div>
    ''', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Global XAI Chat Panel — rendered into any column / container
# ---------------------------------------------------------------------------

# Suggested template questions shown as clickable chips
XAI_SUGGESTED_QUESTIONS = [
    "Why is the risk high?",
    "What is the biggest factor?",
    "Is this safe to drink?",
    "What is PFOS?",
    "How accurate is this model?",
    "How can risk be reduced?",
    "What does the score mean?",
    "What is PFAS?",
    "What do the map icons mean?",
    "What does the number on the map mean?",
]

def render_xai_panel(xai_engine, *, compact: bool = False):
    """
    Renders the XAI Inquiry Assistant.
    Call this from any page.  Reads/writes st.session_state.chat_history.

    Parameters
    ----------
    xai_engine : XAIEngine | None
    compact    : bool — True means sidebar-style (narrower layout)
    """
    if xai_engine is None:
        st.warning("XAI engine offline.")
        return

    history = st.session_state.get("chat_history", [])

    # ── Chat bubble list ──────────────────────────────────────────────────
    if not history:
        st.markdown(
            '<div class="xai-bubble ai">'
            "PFAS Assistant ready. Run a scan to get context-aware explanations, "
            "or ask a general question below."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        bubble_html = '<div class="xai-chat-wrap">'
        for role, msg in history:
            css_cls = "user" if role == "user" else "ai"
            safe_msg = msg  # Use raw for trusted internal text
            bubble_html += f'<div class="xai-bubble {css_cls}">{safe_msg}</div>'
        bubble_html += "</div>"
        st.markdown(bubble_html, unsafe_allow_html=True)

    # ── Suggested question chips ──────────────────────────────────────────
    st.markdown("<div style='margin-top:0.6rem; margin-bottom:0.2rem;'>", unsafe_allow_html=True)

    if compact:
        # 2 rows of chips in narrow sidebar
        rows = [XAI_SUGGESTED_QUESTIONS[:4], XAI_SUGGESTED_QUESTIONS[4:]]
    else:
        rows = [XAI_SUGGESTED_QUESTIONS]

    for row in rows:
        cols = st.columns(len(row))
        for col, q in zip(cols, row):
            with col:
                if st.button(q, key=f"xai_chip_{q[:20]}", help=q):
                    st.session_state.chat_history = history + [
                        ("user", q),
                        ("ai", xai_engine.chat(q)),
                    ]
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Free-text input ───────────────────────────────────────────────────
    with st.form(key="xai_chat_form", clear_on_submit=True):
        cols_f = st.columns([5, 1])
        with cols_f[0]:
            user_input = st.text_input(
                "Ask anything about this assessment:",
                label_visibility="collapsed",
                placeholder="e.g. Why is the risk high? What is PFOS?",
            )
        with cols_f[1]:
            submitted = st.form_submit_button("Send", use_container_width=True)

    if submitted and user_input.strip():
        st.session_state.chat_history = history + [
            ("user", user_input.strip()),
            ("ai", xai_engine.chat(user_input.strip())),
        ]
        st.rerun()

    # ── Clear chat ────────────────────────────────────────────────────────
    if history:
        if st.button("Clear conversation", key="xai_clear_btn"):
            st.session_state.chat_history = []
            st.rerun()
