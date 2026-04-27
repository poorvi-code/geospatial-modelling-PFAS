import streamlit as st
import sys
from pathlib import Path

# Add parent to path to allow imports from dashboard/ folder
sys.path.append(str(Path(__file__).resolve().parent))

from common import (  # type: ignore[import]
    load_summary, load_trend_data, load_hotspots, get_backend,
    inject_tailwind, _MODELS_AVAILABLE, render_xai_panel,
)
from page_overview   import render_overview
from page_scanner    import render_scanner
from page_simulation import render_simulation
from page_analysis   import render_analysis
from page_explorer   import render_explorer

# ── Page configuration ────────────────────────────────────────────────────
st.set_page_config(
    page_title="PFAS Risk Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────────
_DEFAULTS = {
    "chat_history": [],
    "scan_result":  None,
    "sim_result":   None,
    "xai_result":   None,
    "selected_tab": "Overview",
    "sidebar_addr_prev": "",   # for geocoder debouncing
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Styles ────────────────────────────────────────────────────────────────
inject_tailwind()

# ── Lazy-load backend once (cached as resource) ───────────────────────────
pred, sim, xai = get_backend()

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="text-2xl font-bold mb-6 text-emerald-500 tracking-tighter">PFAS ENGINE</div>',
                unsafe_allow_html=True)

    # --- Geocoder (only fires when address actually changes) ----------
    st.markdown('<div class="text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">Location Search</div>',
                unsafe_allow_html=True)
    addr = st.text_input("Address", placeholder="e.g. Brussels", label_visibility="collapsed", key="sidebar_search")

    if addr and addr != st.session_state.sidebar_addr_prev:
        # Debounce: only geocode when the text has changed
        st.session_state.sidebar_addr_prev = addr
        try:
            from geopy.geocoders import Nominatim
            geolocator = Nominatim(user_agent="pfas_risk_dashboard_v5")
            loc = geolocator.geocode(addr)
            if loc:
                st.info(f"{loc.address[:45]}…")
                st.session_state["preset_lat"] = loc.latitude
                st.session_state["preset_lon"] = loc.longitude
            else:
                st.warning("Location not found.")
        except Exception:
            st.warning("Search error — check your connection.")

    st.markdown('<div class="my-4 border-t border-slate-800"></div>', unsafe_allow_html=True)

    # --- Navigation ---------------------------------------------------
    st.markdown('<div class="text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">Navigation</div>',
                unsafe_allow_html=True)
    for lbl in ["Overview", "Scanner", "Simulation", "Analysis", "Explorer"]:
        if st.button(lbl, use_container_width=True, key=f"nav_{lbl}"):
            st.session_state.selected_tab = lbl
            st.rerun()

    st.markdown('<div class="my-4 border-t border-slate-800"></div>', unsafe_allow_html=True)

    # --- System status ------------------------------------------------
    st.markdown('<div class="text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">System Status</div>',
                unsafe_allow_html=True)
    status_color = "text-emerald-500" if pred else "text-red-500"
    status_text  = "Operational" if pred else "Offline"
    st.markdown(
        f'<div class="flex items-center space-x-2">'
        f'<div class="w-2 h-2 rounded-full bg-current {status_color}"></div>'
        f'<div class="text-sm {status_color}">{status_text}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # --- Sidebar XAI mini-panel (compact chip strip) ------------------
    if xai:
        st.markdown('<div class="my-4 border-t border-slate-800"></div>', unsafe_allow_html=True)
        st.markdown('<div class="text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">Quick Questions</div>',
                    unsafe_allow_html=True)
        from common import XAI_SUGGESTED_QUESTIONS
        for q in XAI_SUGGESTED_QUESTIONS[:4]:
            if st.button(q, key=f"sidebar_chip_{q[:18]}", use_container_width=True):
                st.session_state.chat_history = st.session_state.chat_history + [
                    ("user", q),
                    ("ai", xai.chat(q)),
                ]
                # Switch to Analysis tab to see the full conversation
                st.session_state.selected_tab = "Analysis"
                st.rerun()

# ── Hero banner ─────────────────────────────────────────────────────────
st.markdown("""
<div class="bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-12 mb-10 relative">
    <div class="text-emerald-500 text-[10px] font-bold tracking-[0.3em] uppercase mb-6 flex items-center">
        <span class="w-8 h-[1px] bg-emerald-500 mr-3"></span>
        Research Intelligence
    </div>
    <h1 class="text-4xl font-black mb-6 tracking-tight text-slate-900 dark:text-white">PFAS Risk Analysis Framework</h1>
    <p class="text-slate-600 dark:text-slate-400 max-w-2xl text-lg leading-relaxed font-medium">
        Geospatial predictive modeling for Polyfluoroalkyl substances.
        Leveraging high-fidelity environmental telemetry to map chemical prevalence and simulate exposure scenarios.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Page routing ──────────────────────────────────────────────────────────
tab = st.session_state.selected_tab

if tab == "Overview":
    sum_data = load_summary()
    hot_data = load_hotspots()
    render_overview(sum_data, hot_data, xai_engine=xai)

elif tab == "Scanner":
    if pred:
        render_scanner(pred, xai)
    else:
        st.error("Engine offline — model files not found. Run `python main.py` first.")

elif tab == "Simulation":
    if pred and sim:
        from common import SCENARIO_PRESETS
        render_simulation(pred, sim, SCENARIO_PRESETS, xai_engine=xai)
    else:
        st.error("Engine offline.")

elif tab == "Analysis":
    if xai:
        render_analysis(xai)
    else:
        st.error("Engine offline.")

elif tab == "Explorer":
    df = load_trend_data()
    render_explorer(df, xai_engine=xai)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown('<div class="my-16 border-t border-slate-100 dark:border-slate-800"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="text-center text-[10px] text-slate-400 uppercase tracking-widest font-bold opacity-50 pb-10">
    PFAS Research Platform — V5.1 &nbsp;·&nbsp; Non-Regulatory Screening Only
</div>
""", unsafe_allow_html=True)
