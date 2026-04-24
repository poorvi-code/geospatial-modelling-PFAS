import streamlit as st
import sys
from pathlib import Path
from geopy.geocoders import Nominatim

# Add parent to path to allow imports from dashboard/ folder
sys.path.append(str(Path(__file__).resolve().parent))

from common import (
    load_summary, load_trend_data, load_hotspots, get_backend, 
    inject_tailwind, _MODELS_AVAILABLE
)
from page_overview import render_overview
from page_scanner import render_scanner
from page_simulation import render_simulation
from page_analysis import render_analysis
from page_explorer import render_explorer

# Page configuration
st.set_page_config(
    page_title="PFAS Risk Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
for _k in ["chat_history", "scan_result", "sim_result", "xai_result", "selected_tab"]:
    if _k not in st.session_state:
        st.session_state[_k] = "Overview" if _k == "selected_tab" else ([] if _k == "chat_history" else None)

# Inject Tailwind CSS
inject_tailwind()

geolocator = Nominatim(user_agent="pfas_risk_dashboard_v5")

# Helper for Nav buttons
def nav_button(label, icon_text=""):
    active = st.session_state.selected_tab == label
    bg = "bg-emerald-600 shadow-lg shadow-emerald-900/20" if active else "bg-slate-800 hover:bg-slate-700"
    text = "text-white" if active else "text-slate-400"
    if st.sidebar.button(label, width='stretch', key=f"nav_{label}"):
        st.session_state.selected_tab = label
        st.rerun()

# Sidebar
with st.sidebar:
    st.markdown('<div class="text-2xl font-bold mb-6 text-emerald-500 tracking-tighter">PFAS ENGINE</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">Location Search</div>', unsafe_allow_html=True)
    addr = st.text_input("Address", placeholder="e.g. Brussels", label_visibility="collapsed", key="sidebar_search")
    if addr:
        try:
            loc = geolocator.geocode(addr)
            if loc:
                st.info(f"Location: {loc.address[:40]}...")
                st.session_state["preset_lat"] = loc.latitude
                st.session_state["preset_lon"] = loc.longitude
            else:
                st.warning("Not found.")
        except:
            st.warning("Search error.")

    st.markdown('<div class="my-6 border-t border-slate-800"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">Navigation</div>', unsafe_allow_html=True)
    for lbl in ["Overview", "Scanner", "Simulation", "Analysis", "Explorer"]:
        nav_button(lbl)

    st.markdown('<div class="my-6 border-t border-slate-800"></div>', unsafe_allow_html=True)
    
    # Lazy load minimal status info for sidebar
    _p, _s, _x = get_backend()
    st.markdown('<div class="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">System Status</div>', unsafe_allow_html=True)
    status_color = "text-emerald-500" if _p else "text-red-500"
    status_text = "Operational" if _p else "Offline"
    st.markdown(f'<div class="flex items-center space-x-2"><div class="w-2 h-2 rounded-full bg-current {status_color}"></div><div class="text-sm {status_color}">{status_text}</div></div>', unsafe_allow_html=True)

# Hero Section
st.markdown(f"""
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

# Current Tab Rendering
selected_tab = st.session_state.selected_tab

if selected_tab == "Overview":
    sum_data = load_summary()
    hot_data = load_hotspots()
    render_overview(sum_data, hot_data)
elif selected_tab == "Scanner":
    pred, sim, xai = get_backend()
    if pred:
        render_scanner(pred, xai)
    else:
        st.error("Engine offline.")
elif selected_tab == "Simulation":
    pred, sim, xai = get_backend()
    if pred and sim:
        from common import SCENARIO_PRESETS
        render_simulation(pred, sim, SCENARIO_PRESETS)
    else:
        st.error("Engine offline.")
elif selected_tab == "Analysis":
    pred, sim, xai = get_backend()
    if xai:
        render_analysis(xai)
    else:
        st.error("Engine offline.")
elif selected_tab == "Explorer":
    df = load_trend_data()
    render_explorer(df)

st.markdown('<div class="my-16 border-t border-slate-100 dark:border-slate-800"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="text-center text-[10px] text-slate-400 uppercase tracking-widest font-bold opacity-50 pb-10">
    PFAS Research Platform - V5.0 Modular Architecture<br>
    Non-Regulatory screening Only - Grounded in PDH telemetry
</div>
""", unsafe_allow_html=True)
