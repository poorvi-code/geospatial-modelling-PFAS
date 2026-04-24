import streamlit as st
import numpy as np
import plotly.graph_objects as go
from common import get_plotly_layout, card

def _gauge_chart(value, title="Simulated Risk", color="#10b981"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 14}},
        number={"font": {"size": 32}, "suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(128,128,128,0.05)",
        },
    ))
    fig.update_layout(**get_plotly_layout(height=240))
    return fig

def render_simulation(predictor, sim_engine, presets):
    if st.session_state.scan_result is None:
        st.info("Run a Scanner analysis first to establish a baseline for simulation.")
        return

    res = st.session_state.scan_result
    card("Scenario Simulation Laboratory", f"Baseline risk: {res['exceedance_prob']*100:.1f}%. Test changes to observe impact.")

    col_preset, col_sliders, col_outcome = st.columns([1, 1.2, 1.2])

    with col_preset:
        st.markdown("#### Presets")
        preset_map = {v["label"]: k for k, v in presets.items()}
        chosen_label = st.radio("Select Scenario", list(preset_map.keys()), label_visibility="collapsed")
        chosen_key = preset_map[chosen_label]
        
        # Clean description too
        desc = presets[chosen_key]["description"]
        card("Scenario Description", desc)

    with col_sliders:
        st.markdown("#### Parameters")
        slider_mods = presets[chosen_key]["mods"].copy()
        
        ind_pct = st.slider("Industrial Intensity", 0, 300, 100, format="%d%%")
        slider_mods["spatial_density_boost"] = ind_pct / 100.0

        airport_km = st.slider("Airport Proximity (km)", 1, 300, int(np.clip(res.get("dist_to_airport_km", 50), 1, 300)))
        slider_mods["airport_distance_km"] = float(airport_km)

        cleanup_pct = st.slider("Intervention Efficiency", 0, 90, 0, format="%d%%")
        if cleanup_pct > 0:
            slider_mods["mean_log_value_reduction"] = cleanup_pct / 100.0

        if st.button("Execute Simulation", width='stretch'):
            X_feat, _, _ = predictor.build_feature_frame(st.session_state.get("scan_lat", 51.5), st.session_state.get("scan_lon", -0.12))
            st.session_state.sim_result = sim_engine.run_custom(X_feat, slider_mods, label=chosen_label)

    with col_outcome:
        st.markdown("#### Outcome")
        sim_res = st.session_state.sim_result
        if sim_res:
            d = sim_res.delta_pts
            color = "#ef4444" if d > 5 else "#10b981" if d < -5 else "#f59e0b"
            st.plotly_chart(_gauge_chart(sim_res.scenario_score, color=color), width='stretch', theme="streamlit")
            st.metric("Risk Variance", f"{d:+.1f}%", delta=f"{'Increase' if d>0 else 'Reduction'}", delta_color="inverse")
            card("Outcome Narrative", sim_res.plain_explanation)
