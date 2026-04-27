from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st # noqa: E402
import plotly.graph_objects as go # noqa: E402
from common import get_plotly_layout, card, render_xai_panel # type: ignore[import]



def _gauge_chart(value, title="Risk Score", color="#10b981"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 14}},
        number={"font": {"size": 32}, "suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(128,128,128,0.05)",
            "steps": [
                {"range": [0,  20],  "color": "rgba(16, 185, 129, 0.05)"},
                {"range": [20, 60],  "color": "rgba(245, 158, 11, 0.05)"},
                {"range": [60, 100], "color": "rgba(239, 68, 68, 0.05)"},
            ],
        },
    ))
    fig.update_layout(get_plotly_layout(height=240))
    return fig


def render_scanner(predictor, xai_engine):
    col_form, col_result = st.columns([1, 1.4])

    with col_form:
        card("Location & Parameters", "Specify coordinates for site-specific risk assessment.")
        scan_lat  = st.number_input("Latitude",  value=float(st.session_state.get("preset_lat", 51.5)), format="%.5f", step=0.1)
        scan_lon  = st.number_input("Longitude", value=float(st.session_state.get("preset_lon", -0.12)), format="%.5f", step=0.1)
        substance = st.selectbox("PFAS Compound", ["GENERAL (Total)"] + ["PFOS", "PFOA", "PFHXS", "PFNA", "PFDA", "PFHPA", "PFBS"])
        
        with st.expander("Advanced Parameters", expanded=False):
            year = st.slider("Assessment Year", 2001, 2030, 2024)
            media = st.selectbox("Media Type", ["Surface Water", "Groundwater", "Drinking Water", "Soil", "Sediment", "Wastewater"])

        run_btn   = st.button("Run Risk Assessment", use_container_width=True)

    with col_result:
        if run_btn or st.session_state.scan_result:
            if run_btn:
                with st.spinner("Running analysis…"):
                    try:
                        clean_sub = substance.split(" ")[0] # handle 'GENERAL (Total)'
                        res = predictor.predict(scan_lat, scan_lon, substance=clean_sub, year=year, media_type=media)
                        st.session_state.scan_result = res
                        st.session_state.scan_lat    = scan_lat
                        st.session_state.scan_lon    = scan_lon
                        if xai_engine:
                            X_feat, _, _ = predictor.build_feature_frame(scan_lat, scan_lon, substance=clean_sub, year=year, media_type=media)
                            xai_res = xai_engine.explain(
                                X_feat,
                                res["exceedance_prob"],
                                res["predicted_value_ngl"],
                                clean_sub,
                                res["dist_to_nearest_sample_km"],
                            )
                            st.session_state.xai_result = xai_res
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

            res = st.session_state.scan_result
            if res:
                prob  = res["exceedance_prob"]
                color = "#ef4444" if prob >= 0.65 else "#f59e0b" if prob >= 0.35 else "#10b981"
                st.plotly_chart(_gauge_chart(prob * 100, color=color), use_container_width=True, theme="streamlit")

                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Risk Probability",  f"{prob*100:.1f}%")
                col_m2.metric("Est. Conc.",   f"{res['predicted_value_ngl']:.1f} ng/L")
                col_m3.metric("Nearest Sample", f"{res['dist_to_nearest_sample_km']:.0f} km")
                col_m4.metric("Airport Prox.",  f"{res['dist_to_airport_km']:.0f} km")

                if st.session_state.xai_result:
                    card("Analysis Summary", st.session_state.xai_result.headline)

    # ── XAI Panel below the scanner ───────────────────────────────────────
    st.markdown("""
    <div class="mt-6 mb-2 text-xs font-bold text-slate-500 uppercase tracking-widest">
        AI Inquiry Assistant
    </div>
    """, unsafe_allow_html=True)
    render_xai_panel(xai_engine)
