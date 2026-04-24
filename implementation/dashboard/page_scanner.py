import streamlit as st
import plotly.graph_objects as go
from common import get_plotly_layout, card

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
    fig.update_layout(**get_plotly_layout(height=240))
    return fig

def render_scanner(predictor, xai_engine):
    col_form, col_result = st.columns([1, 1.4])

    with col_form:
        card("Location & Parameters", "Specify coordinates for site-specific risk assessment.")
        scan_lat = st.number_input("Latitude",  value=float(st.session_state.get("preset_lat", 51.5)), format="%.5f", step=0.1)
        scan_lon = st.number_input("Longitude", value=float(st.session_state.get("preset_lon", -0.12)), format="%.5f", step=0.1)
        substance = st.selectbox("PFAS Compound", ["PFOS", "PFOA", "PFHXS", "PFNA", "PFDA", "PFHPA", "PFBS"])
        run_btn = st.button("Run Risk Assessment", width='stretch')

    with col_result:
        if run_btn or st.session_state.scan_result:
            if run_btn:
                try:
                    res = predictor.predict(scan_lat, scan_lon, substance=substance)
                    st.session_state.scan_result = res
                    st.session_state.scan_lat    = scan_lat
                    st.session_state.scan_lon    = scan_lon
                    if xai_engine:
                        X_feat, _, _ = predictor.build_feature_frame(scan_lat, scan_lon, substance=substance)
                        xai_res = xai_engine.explain(X_feat, res["exceedance_prob"], res["predicted_value_ngl"], substance, res["dist_to_nearest_sample_km"])
                        st.session_state.xai_result = xai_res
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

            res = st.session_state.scan_result
            if res:
                prob = res["exceedance_prob"]
                st.plotly_chart(_gauge_chart(prob * 100), width='stretch', theme="streamlit")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Probability", f"{prob*100:.1f}%")
                m2.metric("Est. Conc.",  f"{res['predicted_value_ngl']:.1f} ng/L")
                m3.metric("Proximity", f"{res['dist_to_nearest_sample_km']:.0f} km")

                if st.session_state.xai_result:
                    card("Analysis Summary", st.session_state.xai_result.headline)
