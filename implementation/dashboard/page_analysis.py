from __future__ import annotations

import sys
from pathlib import Path

# Allow the type-checker and any direct invocation to resolve sibling modules.
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from common import get_plotly_layout, card, render_xai_panel  # type: ignore[import]



def render_analysis(xai_engine):
    xai_res = st.session_state.xai_result

    if xai_res is None:
        st.info("Run a Scanner assessment first to view factor analysis.")
        # Still surface the XAI chat — user can ask general questions
        st.markdown("---")
        st.markdown("Ask the AI assistant a general question while you wait:")
        render_xai_panel(xai_engine)
        return

    # ── SHAP waterfall bar ────────────────────────────────────────────────
    st.markdown("#### Weight of Evidence — Key Risk Drivers")
    shap_df = pd.DataFrame(xai_res.top_features).head(10)
    if not shap_df.empty:
        shap_df["color"] = shap_df["shap"].apply(lambda v: "#ef4444" if v > 0 else "#10b981")
        fig_shap = go.Figure(go.Bar(
            x=shap_df["shap"],
            y=shap_df["label"],
            orientation="h",
            marker_color=shap_df["color"],
            text=shap_df["shap"].apply(lambda v: f"{v:+.3f}"),
            textposition="outside",
        ))
        fig_shap.update_layout(
            get_plotly_layout(height=380),
            title="",
            xaxis_title="SHAP Impact on Risk Score",
            yaxis={"autorange": "reversed"},
        )
        st.plotly_chart(fig_shap, use_container_width=True, theme="streamlit")

    # ── Risk & Protective factor cards ────────────────────────────────────
    col_risk, col_prot = st.columns(2)
    with col_risk:
        card("Risk Drivers",
             "<br>".join(f"▲ {d}" for d in xai_res.risk_drivers) or "None identified.")
    with col_prot:
        card("Protective Factors",
             "<br>".join(f"▼ {d}" for d in xai_res.protective_factors) or "None identified.")

    card("Data Quality", xai_res.data_quality_note)

    # ── XAI Chat ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### AI Inquiry Assistant")
    render_xai_panel(xai_engine)
