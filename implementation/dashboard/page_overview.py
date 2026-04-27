from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st # noqa: E402
import plotly.express as px # noqa: E402
from streamlit_folium import st_folium # noqa: E402
import folium # noqa: E402
from folium.plugins import HeatMap, FastMarkerCluster # noqa: E402
from common import get_plotly_layout, EMERALD_SEQUENCE, card, render_xai_panel # type: ignore[import]



def render_overview(summary, hotspots, xai_engine=None):
    col_map, col_stats = st.columns([2, 1])

    with col_map:
        card("Contamination Density Map",
             "Spatial distribution of PFAS measurements. Point clusters indicate historical sampling density.")

        m = folium.Map(location=[51.0, 10.0], zoom_start=4, tiles="CartoDB dark_matter")

        if hotspots is not None and not hotspots.empty:
            heat_data = [[r["lat"], r["lon"], max(float(r["gi_zscore"]), 0)] for _, r in hotspots.iterrows()]
            HeatMap(heat_data, radius=18, blur=22,
                    gradient={"0.2": "#064e3b", "0.5": "#059669", "0.8": "#10b981", "1.0": "#ffffff"}).add_to(m)

        if summary and summary["map_points"]:
            # FastMarkerCluster is a single JS call — orders of magnitude faster
            # than looping CircleMarker in Python
            FastMarkerCluster(summary["map_points"]).add_to(m)

        st_folium(m, width="100%", height=520)

    with col_stats:
        card("Data Composition", "Analysis of records by source and target compounds.")

        if summary and not summary["source_mix"].empty:
            df_src = summary["source_mix"].reset_index()
            df_src.columns = ["source", "count"]
            fig_src = px.bar(df_src, x="count", y="source", orientation="h",
                             color_discrete_sequence=["#10b981"])
            fig_src.update_layout(get_plotly_layout(height=240), showlegend=False,
                                  xaxis_title="Records", yaxis_title="")
            st.plotly_chart(fig_src, use_container_width=True, theme="streamlit")

        if summary and not summary["top_substances"].empty:
            fig_sub = px.pie(values=summary["top_substances"].values,
                             names=summary["top_substances"].index,
                             color_discrete_sequence=EMERALD_SEQUENCE)
            fig_sub.update_traces(textinfo="percent", hole=0.4)
            fig_sub.update_layout(get_plotly_layout(height=280))
            st.plotly_chart(fig_sub, use_container_width=True, theme="streamlit")

    # ── XAI Panel below the map grid ──────────────────────────────────────
    st.markdown("""
    <div class="mt-6 mb-2 text-xs font-bold text-slate-500 uppercase tracking-widest">
        AI Inquiry Assistant
    </div>
    """, unsafe_allow_html=True)
    render_xai_panel(xai_engine)
