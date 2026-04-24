import streamlit as st
import plotly.express as px
from streamlit_folium import st_folium
import folium
from folium.plugins import HeatMap
from common import get_plotly_layout, EMERALD_SEQUENCE, card

def render_overview(summary, hotspots):
    col_map, col_stats = st.columns([2, 1])

    with col_map:
        card("Contamination Density Map", 
             "Spatial distribution of PFAS measurements. Point clusters indicate historical sampling density.")
        
        m = folium.Map(location=[51.0, 10.0], zoom_start=4, tiles="CartoDB dark_matter")
        
        if hotspots is not None and not hotspots.empty:
            heat_data = [[r["lat"], r["lon"], max(float(r["gi_zscore"]), 0)] for _, r in hotspots.iterrows()]
            HeatMap(heat_data, radius=18, blur=22,
                    gradient={"0.2": "#064e3b", "0.5": "#059669", "0.8": "#10b981", "1.0": "#ffffff"}).add_to(m)
        
        if summary and summary["df_preview"] is not None:
            df = summary["df_preview"]
            sample = df.dropna(subset=["lat", "lon"]).sample(min(3000, len(df)), random_state=42)
            for _, row in sample.iterrows():
                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=2, weight=0, fill=True,
                    fill_color="#10b981", fill_opacity=0.4
                ).add_to(m)
        
        st_folium(m, width='stretch', height=520)

    with col_stats:
        card("Data Composition", "Analysis of records by source and target compounds.")
        
        if summary and not summary["source_mix"].empty:
            df_src = summary["source_mix"].reset_index()
            df_src.columns = ["source", "count"]
            fig_src = px.bar(df_src, x="count", y="source", orientation="h",
                             color_discrete_sequence=["#10b981"])
            fig_src.update_layout(**get_plotly_layout(height=240), showlegend=False,
                                  xaxis_title="Records", yaxis_title="")
            st.plotly_chart(fig_src, width='stretch', theme="streamlit")

        if summary and not summary["top_substances"].empty:
            fig_sub = px.pie(values=summary["top_substances"].values,
                             names=summary["top_substances"].index,
                             color_discrete_sequence=EMERALD_SEQUENCE)
            fig_sub.update_traces(textinfo="percent", hole=0.4)
            fig_sub.update_layout(**get_plotly_layout(height=280))
            st.plotly_chart(fig_sub, width='stretch', theme="streamlit")
