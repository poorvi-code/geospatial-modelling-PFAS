import streamlit as st
import plotly.express as px
from common import get_plotly_layout, EMERALD_SEQUENCE

def render_explorer(trend_df):
    if trend_df.empty:
        st.warning("Dataset not available.")
        return

    st.markdown("#### Concentration Trends (2001-2024)")
    yearly = trend_df.groupby(["year", "substance"])["log_value"].median().reset_index()
    yearly.columns = ["Year", "Compound", "Median Log Conc."]
    fig_trend = px.line(yearly, x="Year", y="Median Log Conc.", color="Compound", markers=True,
                        color_discrete_sequence=EMERALD_SEQUENCE)
    fig_trend.update_layout(**get_plotly_layout(height=400))
    st.plotly_chart(fig_trend, width='stretch', theme="streamlit")

    cl, cr = st.columns(2)
    with cl:
        st.markdown("#### Exceedance by Compound")
        exc_rate = trend_df.groupby("substance")["above_100_ng_l"].mean().reset_index()
        exc_rate.columns = ["Compound", "Rate"]
        fig_exc = px.bar(exc_rate.sort_values("Rate", ascending=False), x="Compound", y="Rate", color_discrete_sequence=["#10b981"])
        fig_exc.update_layout(**get_plotly_layout(height=300), yaxis_tickformat=".0%")
        st.plotly_chart(fig_exc, width='stretch', theme="streamlit")
    with cr:
        st.markdown("#### Distribution Analysis")
        fig_violin = px.violin(trend_df.sample(min(5000, len(trend_df))), x="substance", y="log_value", box=True,
                               color_discrete_sequence=["#059669"])
        fig_violin.update_layout(**get_plotly_layout(height=300))
        st.plotly_chart(fig_violin, width='stretch', theme="streamlit")
