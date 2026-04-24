import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from common import get_plotly_layout, card

def render_analysis(xai_engine):
    xai_res = st.session_state.xai_result
    if xai_res is None:
        st.info("Run a Scanner assessment to view factor analysis.")
        return

    col_xai, col_chat = st.columns([1, 1])
    
    with col_xai:
        st.markdown("#### Weight of Evidence")
        shap_df = pd.DataFrame(xai_res.top_features).head(10)
        if not shap_df.empty:
            shap_df["color"] = shap_df["shap"].apply(lambda v: "#ef4444" if v > 0 else "#10b981")
            fig_shap = go.Figure(go.Bar(
                x=shap_df["shap"], y=shap_df["label"], orientation="h",
                marker_color=shap_df["color"],
            ))
            fig_shap.update_layout(**get_plotly_layout(height=400), title="Key Risk Drivers", xaxis_title="Impact on Risk Score")
            st.plotly_chart(fig_shap, width='stretch', theme="streamlit")

    with col_chat:
        st.markdown("#### Inquiry Assistant")
        for role, msg in st.session_state.chat_history[-6:]:
            bg = "bg-slate-50 dark:bg-slate-800" if role == "ai" else "bg-indigo-50 dark:bg-slate-900 border-l-4 border-indigo-400"
            align = "mr-12" if role == "ai" else "ml-12"
            st.markdown(f'<div class="{bg} p-4 rounded mb-2 text-sm {align}">{msg}</div>', unsafe_allow_html=True)
        
        with st.form("chat_form", clear_on_submit=True):
            u_msg = st.text_input("Ask a question about this assessment:", label_visibility="collapsed")
            if st.form_submit_button("Submit", width='stretch'):
                if u_msg.strip():
                    st.session_state.chat_history.append(("user", u_msg))
                    st.session_state.chat_history.append(("ai", xai_engine.chat(u_msg)))
                    st.rerun()
