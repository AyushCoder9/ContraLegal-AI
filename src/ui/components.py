import streamlit as st
import pandas as pd
import plotly.express as px
from typing import List

def render_executive_summary(summary_clauses: List[dict]):
    items_html = "".join(
        f'<div class="exec-clause">{item["clause_text"]}</div>'
        for item in summary_clauses
    )
    st.markdown(
        f"""
        <div class="exec-summary-card">
            <h3>Executive Summary — Top {len(summary_clauses)} Critical Clauses</h3>
            {items_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_card(clause: str, label: str):
    css_class = ""
    if label == "High Risk":
        css_class = "risk-high"
    elif label == "Low Risk":
        css_class = "risk-low"
    else:
        css_class = "risk-low"

    st.markdown(
        f"""
        <div class="risk-card {css_class}">
            <div class="risk-label">{label}</div>
            {clause}
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_metrics(results_df: pd.DataFrame):
    counts = results_df["risk_label"].value_counts()
    
    total = len(results_df)
    high = counts.get("High Risk", 0)
    low = counts.get("Low Risk", 0)

    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-card" style="background: #fecaca; color: #991b1b;">
                <h2>{high}</h2>
                <p>High Risk</p>
            </div>
            <div class="metric-card" style="background: #bbf7d0; color: #166534;">
                <h2>{low}</h2>
                <p>Low Risk</p>
            </div>
            <div class="metric-card" style="background: #e0e7ff; color: #3730a3;">
                <h2>{total}</h2>
                <p>Total Clauses</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_pie_chart(results_df: pd.DataFrame):
    dist = results_df["risk_label"].value_counts().reset_index()
    dist.columns = ["Risk Level", "Count"]

    color_map = {
        "High Risk": "#ef4444",
        "Low Risk": "#22c55e",
    }

    fig = px.pie(
        dist,
        names="Risk Level",
        values="Count",
        color="Risk Level",
        color_discrete_map=color_map,
        hole=0.45,
    )
    
    fig.update_traces(textinfo="percent+label", textfont_size=14)
    fig.update_layout(
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        height=350,
    )
    return fig
