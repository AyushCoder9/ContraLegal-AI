import html
from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def _highlight_keywords(clause: str, kw_matches: List[dict]) -> str:
    safe_clause = html.escape(clause)
    if not kw_matches:
        return safe_clause

    sorted_matches = sorted(kw_matches, key=lambda m: m["start"])

    merged: List[tuple] = []
    for m in sorted_matches:
        if merged and m["start"] < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], m["end"]), merged[-1][2])
        else:
            merged.append((m["start"], m["end"], m["weight"]))

    result = []
    prev = 0
    for start, end, weight in merged:
        result.append(html.escape(clause[prev:start]))
        opacity = 0.25 + 0.6 * weight
        result.append(
            f'<mark style="background: rgba(239,68,68,{opacity:.2f}); '
            f'color:var(--text-color); border-radius:3px; padding:0 3px;">'
            f"{html.escape(clause[start:end])}</mark>"
        )
        prev = end
    result.append(html.escape(clause[prev:]))
    return "".join(result)


def render_contract_risk_banner(contract_risk: dict):
    overall = contract_risk["overall_score"]
    label = contract_risk["risk_label"]
    highest_clause = html.escape(contract_risk["highest_risk_clause"])
    highest_score = contract_risk["highest_risk_score"]
    top_kws = contract_risk.get("top_keywords", [])

    if label == "High Risk":
        css = "banner-high"
        border_color = "#ef4444"
    elif label == "Medium Risk":
        css = "banner-medium"
        border_color = "#f59e0b"
    else:
        css = "banner-low"
        border_color = "#10b981"

    kw_pills = "".join(
        f'<span class="kw-pill">{html.escape(kw)}</span>'
        for kw in top_kws
    )

    st.markdown(
        f"""
        <div class="risk-banner {css}" style="border-left-color: {border_color};">
            <div style="font-size:1.4rem; font-weight:800; margin-bottom:0.4rem;">
                Overall Contract Risk: {html.escape(label)}
                <span style="font-size:1rem; font-weight:600; margin-left:0.5rem;">
                    ({overall:.0%})
                </span>
            </div>
            <div style="font-size:0.88rem; margin-bottom:0.5rem;">
                <strong>Most Frequent Risk Keywords:</strong>&nbsp;{kw_pills if kw_pills else "—"}
            </div>
            <div style="font-size:0.88rem; opacity:0.85; border-top:1px solid rgba(255,255,255,0.15);
                        padding-top:0.5rem; margin-top:0.5rem;">
                <strong>Highest-Risk Clause ({highest_score:.0%}):</strong>
                &nbsp;{highest_clause[:180]}{"..." if len(highest_clause) > 180 else ""}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_executive_summary(summary_clauses: List[dict]):
    st.markdown(
        '<div class="section-title">Executive Summary — Top Critical Clauses</div>',
        unsafe_allow_html=True,
    )
    for i, item in enumerate(summary_clauses, 1):
        clause_text = item["clause_text"]
        preview = clause_text[:80] + ("..." if len(clause_text) > 80 else "")
        with st.expander(f"Clause {i}: {preview}"):
            st.markdown(clause_text)


def render_metrics(contract_risk: dict):
    high = contract_risk["high_risk_count"]
    medium = contract_risk["medium_risk_count"]
    low = contract_risk["low_risk_count"]
    total = high + medium + low

    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-card">
                <h2 style="color: #ef4444;">{high}</h2><p>High Risk</p>
            </div>
            <div class="metric-card">
                <h2 style="color: #f59e0b;">{medium}</h2><p>Medium Risk</p>
            </div>
            <div class="metric-card">
                <h2 style="color: #10b981;">{low}</h2><p>Low Risk</p>
            </div>
            <div class="metric-card">
                <h2 style="color: #4f46e5;">{total}</h2><p>Total Clauses</p>
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
        "Medium Risk": "#f59e0b",
        "Low Risk": "#10b981",
    }
    fig = px.pie(
        dist,
        names="Risk Level",
        values="Count",
        color="Risk Level",
        color_discrete_map=color_map,
        hole=0.45,
    )
    fig.update_traces(
        textinfo="percent+label",
        textfont_size=13,
        textfont_color="white",
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, sans-serif"),
    )
    return fig


def render_risk_heatmap(results_df: pd.DataFrame, top_n: int = 10):
    top = results_df.nlargest(top_n, "final_score").copy()
    top["label"] = [f"Clause #{i+1}" for i in range(len(top))]
    top["full_text"] = top["clause_text"].str.slice(0, 120) + "..."

    fig = go.Figure(
        go.Bar(
            x=top["final_score"],
            y=top["label"],
            orientation="h",
            marker=dict(
                color=top["final_score"],
                colorscale=[[0, "#10b981"], [0.5, "#f59e0b"], [1, "#ef4444"]],
                showscale=True,
                colorbar=dict(title="Risk", thickness=10),
            ),
            text=[f"{s:.0%}" for s in top["final_score"]],
            textposition="outside",
            textfont=dict(color="gray", size=12),
            customdata=top["full_text"],
            hovertemplate="<b>%{y}</b><br>Risk Score: %{x:.0%}<br><br>%{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis=dict(range=[0, 1], tickformat=".0%", title="Risk Score"),
        margin=dict(l=10, r=80, t=10, b=10),
        height=max(280, top_n * 36),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # -- Interactive Clause Viewer --
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("##### Interactive Clause Viewer")
    st.markdown(
        "<div style='font-size:0.85rem; color:var(--text-color); opacity:0.7; margin-bottom:1rem;'>"
        "Select a clause from the chart above to read its full text and risk details."
        "</div>",
        unsafe_allow_html=True
    )
    
    options = [f"{row['label']} (Score: {row['final_score']:.0%})" for _, row in top.iterrows()]
    selected_option = st.selectbox("Select Clause to view:", options, label_visibility="collapsed")
    
    if selected_option:
        # Extract the label, e.g., "Clause #1"
        selected_label = selected_option.split(" (")[0]
        selected_row = top[top["label"] == selected_label].iloc[0]
        
        st.markdown(
            f"""
            <div style="background: var(--secondary-background-color); border: 1px solid rgba(128,128,128,0.15); 
                        border-radius: 12px; padding: 1.5rem; font-size: 0.95rem; line-height: 1.7; 
                        color: var(--text-color); box-shadow: 0 4px 20px rgba(0,0,0,0.06); margin-top: 1rem;
                        animation: fadeIn 0.4s ease-out;">
                <div style="display: flex; align-items: center; margin-bottom: 1.25rem;">
                    <div style="background: rgba(168, 85, 247, 0.15); color: #a855f7; padding: 4px 12px; 
                                border-radius: 20px; font-weight: 700; font-size: 0.8rem; letter-spacing: 0.05em; 
                                text-transform: uppercase;">
                        {selected_label} Details
                    </div>
                </div>
                <div style="color: var(--text-color); opacity: 0.9;">
                    {html.escape(selected_row['clause_text'])}
                </div>
            </div>
            <style>
                @keyframes fadeIn {{
                    from {{ opacity: 0; transform: translateY(10px); }}
                    to {{ opacity: 1; transform: translateY(0); }}
                }}
            </style>
            """, 
            unsafe_allow_html=True
        )


def render_keyword_frequency(keyword_frequency: List[dict]):
    if not keyword_frequency:
        st.caption("No keywords matched across clauses.")
        return

    terms = [entry["term"] for entry in keyword_frequency]
    counts = [entry["count"] for entry in keyword_frequency]

    fig = go.Figure(
        go.Bar(
            x=counts,
            y=terms,
            orientation="h",
            marker=dict(
                color=counts,
                colorscale=[[0, "#10b981"], [0.5, "#f59e0b"], [1, "#ef4444"]],
                showscale=False,
            ),
            text=counts,
            textposition="outside",
            textfont=dict(color="gray"),
        )
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis=dict(title="Clause Count"),
        margin=dict(l=10, r=40, t=10, b=10),
        height=max(260, len(terms) * 32),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_risk_card(clause: str, label: str, kw_matches: List[dict] | None = None):
    if label == "High Risk":
        css_class = "risk-high"
    elif label == "Medium Risk":
        css_class = "risk-medium"
    else:
        css_class = "risk-low"

    highlighted = _highlight_keywords(clause, kw_matches or [])

    st.markdown(
        f"""
        <div class="risk-card {css_class}">
            <div class="risk-label">{html.escape(label)}</div>
            {highlighted}
        </div>
        """,
        unsafe_allow_html=True,
    )
