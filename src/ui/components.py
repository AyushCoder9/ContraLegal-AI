from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def _highlight_keywords(clause: str, kw_matches: List[dict]) -> str:
    if not kw_matches:
        return clause

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
        result.append(clause[prev:start])
        opacity = 0.25 + 0.5 * weight
        result.append(
            f'<mark style="background: rgba(239,68,68,{opacity:.2f}); '
            f'border-radius:3px; padding:0 2px;">'
            f"{clause[start:end]}</mark>"
        )
        prev = end
    result.append(clause[prev:])
    return "".join(result)


def render_contract_risk_banner(contract_risk: dict):
    overall = contract_risk["overall_score"]
    label = contract_risk["risk_label"]
    highest_clause = contract_risk["highest_risk_clause"]
    highest_score = contract_risk["highest_risk_score"]
    top_kws = contract_risk.get("top_keywords", [])

    if label == "High Risk":
        bg, border, text = "#fecaca", "#dc2626", "#7f1d1d"
    elif label == "Medium Risk":
        bg, border, text = "#ffedd5", "#ea580c", "#7c2d12"
    else:
        bg, border, text = "#bbf7d0", "#16a34a", "#14532d"


    kw_pills = "".join(
        f'<span style="background:rgba(0,0,0,0.08); border-radius:12px; '
        f'padding:2px 10px; margin-right:6px; font-size:0.8rem;">{kw}</span>'
        for kw in top_kws
    )

    st.markdown(
        f"""
        <div style="background:{bg}; border-left:6px solid {border}; color:{text};
                    border-radius:12px; padding:1.25rem 1.75rem; margin-bottom:1.5rem;">
            <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.5rem;">
                <div>
                    <div style="font-size:1.4rem; font-weight:800;">
                        Overall Contract Risk: {label}
                        <span style="font-size:1rem; font-weight:600; margin-left:0.5rem;">
                            ({overall:.0%})
                        </span>
                    </div>
                </div>
            </div>
            <div style="font-size:0.88rem; margin-bottom:0.5rem;">
                <strong>Most Frequent Risk Keywords:</strong>&nbsp;{kw_pills if kw_pills else "—"}
            </div>
            <div style="font-size:0.88rem; opacity:0.85; border-top:1px solid rgba(0,0,0,0.1);
                        padding-top:0.5rem; margin-top:0.5rem;">
                <strong>Highest-Risk Clause ({highest_score:.0%}):</strong>
                &nbsp;{highest_clause[:180]}{"…" if len(highest_clause) > 180 else ""}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def render_metrics(contract_risk: dict):
    high = contract_risk["high_risk_count"]
    medium = contract_risk["medium_risk_count"]
    low = contract_risk["low_risk_count"]
    total = high + medium + low

    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-card" style="background:#fecaca; color:#991b1b;">
                <h2>{high}</h2><p>High Risk</p>
            </div>
            <div class="metric-card" style="background:#ffedd5; color:#9a3412;">
                <h2>{medium}</h2><p>Medium Risk</p>
            </div>
            <div class="metric-card" style="background:#bbf7d0; color:#166534;">
                <h2>{low}</h2><p>Low Risk</p>
            </div>
            <div class="metric-card" style="background:#e0e7ff; color:#3730a3;">
                <h2>{total}</h2><p>Total Clauses</p>
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
        "Medium Risk": "#f97316",
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


def render_risk_heatmap(results_df: pd.DataFrame, top_n: int = 10):
    top = results_df.nlargest(top_n, "final_score").copy()
    top["label"] = top["clause_text"].str.slice(0, 60) + "…"

    fig = go.Figure(
        go.Bar(
            x=top["final_score"],
            y=top["label"],
            orientation="h",
            marker=dict(
                color=top["final_score"],
                colorscale=[[0, "#22c55e"], [0.5, "#f59e0b"], [1, "#ef4444"]],
                showscale=True,
                colorbar=dict(title="Risk Score", thickness=12),
            ),
            text=[f"{s:.0%}" for s in top["final_score"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis=dict(range=[0, 1], tickformat=".0%", title="Risk Score"),
        margin=dict(l=10, r=80, t=10, b=10),
        height=max(280, top_n * 36),
    )
    return fig


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
                colorscale=[[0, "#bbf7d0"], [0.5, "#fef08a"], [1, "#ef4444"]],
                showscale=False,
            ),
            text=counts,
            textposition="outside",
        )
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis=dict(title="Clause Count"),
        margin=dict(l=10, r=40, t=10, b=10),
        height=max(260, len(terms) * 32),
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
            <div class="risk-label">{label}</div>
            {highlighted}
        </div>
        """,
        unsafe_allow_html=True,
    )
