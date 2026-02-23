import os
import tempfile

import streamlit as st

from src.data_pipeline.pipeline import DataPipeline
from src.inference.predictor import (
    build_xlsx_export,
    compute_contract_risk,
    generate_clause_clusters,
    load_model,
    predict_hybrid,
    summarize_contract,
)
from src.ui.components import (
    render_contract_risk_banner,
    render_executive_summary,
    render_keyword_frequency,
    render_metrics,
    render_pie_chart,
    render_risk_card,
    render_risk_heatmap,
)

st.set_page_config(
    page_title="ContraLegal-AI · Risk Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }

    .risk-card {
        padding: 1rem 1.25rem;
        border-radius: 10px;
        margin-bottom: 0.75rem;
        border-left: 5px solid;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .risk-high   { background: #fecaca; border-color: #dc2626; color: #7f1d1d; }
    .risk-medium { background: #ffedd5; border-color: #ea580c; color: #7c2d12; }
    .risk-low    { background: #bbf7d0; border-color: #16a34a; color: #14532d; }

    .risk-label {
        font-weight: 700;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }

    section[data-testid="stSidebar"] {
        background: #1e1e2f;
        color: #e0e0e0;
    }

    .header-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0rem;
    }
    .header-sub {
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        flex: 1;
        padding: 1.25rem;
        border-radius: 12px;
        text-align: center;
    }
    .metric-card h2 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-card p {
        margin: 0.25rem 0 0 0;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .exec-summary-card {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.25);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
    }
    .exec-summary-card h3 {
        font-size: 1.2rem;
        font-weight: 700;
        color: #6366f1;
        margin-bottom: 1rem;
    }
    .exec-clause {
        padding: 0.6rem 1rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #a855f7;
        background: rgba(168, 85, 247, 0.06);
        border-radius: 0 8px 8px 0;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚖️ ContraLegal-AI")
    st.divider()

    uploaded_file = st.file_uploader("📄 Upload a Contract (PDF)", type=["pdf"])

    st.divider()
    st.markdown("#### 🔬 Or paste text directly")
    pasted_text = st.text_area("Paste contract clauses below:", height=200)

    st.divider()
    analyze_btn = st.button("🚀  Analyse Risk", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="header-title">ContraLegal-AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="header-sub">Intelligent Contract Risk Analysis · Milestone 1 Dashboard</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Model load
# ---------------------------------------------------------------------------
vectorizer, model = load_model()

if vectorizer is None or model is None:
    st.warning(
        "⚠️  **Model not found.** Please train the model first by running:\n\n"
        "```bash\npython -m src.model_trainer\n```\n\n"
        "This will generate `models/vectorizer.pkl` and `models/model.pkl`.",
        icon="🔧",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------
if "analyzed_df" not in st.session_state:
    st.session_state.analyzed_df = None
if "contract_risk" not in st.session_state:
    st.session_state.contract_risk = None
if "summary_clauses" not in st.session_state:
    st.session_state.summary_clauses = None

if analyze_btn:
    clauses = []
    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF…"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            try:
                pipeline = DataPipeline()
                clauses = pipeline.process_document(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    elif pasted_text.strip():
        clauses = [
            line.strip()
            for line in pasted_text.strip().split("\n")
            if len(line.strip()) > 10
        ]
    else:
        st.error("Please upload a PDF or paste some text in the sidebar.", icon="📝")

    if clauses:
        with st.spinner("Analysing clauses…"):
            r_df = predict_hybrid(clauses, vectorizer, model)
            r_df["risk_label"] = r_df["final_score"].apply(
                lambda s: "High Risk" if s >= 0.5 else ("Medium Risk" if s >= 0.3 else "Low Risk")
            )
            c_risk = compute_contract_risk(r_df)
            s_clauses = summarize_contract(clauses, top_n=5)
            
            st.session_state.analyzed_df = r_df
            st.session_state.contract_risk = c_risk
            st.session_state.summary_clauses = s_clauses

# ---------------------------------------------------------------------------
# Analysis & rendering
# ---------------------------------------------------------------------------
if st.session_state.analyzed_df is not None:
    results_df = st.session_state.analyzed_df
    contract_risk = st.session_state.contract_risk
    summary_clauses = st.session_state.summary_clauses

    # -- Contract-level risk banner
    render_contract_risk_banner(contract_risk)

    # -- Executive summary
    render_executive_summary(summary_clauses)

    st.markdown("---")
    st.markdown("### 📊 Risk Dashboard")

    render_metrics(contract_risk)

    col_chart, col_heatmap = st.columns([1, 1.5])

    with col_chart:
        st.markdown("#### Risk Distribution")
        fig_pie = render_pie_chart(results_df)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_heatmap:
        st.markdown("#### Top 10 Riskiest Clauses")
        fig_heat = render_risk_heatmap(results_df, top_n=10)
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")
    st.markdown("### Keyword Frequency Analysis")
    st.caption("How many clauses each risk keyword appears in across the full contract.")
    render_keyword_frequency(contract_risk.get("keyword_frequency", []))

    st.markdown("---")
    st.markdown("### 📝 Clause-Level Analysis")
    st.caption("Clauses are ranked highest-risk first. Keywords are highlighted in red.")

    filter_col, _ = st.columns([1, 3])
    with filter_col:
        risk_filter = st.multiselect(
            "Filter by risk level:",
            options=["High Risk", "Medium Risk", "Low Risk"],
            default=["High Risk", "Medium Risk", "Low Risk"],
        )

    filtered = results_df[results_df["risk_label"].isin(risk_filter)]

    for _, row in filtered.iterrows():
        kw_matches = row["keyword_matches"] if "keyword_matches" in row else []
        render_risk_card(row["clause_text"], row["risk_label"], kw_matches)

    st.markdown("---")
    st.markdown("### 🧩 Thematic Theme Analysis")
    st.markdown("*(Unsupervised Clause Clustering via K-Means)*")

    with st.spinner("Discovering themes…"):
        cluster_labels, cluster_headings = generate_clause_clusters(
            results_df["clause_text"].tolist(), vectorizer
        )
        results_df["cluster"] = cluster_labels

    for c_id in sorted(results_df["cluster"].unique()):
        cluster_clauses = results_df[results_df["cluster"] == c_id]
        heading = cluster_headings.get(c_id, f"Theme {c_id + 1}")
        with st.expander(f"📁 Theme: {heading} ({len(cluster_clauses)} clauses)"):
            for _, row in cluster_clauses.iterrows():
                st.markdown(f"- {row['clause_text']}")

    st.markdown("---")
    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        csv = results_df.drop(columns=["keyword_matches"], errors="ignore").to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥  Download Clause Report (CSV)",
            data=csv,
            file_name="contralegal_clause_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with dl_col2:
        xlsx_bytes = build_xlsx_export(results_df, contract_risk)
        st.download_button(
            "📊  Download Full Report (Excel)",
            data=xlsx_bytes,
            file_name="contralegal_risk_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

else:
    st.info(
        "👈 **Upload a PDF** or **paste contract text** in the sidebar, then click **Analyse Risk**.",
        icon="📄",
    )
