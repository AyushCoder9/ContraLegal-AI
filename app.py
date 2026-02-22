import tempfile
import streamlit as st
from src.ui.components import render_risk_card, render_metrics, render_pie_chart, render_executive_summary
from src.inference.predictor import load_model, predict_hybrid, generate_clause_clusters, summarize_contract
from src.data_pipeline.pipeline import DataPipeline

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
    .risk-medium { background: #fef08a; border-color: #ca8a04; color: #713f12; }
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

with st.sidebar:
    st.markdown("## ⚖️ ContraLegal-AI")
    st.divider()

    uploaded_file = st.file_uploader(
        "📄 Upload a Contract (PDF)",
        type=["pdf"],
    )

    st.divider()

    st.markdown("#### 🔬 Or paste text directly")
    pasted_text = st.text_area(
        "Paste contract clauses below:",
        height=200,
    )

    st.divider()
    analyze_btn = st.button("🚀  Analyse Risk", use_container_width=True, type="primary")

st.markdown('<div class="header-title">ContraLegal-AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="header-sub">Intelligent Contract Risk Analysis · Milestone 1 Dashboard</div>',
    unsafe_allow_html=True,
)

vectorizer, model = load_model()

if vectorizer is None or model is None:
    st.warning(
        "⚠️  **Model not found.** Please train the model first by running:\n\n"
        "```bash\npython -m src.model_trainer\n```\n\n"
        "This will generate `models/vectorizer.pkl` and `models/model.pkl`.",
        icon="🔧",
    )
    st.stop()

clauses: list[str] = []

if analyze_btn:
    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF (Data Pipeline)..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                pipeline = DataPipeline()
                clauses = pipeline.process_document(tmp_path)
            finally:
                import os
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
    elif pasted_text.strip():
        raw_lines = pasted_text.strip().split("\n")
        
        parsed_lines = []
        for line in raw_lines:
            cleaned_line = line.strip()
            if len(cleaned_line) > 10:
                parsed_lines.append(cleaned_line)
                
        clauses = parsed_lines
    else:
        st.error("Please upload a PDF or paste some text in the sidebar.", icon="📝")

if clauses:
    with st.spinner("Analysing clauses..."):
        results_df = predict_hybrid(clauses, vectorizer, model)
        results_df["risk_label"] = results_df["final_score"].apply(
            lambda s: "High Risk" if s >= 0.5 else "Low Risk"
        )
        summary_clauses = summarize_contract(clauses, top_n=5)

    render_executive_summary(summary_clauses)

    st.markdown("---")
    st.markdown("### 📊 Risk Dashboard")

    render_metrics(results_df)

    col_chart, col_details = st.columns([1, 1.5])

    with col_chart:
        st.markdown("#### Risk Distribution")
        fig = render_pie_chart(results_df)
        st.plotly_chart(fig, use_container_width=True)

    with col_details:
        st.markdown("#### Risk Breakdown")
        breakdown = results_df["risk_label"].value_counts().reset_index()
        breakdown.columns = ["Risk Level", "Clause Count"]
        st.dataframe(breakdown, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📝 Clause-Level Analysis")

    filter_col1, filter_col2 = st.columns([1, 3])
    with filter_col1:
        risk_filter = st.multiselect(
            "Filter by risk level:",
            options=["High Risk", "Low Risk"],
            default=["High Risk", "Low Risk"],
        )

    filtered = results_df[results_df["risk_label"].isin(risk_filter)]

    for _, row in filtered.iterrows():
        render_risk_card(row["clause_text"], row["risk_label"])

    st.markdown("---")
    st.markdown("### 🧩 Thematic Theme Analysis")
    st.markdown("*(Unsupervised Clause Clustering via K-Means)*")
    
    with st.spinner("Discovering themes..."):
        cluster_labels, cluster_headings = generate_clause_clusters(results_df["clause_text"].tolist(), vectorizer)
        results_df["cluster"] = cluster_labels
        
    unique_clusters = sorted(results_df["cluster"].unique())
    
    for c_id in unique_clusters:
        cluster_clauses = results_df[results_df["cluster"] == c_id]
        heading = cluster_headings.get(c_id, f"Theme {c_id + 1}")
        
        with st.expander(f"📁 Theme: {heading} ({len(cluster_clauses)} clauses)"):
            for _, row in cluster_clauses.iterrows():
                st.markdown(f"- {row['clause_text']}")

    st.markdown("---")
    csv = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥  Download Full Report (CSV)",
        data=csv,
        file_name="contralegal_risk_report.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.info(
        "👈 **Upload a PDF** or **paste contract text** in the sidebar, then click **Analyse Risk**.",
        icon="📄",
    )
