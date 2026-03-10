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
    page_title="ContraLegal - Risk Dashboard",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }

    .main .block-container {
        padding-top: 2.5rem;
        max-width: 1100px;
    }

    /* ---- Brand ---- */
    .brand-name {
        font-size: 2.6rem;
        font-weight: 900;
        background: linear-gradient(135deg, #4f46e5, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
        margin-bottom: 0;
        line-height: 1.1;
        text-shadow: 0 4px 20px rgba(168, 85, 247, 0.2);
    }
    .brand-tagline {
        font-size: 1rem;
        color: var(--text-color);
        opacity: 0.7;
        font-weight: 400;
        margin-bottom: 2rem;
        letter-spacing: 0.02em;
    }

    /* ---- Input card ---- */
    .input-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }

    /* ---- Section headers ---- */
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-color);
        letter-spacing: -0.01em;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    .section-caption {
        font-size: 0.85rem;
        color: var(--text-color);
        opacity: 0.6;
        margin-bottom: 1rem;
    }

    /* ---- Metric cards ---- */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        flex: 1;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(128,128,128,0.15);
        background: var(--background-color);
        transition: transform 0.3s cubic-bezier(0.25, 0.8, 0.25, 1), box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    }
    .metric-card h2 {
        margin: 0;
        font-size: 2rem;
        font-weight: 800;
    }
    .metric-card p {
        margin: 0.25rem 0 0 0;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-color);
        opacity: 0.7;
    }

    /* ---- Risk cards ---- */
    .risk-card {
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 0.75rem;
        border-left: 4px solid;
        font-size: 0.95rem;
        line-height: 1.7;
        background: var(--secondary-background-color);
        color: var(--text-color);
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .risk-card:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .risk-high {
        border-color: #ef4444;
    }
    .risk-medium {
        border-color: #f59e0b;
    }
    .risk-low {
        border-color: #10b981;
    }
    .risk-label {
        font-weight: 700;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.25rem;
    }

    /* ---- Banner ---- */
    .risk-banner {
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        border-left: 6px solid;
        transition: all 0.3s ease;
    }
    .banner-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(0,0,0,0));
        border-color: #ef4444;
        color: var(--text-color);
        box-shadow: 0 8px 24px rgba(239, 68, 68, 0.1);
    }
    .banner-medium {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(0,0,0,0));
        border-color: #f59e0b;
        color: var(--text-color);
        box-shadow: 0 8px 24px rgba(245, 158, 11, 0.1);
    }
    .banner-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(0,0,0,0));
        border-color: #10b981;
        color: var(--text-color);
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.1);
    }

    /* ---- Keyword pills ---- */
    .kw-pill {
        background: rgba(128,128,128,0.1);
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 16px;
        padding: 4px 12px;
        margin-right: 8px;
        margin-bottom: 8px;
        font-size: 0.85rem;
        display: inline-block;
        transition: all 0.2s ease;
    }
    .kw-pill:hover {
        background: rgba(128,128,128,0.2);
        transform: scale(1.05);
    }

    /* ---- Divider ---- */
    .clean-divider {
        border: none;
        border-top: 1px solid #e5e5e5;
        margin: 2rem 0;
    }

    /* ---- Hide sidebar toggle ---- */
    [data-testid="collapsedControl"] {
        display: none;
    }

    /* ---- Download buttons ---- */
    .stDownloadButton > button {
        background: #1a1a1a !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stDownloadButton > button:hover {
        background: #333 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 15px rgba(0,0,0,0.2) !important;
    }

    /* ---- Primary button override ---- */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.4) !important;
    }

    /* ---- Tab underline override ---- */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #4f46e5 !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--text-color) !important;
        opacity: 0.6 !important;
        font-weight: 500 !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--text-color) !important;
        opacity: 1.0 !important;
        font-weight: 700 !important;
    }

    /* ---- Hero Feature Cards ---- */
    .hero-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 2.5rem;
    }
    .hero-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,0.15);
        border-radius: 16px;
        padding: 1.75rem;
        text-align: left;
        transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
        backdrop-filter: blur(10px);
    }
    .hero-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.12);
        border-color: rgba(168, 85, 247, 0.4);
    }
    .hero-card h3 {
        margin-top: 0;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-color);
    }
    .hero-card p {
        margin: 0;
        font-size: 0.9rem;
        color: var(--text-color);
        opacity: 0.7;
        line-height: 1.5;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Brand header
# ---------------------------------------------------------------------------
st.markdown('<div class="brand-name">ContraLegal</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="brand-tagline">Intelligent Contract Risk Analysis</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Model load
# ---------------------------------------------------------------------------
vectorizer, model = load_model()

if vectorizer is None or model is None:
    st.error(
        "**Model not found.** Please train the model first by running:\n\n"
        "```bash\npython -m src.model_trainer\n```\n\n"
        "This will generate `models/vectorizer.pkl` and `models/model.pkl`.",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Input section & Hero
# ---------------------------------------------------------------------------
if "analyzed_df" not in st.session_state:
    st.session_state.analyzed_df = None
if "contract_risk" not in st.session_state:
    st.session_state.contract_risk = None
if "summary_clauses" not in st.session_state:
    st.session_state.summary_clauses = None

# Show Hero Section only if no analysis has been done yet
if st.session_state.analyzed_df is None:
    st.markdown(
        """
        <div class="hero-grid">
            <div class="hero-card">
                <h3>AI-Powered Analysis</h3>
                <p>Upload any legal contract and our hybrid NLP models will instantly extract and analyze every single clause.</p>
            </div>
            <div class="hero-card">
                <h3>Instant Risk Scoring</h3>
                <p>Identify critical liabilities, unfair termination rights, and hidden financial risks before you sign.</p>
            </div>
            <div class="hero-card">
                <h3>Clause-by-Clause Context</h3>
                <p>Don't just get a score. See exactly which keywords triggered the risk, highlighted directly in the text.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

tab_upload, tab_paste, tab_demo = st.tabs(["Upload PDF", "Paste Text", "Try Demo"])

with tab_upload:
    uploaded_file = st.file_uploader(
        "Select a contract file",
        type=["pdf"],
        label_visibility="collapsed",
    )
    analyze_btn_upload = st.button("Analyse PDF", use_container_width=True, type="primary")

with tab_paste:
    pasted_text = st.text_area(
        "Paste contract clauses below",
        height=180,
        label_visibility="collapsed",
        placeholder="Paste your contract clauses here, one per line...",
    )
    analyze_btn_paste = st.button("Analyse Text", use_container_width=True, type="primary")

with tab_demo:
    st.markdown("**(Demo Mode)** Don't have a contract? Click below to instantly load and analyze a sample Master Services Agreement containing several risky clauses.")
    analyze_btn_demo = st.button("Load & Analyze Demo Contract", use_container_width=True, type="primary")

st.markdown('<hr class="clean-divider">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------
if analyze_btn_upload or analyze_btn_paste or analyze_btn_demo:
    clauses = []
    
    if analyze_btn_demo:
        from src.utils.dummy_contract import DEMO_CONTRACT_TEXT
        clauses = [
            line.strip()
            for line in DEMO_CONTRACT_TEXT.strip().split("\n")
            if len(line.strip()) > 10
        ]
        
    elif analyze_btn_upload and uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            try:
                pipeline = DataPipeline()
                clauses = pipeline.process_document(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    elif analyze_btn_paste and pasted_text.strip():
        clauses = [
            line.strip()
            for line in pasted_text.strip().split("\n")
            if len(line.strip()) > 10
        ]
    else:
        st.error("Please provide an input or use the demo.")

    if clauses:
        with st.spinner("Analysing clauses..."):
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

    # -- Executive summary (collapsible)
    render_executive_summary(summary_clauses)

    st.markdown('<hr class="clean-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Risk Dashboard</div>', unsafe_allow_html=True)

    render_metrics(contract_risk)

    col_chart, col_heatmap = st.columns([1, 1.5])

    with col_chart:
        st.markdown('<div class="section-title">Risk Distribution</div>', unsafe_allow_html=True)
        fig_pie = render_pie_chart(results_df)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_heatmap:
        st.markdown('<div class="section-title">Top 10 Riskiest Clauses</div>', unsafe_allow_html=True)
        render_risk_heatmap(results_df)

    st.markdown('<hr class="clean-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Keyword Frequency Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">How many clauses each risk keyword appears in across the full contract.</div>', unsafe_allow_html=True)
    render_keyword_frequency(contract_risk.get("keyword_frequency", []))

    st.markdown('<hr class="clean-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Clause-Level Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Select a risk level to view the corresponding clauses.</div>',
        unsafe_allow_html=True,
    )

    risk_tabs = st.tabs(["High Risk", "Medium Risk", "Low Risk"])

    for idx, level in enumerate(["High Risk", "Medium Risk", "Low Risk"]):
        with risk_tabs[idx]:
            filtered = results_df[results_df["risk_label"] == level]
            if filtered.empty:
                st.caption(f"No {level.lower()} clauses found.")
            else:
                for _, row in filtered.iterrows():
                    kw_matches = row["keyword_matches"] if "keyword_matches" in row else []
                    render_risk_card(row["clause_text"], row["risk_label"], kw_matches)

    st.markdown('<hr class="clean-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Thematic Clustering</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Unsupervised clause grouping via K-Means</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Discovering themes..."):
        cluster_labels, cluster_headings = generate_clause_clusters(
            results_df["clause_text"].tolist(), vectorizer
        )
        results_df["cluster"] = cluster_labels

    for c_id in sorted(results_df["cluster"].unique()):
        cluster_clauses = results_df[results_df["cluster"] == c_id]
        heading = cluster_headings.get(c_id, f"Theme {c_id + 1}")
        with st.expander(f"Theme: {heading}  ({len(cluster_clauses)} clauses)"):
            for _, row in cluster_clauses.iterrows():
                st.markdown(f"- {row['clause_text']}")

    st.markdown('<hr class="clean-divider">', unsafe_allow_html=True)
    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        csv = results_df.drop(columns=["keyword_matches"], errors="ignore").to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Clause Report (CSV)",
            data=csv,
            file_name="contralegal_clause_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with dl_col2:
        xlsx_bytes = build_xlsx_export(results_df, contract_risk)
        st.download_button(
            "Download Full Report (Excel)",
            data=xlsx_bytes,
            file_name="contralegal_risk_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

else:
    st.markdown(
        '<div style="text-align:center; color:#aaa; padding:4rem 0; font-size:1rem;">'
        "Upload a PDF or paste contract text above, then click <strong>Analyse Risk</strong>."
        "</div>",
        unsafe_allow_html=True,
    )
