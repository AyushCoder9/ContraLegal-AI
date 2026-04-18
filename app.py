
import os
import tempfile
import streamlit as st
from src.utils.pdf_annotator import highlight_contract_risks
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
def cleanup_files():
    try:
        if os.path.exists("marked_contract.pdf"):
            os.remove("marked_contract.pdf")
        if "original_pdf_path" in st.session_state and st.session_state.original_pdf_path:
            old_path = st.session_state.original_pdf_path
            if os.path.exists(old_path):
                os.remove(old_path)
            st.session_state.original_pdf_path = None
            
        st.session_state.pdf_ready = False 
        
    except Exception as e:
        print(f"Silent Cleanup: {e}")

st.set_page_config(
    page_title="ContraLegal - Risk Dashboard",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
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

    /* ---- Sidebar nav buttons ---- */
    .nav-active {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        color: #fff !important;
        border: none !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

if "analyzed_df" not in st.session_state:
    st.session_state.analyzed_df = None
if "contract_risk" not in st.session_state:
    st.session_state.contract_risk = None
if "summary_clauses" not in st.session_state:
    st.session_state.summary_clauses = None
if "current_view" not in st.session_state:
    st.session_state.current_view = "dashboard"

# GenAI state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_chain" not in st.session_state:
    st.session_state.chat_chain = None
if "raw_contract_text" not in st.session_state:
    st.session_state.raw_contract_text = None
if "llm_instance" not in st.session_state:
    st.session_state.llm_instance = None
if "contract_summary" not in st.session_state:
    st.session_state.contract_summary = ""
if "risk_brief" not in st.session_state:
    st.session_state.risk_brief = ""
if "explain_results" not in st.session_state:
    st.session_state.explain_results = {}
if "rewrite_results" not in st.session_state:
    st.session_state.rewrite_results = {}


with st.sidebar:
    st.markdown("### ContraLegal")
    st.caption("Intelligent Contract Risk Analysis")
    st.markdown("---")

    # AI Settings — render FIRST so widget values are committed before nav buttons
    with st.expander("AI Settings", expanded=False):
        llm_provider = st.selectbox(
            "LLM Provider",
            ["Google Gemini (Free)", "Groq (Free)", "OpenAI"],
            index=0,
            key="llm_provider_select",
        )

        provider_map = {
            "Google Gemini (Free)": ("gemini", "GOOGLE_API_KEY"),
            "Groq (Free)": ("groq", "GROQ_API_KEY"),
            "OpenAI": ("openai", "OPENAI_API_KEY"),
        }
        provider_key, env_key_name = provider_map[llm_provider]
        st.session_state._provider_key = provider_key
        env_key = os.environ.get(env_key_name, "")

        # Seed the widget key from env var on first load
        if "api_key_input" not in st.session_state and env_key:
            st.session_state.api_key_input = env_key

        st.text_input(
            "API Key",
            type="password",
            key="api_key_input",
        )
        st.caption("Press Enter to save")

    # Read state AFTER widgets have rendered
    ai_enabled = bool(st.session_state.get("api_key_input", ""))
    has_analysis = st.session_state.analyzed_df is not None

    if ai_enabled:
        st.success("AI enabled")
    else:
        st.info("No API key set")

    st.markdown("---")

    # Navigation
    if st.button("Risk Dashboard", use_container_width=True,
                 type="primary" if st.session_state.current_view == "dashboard" else "secondary"):
        st.session_state.current_view = "dashboard"
        st.rerun()

    if ai_enabled and has_analysis:
        if st.button("AI Assistant", use_container_width=True,
                     type="primary" if st.session_state.current_view == "assistant" else "secondary"):
            st.session_state.current_view = "assistant"
            st.rerun()
    elif not ai_enabled and has_analysis:
        st.caption("Add an API key to unlock AI Assistant")

# Rebuild LLM when provider or key changes
provider_key = st.session_state.get("_provider_key", "gemini")
api_key = st.session_state.get("api_key_input", "")
ai_enabled = bool(api_key)
_current_llm_id = f"{provider_key}:{api_key}"
if ai_enabled and st.session_state.get("_llm_id") != _current_llm_id:
    from src.inference.llm_engine import get_llm, create_chat_chain
    try:
        llm = get_llm(provider=provider_key, api_key=api_key)
        st.session_state.llm_instance = llm
        st.session_state._llm_id = _current_llm_id
        if st.session_state.vector_store is not None:
            st.session_state.chat_chain = create_chat_chain(
                st.session_state.vector_store, llm,
                st.session_state.get("contract_summary", ""),
                st.session_state.get("risk_brief", ""),
            )
    except Exception as e:
        st.sidebar.warning(f"Could not connect to {llm_provider}: {e}")


vectorizer, model = load_model()

if vectorizer is None or model is None:
    st.error(
        "**Model not found.** Please train the model first by running:\n\n"
        "```bash\npython -m src.model_trainer\n```\n\n"
        "This will generate `models/vectorizer.pkl` and `models/model.pkl`.",
    )
    st.stop()


st.markdown('<div class="brand-name">ContraLegal</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="brand-tagline">Intelligent Contract Risk Analysis</div>',
    unsafe_allow_html=True,
)

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


if analyze_btn_upload or analyze_btn_paste or analyze_btn_demo:
    clauses = []

    if analyze_btn_demo:
        from src.utils.dummy_contract import DEMO_CONTRACT_TEXT
        st.session_state.raw_contract_text = DEMO_CONTRACT_TEXT.strip()
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
            st.session_state.original_pdf_path = tmp_path

            try:
                from src.data_pipeline.pdf_extractor import PDFExtractor
                st.session_state.raw_contract_text = PDFExtractor().extract_text(tmp_path)
                pipeline = DataPipeline()
                clauses = pipeline.process_document(tmp_path)
            except Exception as e:
                st.error(f"Error processing PDF: {e}")

    elif analyze_btn_paste and pasted_text.strip():
        st.session_state.raw_contract_text = pasted_text.strip()
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
        st.session_state.pdf_ready = False

        # Build RAG index for the new document
        if ai_enabled and st.session_state.raw_contract_text:
            with st.spinner("Building AI knowledge base..."):
                from src.inference.llm_engine import (
                    build_vector_store, create_chat_chain, generate_contract_summary,
                    generate_risk_brief, get_llm,
                )
                try:
                    vs = build_vector_store(st.session_state.raw_contract_text)
                    st.session_state.vector_store = vs
                    _pk = st.session_state.get("_provider_key", "gemini")
                    _ak = st.session_state.get("api_key_input", "")
                    llm = get_llm(provider=_pk, api_key=_ak)
                    st.session_state.llm_instance = llm
                    summary = generate_contract_summary(
                        st.session_state.raw_contract_text, llm,
                    )
                    st.session_state.contract_summary = summary
                    brief = generate_risk_brief(r_df, c_risk)
                    st.session_state.risk_brief = brief
                    st.session_state.chat_chain = create_chat_chain(
                        vs, llm, summary, brief,
                    )
                except Exception as e:
                    st.warning(f"AI features unavailable: {e}")
            # Reset caches for new document
            st.session_state.chat_history = []
            st.session_state.explain_results = {}
            st.session_state.rewrite_results = {}

        st.session_state.current_view = "dashboard"
        st.rerun()


if st.session_state.analyzed_df is not None and st.session_state.current_view == "dashboard":
    results_df = st.session_state.analyzed_df
    contract_risk = st.session_state.contract_risk
    summary_clauses = st.session_state.summary_clauses

    render_contract_risk_banner(contract_risk)
    render_executive_summary(summary_clauses)

    st.markdown('<hr class="clean-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Risk Dashboard</div>', unsafe_allow_html=True)

    render_metrics(contract_risk)

    col_chart, col_heatmap = st.columns([1, 1.5])

    with col_chart:
        st.markdown('<div class="section-title">Risk Distribution</div>', unsafe_allow_html=True)
        fig_pie = render_pie_chart(results_df)
        st.plotly_chart(fig_pie, width="stretch")

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
    dl_col1, dl_col2, dl_col3= st.columns(3)

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
    with dl_col3:
        if "pdf_ready" not in st.session_state:
            st.session_state.pdf_ready = False

        if st.session_state.get("original_pdf_path") and os.path.exists(st.session_state.original_pdf_path):

            output_pdf_path = "marked_contract.pdf"

            risky_data = [
                {"text": row["clause_text"], "risk": row["risk_label"]}
                for _, row in results_df[
                    results_df["risk_label"].isin(["High Risk", "Medium Risk"])
                ].iterrows()
            ]

            if not st.session_state.pdf_ready:
                if st.button("Generate Marked-up PDF", use_container_width=True):
                    with st.spinner("Applying spatial highlights..."):
                        try:
                            highlight_contract_risks(
                                st.session_state.original_pdf_path,
                                output_pdf_path,
                                risky_data
                            )
                            st.session_state.pdf_ready = True
                            st.success("Highlights applied successfully!")
                        except Exception as e:
                            st.error("Could not apply highlights. PDF format may not be supported.")
                            print(e)

            if st.session_state.pdf_ready:
                with open(output_pdf_path, "rb") as f:
                    st.download_button(
                        label="Download Highlighted PDF",
                        data=f,
                        file_name="ContraLegal_Spatial_Analysis.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        on_click=cleanup_files 
                    )
        else:
            st.caption("Upload a PDF to enable highlighted export.")

elif st.session_state.analyzed_df is not None and st.session_state.current_view == "assistant":
    results_df = st.session_state.analyzed_df

    st.markdown('<div class="section-title">AI Contract Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">'
        "Analyze risky clauses and ask questions about your contract."
        "</div>",
        unsafe_allow_html=True,
    )

    #Clause analyzer
    st.markdown('<hr class="clean-divider">', unsafe_allow_html=True)
    st.markdown("#### Clause Analyzer")
    st.caption("Select a risky clause to get an AI explanation or a fairer rewrite.")

    risky = results_df[results_df["risk_label"].isin(["High Risk", "Medium Risk"])].copy()

    if risky.empty:
        st.info("No high or medium risk clauses found in this contract.")
    else:
        clause_options = {}
        for _, row in risky.iterrows():
            label_tag = "HIGH" if row["risk_label"] == "High Risk" else "MED"
            preview = row["clause_text"][:80].replace("\n", " ")
            clause_options[f"[{label_tag}] {preview}..."] = int(row["clause_index"])

        selected_label = st.selectbox(
            "Select a clause to analyze",
            list(clause_options.keys()),
            label_visibility="collapsed",
        )
        selected_idx = clause_options[selected_label]
        selected_row = risky[risky["clause_index"] == selected_idx].iloc[0]

        # Show full clause text
        with st.container(border=True):
            risk_color = "#ef4444" if selected_row["risk_label"] == "High Risk" else "#f59e0b"
            st.markdown(
                f'<span style="color:{risk_color}; font-weight:700; font-size:0.75rem; '
                f'text-transform:uppercase; letter-spacing:0.08em;">'
                f'{selected_row["risk_label"]}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(selected_row["clause_text"])

        col1, col2 = st.columns(2)
        with col1:
            explain_btn = st.button("Explain Risk", use_container_width=True, type="primary")
        with col2:
            rewrite_btn = st.button("Rewrite Clause", use_container_width=True, type="primary")

        if explain_btn and st.session_state.llm_instance:
            with st.spinner("Analyzing clause..."):
                from src.inference.llm_engine import explain_clause
                st.session_state.explain_results[selected_idx] = explain_clause(
                    selected_row["clause_text"],
                    selected_row["risk_label"],
                    selected_row.get("keyword_flags", []),
                    st.session_state.llm_instance,
                )

        if rewrite_btn and st.session_state.llm_instance:
            with st.spinner("Rewriting clause..."):
                from src.inference.llm_engine import rewrite_clause
                st.session_state.rewrite_results[selected_idx] = rewrite_clause(
                    selected_row["clause_text"],
                    selected_row["risk_label"],
                    selected_row.get("keyword_flags", []),
                    st.session_state.llm_instance,
                )

        if selected_idx in st.session_state.explain_results:
            with st.expander("AI Risk Explanation", expanded=True):
                st.markdown(st.session_state.explain_results[selected_idx])

        if selected_idx in st.session_state.rewrite_results:
            with st.expander("AI Suggested Rewrite", expanded=True):
                st.markdown(st.session_state.rewrite_results[selected_idx])

    #Chat with Contract
    if st.session_state.chat_chain is not None:
        st.markdown('<hr class="clean-divider">', unsafe_allow_html=True)
        st.markdown("#### Chat with Your Contract")
        st.caption(
            "Ask questions about your contract. The AI retrieves relevant "
            "sections and answers based on the actual document text."
        )

        # Render chat history
        for human_msg, ai_msg in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(human_msg)
            with st.chat_message("assistant"):
                st.markdown(ai_msg)

        # Chat input
        if user_question := st.chat_input("Ask about your contract..."):
            with st.chat_message("user"):
                st.markdown(user_question)
            with st.chat_message("assistant"):
                with st.spinner("Searching contract..."):
                    from src.inference.llm_engine import ask_question
                    try:
                        answer = ask_question(
                            st.session_state.chat_chain,
                            user_question,
                            st.session_state.chat_history,
                        )
                        st.markdown(answer)
                        st.session_state.chat_history.append((user_question, answer))
                    except Exception as e:
                        st.error(f"AI error: {e}")

        # Suggested questions (only when chat is empty)
        if not st.session_state.chat_history:
            st.markdown("**Try asking:**")
            suggestions = [
                "What are the termination conditions?",
                "Who owns the intellectual property?",
                "What are the payment terms?",
                "Are there any non-compete clauses?",
            ]
            cols = st.columns(2)
            for i, suggestion in enumerate(suggestions):
                with cols[i % 2]:
                    if st.button(suggestion, key=f"suggest_{i}"):
                        from src.inference.llm_engine import ask_question
                        try:
                            answer = ask_question(
                                st.session_state.chat_chain,
                                suggestion,
                                st.session_state.chat_history,
                            )
                            st.session_state.chat_history.append((suggestion, answer))
                            st.rerun()
                        except Exception as e:
                            st.error(f"AI error: {e}")

#no ananlysis yet
elif st.session_state.analyzed_df is None:
    st.markdown(
        '<div style="text-align:center; color:#aaa; padding:4rem 0; font-size:1rem;">'
        "Upload a PDF or paste contract text above, then click <strong>Analyse Risk</strong>."
        "</div>",
        unsafe_allow_html=True,
    )


