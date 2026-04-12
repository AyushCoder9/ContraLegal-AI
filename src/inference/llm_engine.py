"""
ContraLegal-AI — LLM Engine (RAG + Generative AI)

Provides:
  - RAG pipeline: chunk contract text, embed, store in FAISS, retrieve
  - Contract summary: one-time extraction of key metadata (parties, type, domain)
  - Contract chatbot: conversational Q&A grounded in the contract
  - Clause explainer: explains WHY a clause is risky
  - Clause rewriter: suggests safer alternative language
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd

import streamlit as st
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.messages import AIMessage, HumanMessage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CHAT_HISTORY = 10

# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------
EXPLAIN_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are ContraLegal AI, an expert legal contract analyst specializing in "
     "risk assessment. Your job is to explain why a contract clause is risky in "
     "clear, accessible language that a non-lawyer can understand.\n\n"
     "Structure your response as:\n"
     "1. **Risk Summary** (1-2 sentences): What is the core risk?\n"
     "2. **Why It's Dangerous** (2-3 bullet points): Specific consequences\n"
     "3. **Legal Context**: How this compares to standard/fair contract practices\n"
     "4. **Who It Hurts**: Which party is disadvantaged and how"),
    ("human",
     'Analyze this contract clause that has been flagged as "{risk_label}":\n\n'
     'CLAUSE TEXT:\n"{clause_text}"\n\n'
     "RISK KEYWORDS DETECTED: {keyword_flags}\n\n"
     "Explain why this clause is risky and what a signer should be concerned about."),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are ContraLegal AI, an expert contract drafter. Your job is to rewrite "
     "risky contract clauses into fairer, more balanced alternatives that protect "
     "both parties while preserving the original business intent.\n\n"
     "Your response MUST follow this exact format:\n"
     "1. **Rewritten Clause**: The full revised clause text\n"
     "2. **Changes Made** (bullet points): What you changed and why each change matters"),
    ("human",
     'Rewrite this "{risk_label}" contract clause to be fairer and more balanced:\n\n'
     'ORIGINAL CLAUSE:\n"{clause_text}"\n\n'
     "RISK KEYWORDS DETECTED: {keyword_flags}\n\n"
     "Provide a balanced alternative that a reasonable party on either side would accept."),
])

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are ContraLegal AI. Extract key metadata from this contract text. "
     "Be concise — one line per field. If a field is not found, write 'Not specified'."),
    ("human",
     "Extract the following from this contract:\n\n"
     "1. **Contract Type**: (e.g., Master Services Agreement, NDA, Employment Contract)\n"
     "2. **Parties**: Who are the parties involved? List their names/roles.\n"
     "3. **Domain/Industry**: (e.g., Technology, Healthcare, Finance)\n"
     "4. **Effective Date**: When does the contract start?\n"
     "5. **Term/Duration**: How long does it last?\n"
     "6. **Key Subject**: What is this contract primarily about? (1 sentence)\n\n"
     "CONTRACT TEXT (first 3000 characters):\n{contract_excerpt}"),
])

CHAT_SYSTEM_TEMPLATE = (
    "You are ContraLegal AI, an expert legal contract analyst. "
    "You answer questions about a specific contract uploaded by the user.\n\n"
    "CONTRACT SUMMARY:\n{contract_summary}\n\n"
    "RISK ANALYSIS RESULTS:\n{risk_brief}\n\n"
    "RULES:\n"
    "- Use ONLY the context below and the risk analysis above to answer.\n"
    "- If the answer is not in the context, say: \"I couldn't find that in the contract.\"\n"
    "- Always cite section numbers when available.\n"
    "- Never contradict yourself across answers.\n"
    "- Identify the parties by their actual names, not generic terms.\n"
    "- When discussing risks, reference the specific risk scores and flagged keywords from the analysis.\n"
    "- Flag any risks you notice in the relevant text.\n\n"
    "CONTEXT FROM CONTRACT:\n{context}"
)


# ---------------------------------------------------------------------------
# LLM Provider Factory
# ---------------------------------------------------------------------------
def get_llm(provider: str = "gemini", api_key: str = "", model_name: str | None = None, temperature: float = 0.3):
    """Return a LangChain chat model instance."""
    if not api_key:
        raise ValueError("API key is required. Please provide it in the sidebar.")

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name or "gemini-2.5-flash",
            google_api_key=api_key,
            temperature=temperature,
            convert_system_message_to_human=True,
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model_name or "llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=temperature,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name or "gpt-4o-mini",
            api_key=api_key,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


# ---------------------------------------------------------------------------
# Embedding & Vector Store
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFaceEmbeddings instance (runs locally)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(contract_text: str) -> FAISS:
    """Chunk the full contract text and build a FAISS vector store."""
    if not contract_text or not contract_text.strip():
        raise ValueError("Contract text is empty — nothing to index.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(contract_text)
    embeddings = get_embeddings()
    return FAISS.from_texts(chunks, embeddings)


# ---------------------------------------------------------------------------
# Contract Summary
# ---------------------------------------------------------------------------
def generate_contract_summary(contract_text: str, llm) -> str:
    """Generate a concise metadata summary of the contract (parties, type, domain)."""
    excerpt = contract_text[:3000]
    chain = SUMMARY_PROMPT | llm
    result = chain.invoke({"contract_excerpt": excerpt})
    return result.content


# ---------------------------------------------------------------------------
# Risk Brief
# ---------------------------------------------------------------------------
def generate_risk_brief(analyzed_df: pd.DataFrame, contract_risk: dict) -> str:
    """Format risk analysis results into a text brief for the chat prompt."""
    lines = [
        f"Overall Risk Score: {contract_risk.get('overall_score', 0):.2f} ({contract_risk.get('risk_label', 'Unknown')})",
        f"High Risk Clauses: {contract_risk.get('high_risk_count', 0)}",
        f"Medium Risk Clauses: {contract_risk.get('medium_risk_count', 0)}",
        f"Low Risk Clauses: {contract_risk.get('low_risk_count', 0)}",
        f"Top Risk Keywords: {', '.join(contract_risk.get('top_keywords', []))}",
        "",
        "FLAGGED HIGH RISK CLAUSES:",
    ]

    high_risk = analyzed_df[analyzed_df["risk_label"] == "High Risk"].sort_values(
        "final_score", ascending=False,
    )
    for i, (_, row) in enumerate(high_risk.head(10).iterrows(), 1):
        text = row["clause_text"][:100].replace("\n", " ")
        flags = row.get("keyword_flags", [])
        kw = ", ".join(flags) if isinstance(flags, list) and flags else "—"
        lines.append(f"{i}. [Score: {row['final_score']:.2f}] \"{text}...\" (Keywords: {kw})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# RAG Chat
# ---------------------------------------------------------------------------
def create_chat_chain(
    vector_store: FAISS,
    llm,
    contract_summary: str = "",
    risk_brief: str = "",
) -> ConversationalRetrievalChain:
    """Build a conversational retrieval chain for contract Q&A."""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6},
    )

    system_template = CHAT_SYSTEM_TEMPLATE.replace(
        "{contract_summary}", contract_summary or "No summary available."
    ).replace(
        "{risk_brief}", risk_brief or "No risk analysis available."
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        verbose=False,
        chain_type="stuff",
        combine_docs_chain_kwargs={
            "prompt": ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template("{question}"),
            ])
        },
    )


def ask_question(
    chain: ConversationalRetrievalChain,
    question: str,
    chat_history: List[Tuple[str, str]],
) -> str:
    """Send a question to the RAG chain and return the answer."""
    # Limit history to last N exchanges to prevent context overflow
    recent_history = chat_history[-MAX_CHAT_HISTORY:]

    lc_history = []
    for human, ai in recent_history:
        lc_history.append(HumanMessage(content=human))
        lc_history.append(AIMessage(content=ai))

    result = chain.invoke({"question": question, "chat_history": lc_history})
    return result["answer"]


# ---------------------------------------------------------------------------
# Clause Explainer & Rewriter
# ---------------------------------------------------------------------------
def explain_clause(clause_text: str, risk_label: str, keyword_flags: List[str], llm) -> str:
    """Explain WHY a clause is risky in plain English."""
    chain = EXPLAIN_PROMPT | llm
    result = chain.invoke({
        "clause_text": clause_text,
        "risk_label": risk_label,
        "keyword_flags": ", ".join(keyword_flags) if keyword_flags else "None",
    })
    return result.content


def rewrite_clause(clause_text: str, risk_label: str, keyword_flags: List[str], llm) -> str:
    """Suggest a fairer, safer alternative version of a risky clause."""
    chain = REWRITE_PROMPT | llm
    result = chain.invoke({
        "clause_text": clause_text,
        "risk_label": risk_label,
        "keyword_flags": ", ".join(keyword_flags) if keyword_flags else "None",
    })
    return result.content
