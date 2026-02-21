"""
Streamlit UI for Global-Context-Aware RAG.

Uses local_context_rag.LocalContextRAG for all operations. Supports:
Upload PDF, Ask Question, Knowledge Base viewer, Manual Search Vector DB,
and configurable params (chroma_path, knowledge_path, model, collection_name).
"""

import tempfile
from pathlib import Path

import streamlit as st

from local_context_rag import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_KNOWLEDGE_PATH,
    DEFAULT_MODEL,
    LocalContextRAG,
)


def _init_session_state() -> None:
    if "chroma_path" not in st.session_state:
        st.session_state.chroma_path = DEFAULT_CHROMA_PATH
    if "knowledge_path" not in st.session_state:
        st.session_state.knowledge_path = DEFAULT_KNOWLEDGE_PATH
    if "model" not in st.session_state:
        st.session_state.model = DEFAULT_MODEL
    if "collection_name" not in st.session_state:
        st.session_state.collection_name = DEFAULT_COLLECTION_NAME


def _get_rag() -> LocalContextRAG | None:
    """Get or create RAG instance; return None if creation fails."""
    if "rag" in st.session_state:
        return st.session_state.rag
    try:
        rag = LocalContextRAG(
            chroma_path=st.session_state.chroma_path,
            knowledge_path=st.session_state.knowledge_path,
            model=st.session_state.model,
            collection_name=st.session_state.collection_name,
        )
        st.session_state.rag = rag
        return rag
    except Exception as e:
        st.sidebar.error(f"Could not create RAG: {e}")
        return None


def main() -> None:
    st.set_page_config(page_title="Global-Context-Aware RAG", layout="wide")
    _init_session_state()

    # Sidebar: Settings
    st.sidebar.title("Settings")
    chroma_path = st.sidebar.text_input("Chroma path", value=st.session_state.chroma_path, key="sidebar_chroma_path")
    knowledge_path = st.sidebar.text_input("Knowledge path", value=st.session_state.knowledge_path, key="sidebar_knowledge_path")
    model = st.sidebar.text_input("Ollama model", value=st.session_state.model, key="sidebar_model")
    collection_name = st.sidebar.text_input("Collection name", value=st.session_state.collection_name, key="sidebar_collection_name")

    if st.sidebar.button("Apply"):
        st.session_state.chroma_path = chroma_path.strip() or DEFAULT_CHROMA_PATH
        st.session_state.knowledge_path = knowledge_path.strip() or DEFAULT_KNOWLEDGE_PATH
        st.session_state.model = model.strip() or DEFAULT_MODEL
        st.session_state.collection_name = collection_name.strip() or DEFAULT_COLLECTION_NAME
        if "rag" in st.session_state:
            del st.session_state["rag"]
        st.sidebar.success("Settings applied. RAG will be re-created on next use.")
        st.rerun()

    # Main: Tabs
    tab_upload, tab_ask, tab_kb, tab_search = st.tabs(["Upload PDF", "Ask Question", "Knowledge Base", "Search Vector DB"])

    with tab_upload:
        st.header("Upload PDF")
        uploaded = st.file_uploader("Choose a PDF file", type=["pdf"], key="upload_pdf")
        if uploaded is not None and st.button("Ingest", key="ingest_btn"):
            rag = _get_rag()
            if rag is None:
                st.error("RAG not available. Check Settings.")
            else:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded.getvalue())
                        tmp_path = tmp.name
                    try:
                        rag.ingest_document(tmp_path)
                        st.success("PDF ingested successfully.")
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    with tab_ask:
        st.header("Ask Question")
        question = st.text_input("Question", key="ask_question_input", placeholder="Enter your question...")
        if st.button("Ask", key="ask_btn") and question.strip():
            rag = _get_rag()
            if rag is None:
                st.error("RAG not available. Check Settings.")
            else:
                with st.spinner("Generating answer..."):
                    result = rag.ask_question(question.strip())
                if result is not None:
                    st.subheader("Answer")
                    st.markdown(result.answer)
                    if result.citations:
                        with st.expander("Citations"):
                            for i, c in enumerate(result.citations, 1):
                                st.markdown(f"**[{i}] {c.source}**")
                                st.caption(c.quote[:500] + ("..." if len(c.quote) > 500 else ""))
                    else:
                        st.caption("(No citations returned.)")
                else:
                    st.error("Failed to generate answer.")

    with tab_kb:
        st.header("Knowledge Base")
        rag = _get_rag()
        if rag is None:
            st.warning("RAG not available. Check Settings.")
        else:
            try:
                content = rag.get_knowledge_content()
                st.markdown(content)
            except Exception as e:
                st.error(f"Could not load knowledge base: {e}")

    with tab_search:
        st.header("Manual Search Vector DB")
        search_query = st.text_input("Search query", key="search_query", placeholder="Enter a search phrase...")
        n_results = st.number_input("Number of results", min_value=1, max_value=50, value=10, key="search_n")
        if st.button("Search", key="search_btn") and search_query.strip():
            rag = _get_rag()
            if rag is None:
                st.error("RAG not available. Check Settings.")
            else:
                results = rag.search_chunks(search_query.strip(), n_results=int(n_results))
                if not results:
                    st.info("No results found.")
                else:
                    for i, r in enumerate(results, 1):
                        with st.expander(f"[{i}] Source: {r['source']}"):
                            st.write(r["text"])
                            if r.get("metadata"):
                                st.caption(f"Metadata: {r['metadata']}")


if __name__ == "__main__":
    main()
