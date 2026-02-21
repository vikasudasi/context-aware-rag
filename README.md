# Global-Context-Aware RAG

A **local** RAG system that ingests PDFs (including scanned/image-only via OCR), chunks by markdown headers, stores in ChromaDB, and maintains a `knowledge.md` file via Ollama. Queries expand to 3 vector searches, then stream a synthesized answer. Everything runs on your machine; no API keys required.

## Tech stack

- **PDF & OCR**: `pymupdf4llm` + `pymupdf.layout` (Tesseract for scanned pages)
- **LLM**: Ollama (e.g. `llama3.2`)
- **Vector DB**: ChromaDB (persistent, default embedding: all-MiniLM-L6-v2, 384 dimensions)
- **Chunking**: `langchain-text-splitters` (MarkdownHeaderTextSplitter)
- **Structured output**: Pydantic schemas for glossary, topic index, and search queries

## Features

- **Ingestion**: PDF → Markdown (with OCR when needed) → header-based chunks → ChromaDB upsert; Ollama extracts glossary and topic index into `knowledge.md`.
- **Query**: Read `knowledge.md` → Ollama generates 3 vector-search queries → ChromaDB retrieval → dedupe → Ollama streams the final answer.
- **Knowledge compaction**: When `knowledge.md` reaches ~50% of a 128K-token context (≈256K chars), Ollama compacts it: deduplicate terms/topics, shorten definitions, keep only what’s essential for query expansion. Runs after merge and before query so the file stays within context.

## Requirements

- **Python 3.10+**
- **Tesseract** installed on your system (for OCR of scanned PDFs)
- **Ollama** running locally with a model such as `llama3.2` (`ollama pull llama3.2`)

## Install

```bash
pip install -r requirements.txt
```

Optional: `opencv-python` improves OCR heuristics for scanned PDFs.

## Usage

**Ingest a PDF** (text or scanned; OCR is used automatically when needed):

```bash
python local_context_rag.py ingest path/to/document.pdf
```

**Ask a question** (reads `knowledge.md`, expands to 3 queries, retrieves from ChromaDB, streams answer):

```bash
python local_context_rag.py ask "Your question here?"
```

## Streamlit UI

A separate Streamlit app uses the same `LocalContextRAG` script internally. Run:

```bash
streamlit run app.py
```

- **Upload PDF** — Upload a PDF and ingest it (chunks + knowledge.md update).
- **Ask Question** — Enter a question; get a structured answer with citations (source PDF + quote).
- **Knowledge Base** — View the current `knowledge.md` (glossary + topic index).
- **Search Vector DB** — Run a custom vector search over stored chunks (no LLM); see source and text per hit.
- **Settings (sidebar)** — Set `chroma_path`, `knowledge_path`, `model`, and `collection_name`; click Apply to re-create the RAG with new params.

## Programmatic use

```python
from local_context_rag import LocalContextRAG

rag = LocalContextRAG(
    chroma_path="./chroma_db",
    knowledge_path="./knowledge.md",
    model="llama3.2",
)
rag.ingest_document("document.pdf")
rag.ask_question("What is the main topic?")
```

## Files

- `local_context_rag.py` — main script and `LocalContextRAG` class
- `app.py` — Streamlit UI (Upload PDF, Ask Question, Knowledge Base, Search Vector DB, Settings)
- `knowledge.md` — created/updated by ingestion (glossary + topic index); compacted automatically when it grows too large
- `chroma_db/` — ChromaDB persistence (default path)
