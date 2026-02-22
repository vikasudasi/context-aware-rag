# Global-Context-Aware RAG

A **local** RAG system that ingests PDFs (including scanned/image-only via OCR), chunks by markdown headers, stores in ChromaDB, and maintains a `knowledge.md` file via Ollama. Queries expand to 3 vector searches, rerank, then synthesize an answer with citations. **Domain rules** extracted during indexing steer query expansion and answer style. Everything runs on your machine; no API keys required.

## Tech stack

- **PDF & OCR**: `pymupdf4llm` + `pymupdf.layout` (Tesseract for scanned pages)
- **LLM**: Ollama (e.g. `llama3.2`)
- **Vector DB**: ChromaDB (persistent, default embedding: all-MiniLM-L6-v2, 384 dimensions)
- **Chunking**: `langchain-text-splitters` (MarkdownHeaderTextSplitter)
- **Reranking**: `sentence-transformers` CrossEncoder (optional; falls back to top-K by vector score if not installed)
- **Structured output**: Pydantic schemas for glossary, topic index, domain rules, search queries, and answers with citations

## Features

- **Ingestion**: PDF → Markdown (with OCR when needed) → header-based chunks → ChromaDB upsert (with `source`, `h1`/`h2`/`h3` metadata). Ollama extracts **glossary**, **topic index**, and **domain rules** into `knowledge.md` (all three sections are merged and compacted when the file grows).
- **Domain rules**: During indexing, the model infers domain rules (query guidance, reasoning, answer structure, terminology/citations) from the document. These are stored in `knowledge.md`, used when expanding the user question into 3 expert-like search queries, and injected into the answer step so the model reasons and structures answers according to the domain.
- **Query**: Read `knowledge.md` (glossary + topic index + domain rules) → Ollama generates 3 vector-search queries → ChromaDB retrieval (top ~50) → **cross-encoder rerank** (top 15) → dedupe → Ollama for final answer with citations.
- **Context with source and section**: Each retrieved chunk is labeled as `[Source: filename | Section: h1 > h2 > h3]` in the context so the model knows which document and section the text came from. Citations are constrained to **valid source filenames** (exact names from retrieved chunks) to avoid invented sources.
- **Knowledge compaction**: When `knowledge.md` reaches ~50% of a 128K-token context (≈256K chars), Ollama compacts all three sections: deduplicate and shorten glossary, topic index, and domain rules. Runs after merge and before query so the file stays within context.

## Requirements

- **Python 3.10+**
- **Tesseract** installed on your system (for OCR of scanned PDFs). OCR is set to **English + Hindi** (`eng+hin`); for scanned Hindi PDFs, install Tesseract Hindi data (e.g. `tessdata` for `hin`).
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

**Ask a question** (reads `knowledge.md`, expands to 3 queries, retrieves and reranks from ChromaDB, returns answer with citations):

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
- **Knowledge Base** — View the current `knowledge.md` (glossary, topic index, and domain rules).
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
answer, raw = rag.ask_question("What is the main topic?")  # returns (AnswerWithCitations | None, raw JSON string)
```

## Files

- `local_context_rag.py` — main script and `LocalContextRAG` class
- `app.py` — Streamlit UI (Upload PDF, Ask Question, Knowledge Base, Search Vector DB, Settings)
- `knowledge.md` — created/updated by ingestion (glossary, topic index, and domain rules); compacted automatically when it grows too large
- `chroma_db/` — ChromaDB persistence (default path)

## License

MIT License. See [LICENSE](LICENSE).
