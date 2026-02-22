"""
Global-Context-Aware RAG: local RAG over PDFs with OCR, ChromaDB, and Ollama.

Ingests scanned or text PDFs (pymupdf4llm + layout/OCR), chunks by headers,
stores in ChromaDB, and maintains a knowledge.md via Ollama. Queries expand
to 3 vector searches, then stream a synthesized answer from Ollama.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

# Critical: import pymupdf.layout BEFORE pymupdf4llm to activate OCR for scanned PDFs.
import pymupdf.layout  # noqa: F401
import pymupdf4llm
import pymupdf
import ollama
import chromadb
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

try:
    from sentence_transformers import CrossEncoder
    _CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CrossEncoder = None  # type: ignore[misc, assignment]
    _CROSS_ENCODER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Prompts (tune here without touching logic)
# -----------------------------------------------------------------------------

PROMPTS = {
    "knowledge_extract_system": """You are a precise knowledge extractor. Given document content in markdown, you must output a JSON object with exactly two keys: "glossary" and "topic_index".

- "glossary": list of objects, each with "term" (string) and "definition" (string). Extract key terms and their definitions from the document.
- "topic_index": list of strings. List the main topics or section themes in order of appearance.

Output only valid JSON matching the schema. No markdown, no explanation.""",
    "knowledge_extract_user": """Extract a structured glossary and topic index from the following document content. Output valid JSON only.

Document content (markdown):
---
{markdown_preview}
---

Use the exact JSON schema: glossary (list of {{"term": "...", "definition": "..."}}), topic_index (list of topic strings).""",
    "query_expand_system": """You are a search query expander. Given a knowledge base summary (glossary and topic index) and a user question, you must output exactly 3 different search queries that would find relevant passages in a vector database.

Each query should be a short, specific phrase (3-10 words) targeting different aspects: one conceptual, one keyword-focused, one rephrased. Output only valid JSON with a single key "queries" whose value is a list of exactly 3 strings.""",
    "query_expand_user": """Knowledge base summary:
---
{knowledge_content}
---

User question: {question}

Generate exactly 3 vector search queries (as a JSON object with key "queries", list of 3 strings). No other text.""",
    "answer_synthesize_system": """You are a precise assistant. Answer the user's question using ONLY the provided context. Each context block is labeled with [Source: filename]. You must cite sources using ONLY the exact source filenames that are listed as "Valid source filenames" in the user message—do not invent or use any other filenames. In the "citations" array, use only those exact strings for "source". Do not invent facts. Keep the answer concise and grounded in the context. Output only valid JSON matching the schema: "answer" (string) and "citations" (list of {"source": "filename", "quote": "short excerpt"}).""",
    "answer_synthesize_user": """Valid source filenames (use ONLY these exact strings in citations, no others): {valid_sources}

Context from the knowledge base (each block is labeled with its source file):
---
{context}
---

User question: {question}

Provide your response as JSON only: "answer" (full answer text, citing sources by filename where relevant), "citations" (list of objects with "source" = one of the valid filenames above and "quote" = a short excerpt from that source that supports the answer). Do not use any source name that is not in the valid list.""",
    "knowledge_compact_system": """You are a knowledge compactor. Given a knowledge base (glossary and topic index), you must output a JSON object with exactly two keys: "glossary" and "topic_index".

Your goal is to keep only what is essential for understanding the corpus and for rewriting user questions into good vector-search queries. You must:
- Deduplicate: merge terms/topics that mean the same thing; keep one canonical form.
- Prioritize: keep domain-specific terms and central concepts; drop generic or redundant entries.
- Compress: shorten every definition to at most one clear sentence.
- Keep the topic_index as a concise list of high-level topics (merge overlapping topics).

Output only valid JSON matching the schema. The total output must stay well under the requested character limit. No markdown, no explanation.""",
    "knowledge_compact_user": """Current knowledge base (may be truncated):
---
{knowledge_preview}
---

Produce a condensed glossary and topic index as JSON. Keep only essential terms and topics. Each definition must be one sentence. Output valid JSON only: glossary (list of {{"term": "...", "definition": "..."}}), topic_index (list of topic strings).""",
}

# -----------------------------------------------------------------------------
# Pydantic schemas for Ollama structured output
# -----------------------------------------------------------------------------


class GlossaryEntry(BaseModel):
    """Single glossary entry: term and definition."""

    term: str
    definition: str


class KnowledgeSchema(BaseModel):
    """Structured output for knowledge extraction: glossary and topic index."""

    glossary: list[GlossaryEntry] = Field(default_factory=list)
    topic_index: list[str] = Field(default_factory=list)


class SearchQueries(BaseModel):
    """Exactly 3 search queries for vector retrieval."""

    queries: list[str] = Field(..., min_length=3, max_length=3)


class CitationEntry(BaseModel):
    """A single citation: source PDF filename and supporting quote."""

    source: str
    quote: str


class AnswerWithCitations(BaseModel):
    """Structured answer with citations for RAG response."""

    answer: str
    citations: list[CitationEntry] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Default paths and config
# -----------------------------------------------------------------------------

DEFAULT_CHROMA_PATH = "./chroma_db"
DEFAULT_KNOWLEDGE_PATH = "./knowledge.md"
DEFAULT_COLLECTION_NAME = "rag_chunks"
DEFAULT_MODEL = "llama3.2"
KNOWLEDGE_MARKDOWN_HEADER = "# Knowledge\n\n## Glossary\n\n"
KNOWLEDGE_TOPIC_HEADER = "\n## Topic Index\n\n"
DEFAULT_KNOWLEDGE_CONTENT = KNOWLEDGE_MARKDOWN_HEADER + "(empty)\n" + KNOWLEDGE_TOPIC_HEADER + "(empty)\n"
MAX_MARKDOWN_FOR_KNOWLEDGE = 12000  # chars per window sent to LLM for knowledge extraction
KNOWLEDGE_WINDOW_OVERLAP = 500      # overlap between consecutive extraction windows
# 50% of 128K context: ~64K tokens * 4 chars/token ≈ 256K chars
THRESHOLD_KNOWLEDGE_CHARS = 256000
MAX_KNOWLEDGE_FOR_COMPACT = 80000  # max chars per compaction window sent to LLM
N_QUERY_RESULTS = 8
# Reranking: retrieve more, then cross-encoder rerank and take top K
RETRIEVE_K = 50          # total chunks to retrieve (before rerank); n_results per query = ceil(RETRIEVE_K/3)
RERANK_TOP_K = 15        # chunks to keep after reranking
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
MAX_CHUNK_SIZE = 1500   # max chars per vector-store chunk (secondary splitter)
CHUNK_OVERLAP = 150     # overlap for RecursiveCharacterTextSplitter
PDF_PAGE_BATCH_SIZE = 50  # pages per batch for converting large PDFs


def _chroma_safe_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Convert LangChain document metadata to Chroma-safe types (str, int, float, bool)."""
    out: dict[str, str | int | float | bool] = {}
    for k, v in metadata.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


class LocalContextRAG:
    """
    Local, global-context-aware RAG: ingest PDFs (with OCR) and answer questions
    using ChromaDB and Ollama with a maintained knowledge.md.
    """

    def __init__(
        self,
        chroma_path: str = DEFAULT_CHROMA_PATH,
        knowledge_path: str = DEFAULT_KNOWLEDGE_PATH,
        model: str = DEFAULT_MODEL,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        """
        Initialize the RAG: persistent Chroma client and collection, paths, and model name.

        :param chroma_path: Directory for ChromaDB persistence.
        :param knowledge_path: Path to knowledge.md file.
        :param model: Ollama model name (e.g. llama3.2).
        :param collection_name: ChromaDB collection name for chunks.
        """
        self.chroma_path = Path(chroma_path)
        self.knowledge_path = Path(knowledge_path)
        self.model = model
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=str(self.chroma_path))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG chunks from ingested PDFs"},
        )
        self._reranker: Any = None

    def _get_reranker(self) -> Any:
        """Lazy-load the cross-encoder model for reranking (if sentence_transformers is installed)."""
        if not _CROSS_ENCODER_AVAILABLE or CrossEncoder is None:
            return None
        if self._reranker is None:
            logger.info("Loading cross-encoder for reranking: %s", CROSS_ENCODER_MODEL)
            self._reranker = CrossEncoder(CROSS_ENCODER_MODEL)
        return self._reranker

    def _rerank_chunks(
        self,
        question: str,
        chunks_with_sources: list[tuple[str, str]],
        top_k: int = RERANK_TOP_K,
    ) -> list[tuple[str, str]]:
        """
        Rerank (text, source) chunks by relevance to the question using a cross-encoder.
        Returns the top_k chunks in descending score order. If cross-encoder is not
        available, returns the first top_k chunks unchanged.
        """
        if not chunks_with_sources or len(chunks_with_sources) <= top_k:
            return chunks_with_sources
        model = self._get_reranker()
        if model is None:
            logger.warning("Reranking skipped (sentence_transformers not installed); using first %d chunks.", top_k)
            return chunks_with_sources[:top_k]
        pairs = [(question, text) for text, _ in chunks_with_sources]
        scores = model.predict(pairs)
        indexed = list(zip(scores, chunks_with_sources))
        indexed.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in indexed[:top_k]]

    def _ensure_knowledge_file(self) -> str:
        """
        Ensure knowledge.md exists; create with default content if missing.
        :return: Current content of knowledge.md.
        """
        if not self.knowledge_path.exists():
            self.knowledge_path.parent.mkdir(parents=True, exist_ok=True)
            self.knowledge_path.write_text(DEFAULT_KNOWLEDGE_CONTENT, encoding="utf-8")
            return DEFAULT_KNOWLEDGE_CONTENT
        return self.knowledge_path.read_text(encoding="utf-8")

    def _merge_knowledge(self, parsed: KnowledgeSchema) -> None:
        """
        Merge parsed glossary and topic_index into existing knowledge.md.
        Appends new entries to the Glossary and Topic Index sections.
        """
        content = self._ensure_knowledge_file()

        # Parse existing terms/topics to avoid duplicates
        existing_terms: set[str] = set()
        existing_topics: set[str] = set()
        for line in content.split("\n"):
            m = re.match(r"^- \*\*(.+?)\*\*:", line)
            if m:
                existing_terms.add(m.group(1).lower().strip())
            elif line.startswith("- ") and "**" not in line:
                existing_topics.add(line[2:].lower().strip())

        # Build new glossary lines and topic lines (skip duplicates)
        new_glossary_lines = []
        for e in parsed.glossary:
            if e.term.lower().strip() not in existing_terms:
                new_glossary_lines.append(f"- **{e.term}**: {e.definition}")
        new_topic_lines = []
        for t in parsed.topic_index:
            if t.lower().strip() not in existing_topics:
                new_topic_lines.append(f"- {t}")

        # Parse existing sections (simple: find ## Glossary and ## Topic Index)
        glossary_marker = "## Glossary"
        topic_marker = "## Topic Index"
        idx_glossary = content.find(glossary_marker)
        idx_topic = content.find(topic_marker)

        if idx_glossary == -1:
            glossary_section = glossary_marker + "\n\n" + "\n".join(new_glossary_lines) if new_glossary_lines else glossary_marker + "\n\n(empty)\n"
        else:
            # Take content from Glossary to Topic Index (or end)
            end_glossary = idx_topic if idx_topic != -1 else len(content)
            existing_glossary = content[idx_glossary:end_glossary].strip()
            if new_glossary_lines:
                existing_glossary += "\n\n" + "\n".join(new_glossary_lines)
            glossary_section = existing_glossary

        if idx_topic == -1:
            topic_section = topic_marker + "\n\n" + "\n".join(new_topic_lines) if new_topic_lines else topic_marker + "\n\n(empty)\n"
        else:
            existing_topic = content[idx_topic:].strip()
            if new_topic_lines:
                existing_topic += "\n\n" + "\n".join(new_topic_lines)
            topic_section = existing_topic

        new_content = "# Knowledge\n\n" + glossary_section + "\n\n" + topic_section + "\n"
        self.knowledge_path.write_text(new_content, encoding="utf-8")
        if len(new_content) >= THRESHOLD_KNOWLEDGE_CHARS:
            self._compact_knowledge()

    def _compact_window(self, window: str) -> KnowledgeSchema | None:
        """Call Ollama to compact a single window of knowledge content."""
        user_msg = PROMPTS["knowledge_compact_user"].format(knowledge_preview=window)
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": PROMPTS["knowledge_compact_system"]},
                    {"role": "user", "content": user_msg},
                ],
                format=KnowledgeSchema.model_json_schema(),
            )
            raw = response.message.content or ""
        except Exception as e:
            logger.warning("Ollama compaction window failed: %s", e)
            return None
        return self._safe_parse_knowledge(raw)

    def _compact_knowledge(self) -> None:
        """
        If knowledge.md exceeds THRESHOLD_KNOWLEDGE_CHARS (50% of ~128K context),
        call Ollama to produce a condensed glossary and topic index.

        For knowledge.md larger than MAX_KNOWLEDGE_FOR_COMPACT, uses a multi-pass
        sliding window so no content is silently truncated: each window is compacted
        independently, results are deduplicated, and a final pass merges the whole.
        """
        content = self._ensure_knowledge_file()
        if len(content) < THRESHOLD_KNOWLEDGE_CHARS:
            return
        logger.info(
            "Compacting knowledge.md (current size %d chars, threshold %d).",
            len(content),
            THRESHOLD_KNOWLEDGE_CHARS,
        )

        if len(content) <= MAX_KNOWLEDGE_FOR_COMPACT:
            # Single-pass (original behaviour for smaller files)
            parsed = self._compact_window(content)
        else:
            # Multi-pass: compact each window, deduplicate across windows
            logger.info(
                "knowledge.md exceeds single-window limit (%d chars); using multi-pass compaction.",
                MAX_KNOWLEDGE_FOR_COMPACT,
            )
            all_glossary: list[GlossaryEntry] = []
            all_topics: list[str] = []
            seen_terms: set[str] = set()
            seen_topics: set[str] = set()
            step = MAX_KNOWLEDGE_FOR_COMPACT - KNOWLEDGE_WINDOW_OVERLAP
            for start in range(0, len(content), step):
                window = content[start : start + MAX_KNOWLEDGE_FOR_COMPACT]
                result = self._compact_window(window)
                if result is None:
                    continue
                for e in result.glossary:
                    key = e.term.lower().strip()
                    if key and key not in seen_terms:
                        seen_terms.add(key)
                        all_glossary.append(e)
                for t in result.topic_index:
                    key = t.lower().strip()
                    if key and key not in seen_topics:
                        seen_topics.add(key)
                        all_topics.append(t)
            parsed = KnowledgeSchema(glossary=all_glossary, topic_index=all_topics) if (all_glossary or all_topics) else None

        if parsed is None:
            logger.warning("Compaction parse failed; knowledge.md unchanged.")
            return
        glossary_lines = [f"- **{e.term}**: {e.definition}" for e in parsed.glossary]
        topic_lines = [f"- {t}" for t in parsed.topic_index]
        glossary_section = "## Glossary\n\n" + ("\n".join(glossary_lines) if glossary_lines else "(empty)")
        topic_section = "## Topic Index\n\n" + ("\n".join(topic_lines) if topic_lines else "(empty)")
        new_content = "# Knowledge\n\n" + glossary_section + "\n\n" + topic_section + "\n"
        self.knowledge_path.write_text(new_content, encoding="utf-8")
        logger.info(
            "Compacted knowledge.md: %d glossary entries, %d topics (%d chars).",
            len(parsed.glossary),
            len(parsed.topic_index),
            len(new_content),
        )

    def _safe_parse_knowledge(self, raw: str) -> KnowledgeSchema | None:
        """Parse LLM response into KnowledgeSchema; return None on failure."""
        try:
            # Strip markdown code blocks if present
            text = raw.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
            return KnowledgeSchema.model_validate_json(text)
        except Exception as e:
            logger.warning("Failed to parse knowledge JSON: %s", e)
            return None

    def _extract_knowledge_window(self, window: str) -> KnowledgeSchema | None:
        """Call Ollama to extract glossary and topic index from a single markdown window."""
        user_msg = PROMPTS["knowledge_extract_user"].format(markdown_preview=window)
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": PROMPTS["knowledge_extract_system"]},
                    {"role": "user", "content": user_msg},
                ],
                format=KnowledgeSchema.model_json_schema(),
            )
            raw = response.message.content or ""
        except Exception as e:
            logger.warning("Ollama knowledge extraction window failed: %s", e)
            return None
        return self._safe_parse_knowledge(raw)

    def _extract_all_knowledge(self, md: str) -> KnowledgeSchema:
        """
        Extract a unified KnowledgeSchema from the full markdown by sliding a window
        of MAX_MARKDOWN_FOR_KNOWLEDGE chars with KNOWLEDGE_WINDOW_OVERLAP overlap.
        Deduplicates glossary terms and topics across all windows.
        """
        all_glossary: list[GlossaryEntry] = []
        all_topics: list[str] = []
        seen_terms: set[str] = set()
        seen_topics: set[str] = set()

        step = MAX_MARKDOWN_FOR_KNOWLEDGE - KNOWLEDGE_WINDOW_OVERLAP
        total = len(md)
        window_count = 0
        for start in range(0, total, step):
            window = md[start : start + MAX_MARKDOWN_FOR_KNOWLEDGE]
            if not window.strip():
                continue
            window_count += 1
            parsed = self._extract_knowledge_window(window)
            if parsed is None:
                continue
            for entry in parsed.glossary:
                key = entry.term.lower().strip()
                if key and key not in seen_terms:
                    seen_terms.add(key)
                    all_glossary.append(entry)
            for topic in parsed.topic_index:
                key = topic.lower().strip()
                if key and key not in seen_topics:
                    seen_topics.add(key)
                    all_topics.append(topic)

        logger.info(
            "Knowledge extraction complete: %d windows, %d terms, %d topics.",
            window_count,
            len(all_glossary),
            len(all_topics),
        )
        return KnowledgeSchema(glossary=all_glossary, topic_index=all_topics)

    def _safe_parse_search_queries(self, raw: str, fallback_query: str) -> SearchQueries:
        """Parse LLM response into SearchQueries; on failure return fallback (same query 3 times)."""
        try:
            text = raw.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
            parsed = SearchQueries.model_validate_json(text)
            if len(parsed.queries) != 3:
                return SearchQueries(queries=[fallback_query, fallback_query, fallback_query])
            return parsed
        except Exception as e:
            logger.warning("Failed to parse search queries JSON: %s", e)
            return SearchQueries(queries=[fallback_query, fallback_query, fallback_query])

    def _safe_parse_answer(self, raw: str) -> AnswerWithCitations | None:
        """Parse LLM response into AnswerWithCitations; return None on failure."""
        try:
            text = raw.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
            return AnswerWithCitations.model_validate_json(text)
        except Exception as e:
            logger.warning("Failed to parse answer JSON: %s", e)
            return None

    def ingest_document(self, pdf_path: str | Path) -> None:
        """
        Ingest a PDF: parse to markdown (with OCR if needed), chunk by headers,
        upsert into ChromaDB, and update knowledge.md via Ollama.

        :param pdf_path: Path to the PDF file.
        :raises FileNotFoundError: If the PDF file does not exist.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # 1) Parse PDF to Markdown in page batches (handles large files without full-memory load)
        logger.info("Parsing PDF to markdown: %s", pdf_path)
        try:
            doc = pymupdf.open(pdf_path)
            try:
                total_pages = len(doc)
                if total_pages <= PDF_PAGE_BATCH_SIZE:
                    md = pymupdf4llm.to_markdown(doc)
                else:
                    logger.info(
                        "Large PDF (%d pages); converting in batches of %d.",
                        total_pages,
                        PDF_PAGE_BATCH_SIZE,
                    )
                    parts: list[str] = []
                    for start in range(0, total_pages, PDF_PAGE_BATCH_SIZE):
                        end = min(start + PDF_PAGE_BATCH_SIZE, total_pages)
                        batch_pages = list(range(start, end))
                        parts.append(pymupdf4llm.to_markdown(doc, pages=batch_pages))
                    md = "\n\n".join(parts)
            finally:
                doc.close()
        except Exception as e:
            logger.error("PDF parse error: %s", e)
            raise

        if not md or not md.strip():
            logger.warning("PDF produced empty markdown; skipping chunk upsert, still attempting knowledge update.")
            md = "(Document produced no text content.)"

        # 2) Chunk by headers, then sub-chunk any oversized sections with a character splitter
        headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]
        header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        raw_chunks = header_splitter.split_text(md)

        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = []
        for chunk in raw_chunks:
            if len(chunk.page_content) > MAX_CHUNK_SIZE:
                sub_docs = char_splitter.create_documents(
                    [chunk.page_content],
                    metadatas=[dict(chunk.metadata)],
                )
                chunks.extend(sub_docs)
            else:
                chunks.append(chunk)
        logger.info("Split into %d chunks (%d raw header chunks).", len(chunks), len(raw_chunks))

        # 3) Upsert into ChromaDB
        if chunks:
            path_stem = pdf_path.stem
            ids_list: list[str] = []
            documents_list: list[str] = []
            metadatas_list: list[dict[str, str | int | float | bool]] = []
            for i, chunk in enumerate(chunks):
                content = chunk.page_content
                if not content.strip():
                    continue
                chunk_id = hashlib.sha256(
                    f"{pdf_path.resolve()!s}|{i}|{content[:80]}".encode()
                ).hexdigest()
                meta = _chroma_safe_metadata(dict(chunk.metadata))
                meta["source"] = str(pdf_path.name)
                meta["chunk_index"] = i
                ids_list.append(chunk_id)
                documents_list.append(content)
                metadatas_list.append(meta)
            if ids_list:
                self._collection.upsert(
                    ids=ids_list,
                    documents=documents_list,
                    metadatas=metadatas_list,
                )
                logger.info("Upserted %d chunks to ChromaDB.", len(ids_list))

        # 4) Update knowledge.md via Ollama (full-document multi-window extraction)
        logger.info("Extracting knowledge from full document (%d chars) via sliding window.", len(md))
        parsed = self._extract_all_knowledge(md)
        if parsed.glossary or parsed.topic_index:
            self._merge_knowledge(parsed)
            logger.info("Merged knowledge (glossary=%d, topics=%d).", len(parsed.glossary), len(parsed.topic_index))
        else:
            logger.warning("Skipped knowledge merge: no entries extracted from document.")

    def get_knowledge_content(self) -> str:
        """
        Return the current content of knowledge.md (create default if missing).

        :return: Full text of knowledge.md.
        """
        return self._ensure_knowledge_file()

    def search_chunks(self, query: str, n_results: int = 10) -> list[dict[str, Any]]:
        """
        Run a single vector search over the chunk collection (manual search, no LLM).

        :param query: Search query string.
        :param n_results: Max number of results to return.
        :return: List of dicts with keys "text", "source", "metadata" per hit.
        """
        try:
            result = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas"],
            )
        except Exception:
            return []
        docs = result.get("documents")
        metadatas = result.get("metadatas")
        if not docs or not docs[0]:
            return []
        ids_list = result.get("ids", [[]])[0]
        meta_list = (metadatas[0] if metadatas and metadatas[0] else []) or []
        out: list[dict[str, Any]] = []
        for idx, doc_text in enumerate(docs[0]):
            meta = meta_list[idx] if idx < len(meta_list) else {}
            raw_source = meta.get("source")
            source = str(raw_source).strip() if raw_source is not None and str(raw_source).strip() else "unknown"
            out.append({"text": doc_text, "source": source, "metadata": meta})
        return out

    def ask_question(self, question: str) -> tuple[AnswerWithCitations | None, str]:
        """
        Answer a question: read knowledge.md, expand to 3 queries, retrieve from ChromaDB
        (with source filename per chunk), dedupe, then get a structured answer with
        citations from Ollama.

        :param question: User question string.
        :return: Tuple of (parsed answer with citations, or None on failure; raw JSON string from Ollama).
        """
        # 1) Read knowledge.md (compact if over threshold)
        knowledge_content = self._ensure_knowledge_file()
        if len(knowledge_content) >= THRESHOLD_KNOWLEDGE_CHARS:
            self._compact_knowledge()
            knowledge_content = self._ensure_knowledge_file()

        # 2) Generate 3 search queries via Ollama
        user_msg = PROMPTS["query_expand_user"].format(
            knowledge_content=knowledge_content,
            question=question,
        )
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": PROMPTS["query_expand_system"]},
                    {"role": "user", "content": user_msg},
                ],
                format=SearchQueries.model_json_schema(),
            )
            raw = response.message.content or ""
        except Exception as e:
            logger.warning("Ollama query expansion failed: %s", e)
            raw = ""
        search_queries = self._safe_parse_search_queries(raw, question)

        # 3) Retrieve more chunks (then rerank to top K)
        n_results_per_query = max(N_QUERY_RESULTS, (RETRIEVE_K + 2) // 3)
        seen_ids: set[str] = set()
        chunks_with_sources: list[tuple[str, str]] = []  # (doc_text, source_filename)
        for q in search_queries.queries:
            try:
                result = self._collection.query(
                    query_texts=[q],
                    n_results=n_results_per_query,
                    include=["documents", "metadatas"],
                )
            except Exception as e:
                logger.warning("ChromaDB query failed for %r: %s", q[:50], e)
                continue
            docs = result.get("documents")
            metadatas = result.get("metadatas")
            if not docs or not docs[0]:
                continue
            ids_list = result.get("ids", [[]])[0]
            meta_list = (metadatas[0] if metadatas and metadatas[0] else []) or []
            for idx, doc_id in enumerate(ids_list):
                if doc_id in seen_ids:
                    continue
                doc_text = docs[0][idx] if idx < len(docs[0]) else ""
                if not doc_text:
                    continue
                meta = meta_list[idx] if idx < len(meta_list) else {}
                raw_source = meta.get("source")
                source = str(raw_source).strip() if raw_source is not None and str(raw_source).strip() else "unknown"
                seen_ids.add(doc_id)
                chunks_with_sources.append((doc_text, source))
        # Rerank with cross-encoder and take top K
        chunks_with_sources = self._rerank_chunks(question, chunks_with_sources, top_k=RERANK_TOP_K)
        # Build context with source labels so the model can cite
        context_parts = [
            f"[Source: {source}]\n{text}" for text, source in chunks_with_sources
        ]
        context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant context found."
        valid_sources = ", ".join(sorted({source for _, source in chunks_with_sources}))

        # 4) Structured answer with citations (non-streaming for JSON)
        user_msg = PROMPTS["answer_synthesize_user"].format(
            context=context, question=question, valid_sources=valid_sources
        )
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": PROMPTS["answer_synthesize_system"]},
                    {"role": "user", "content": user_msg},
                ],
                format=AnswerWithCitations.model_json_schema(),
            )
            raw_content = response.message.content or ""
        except Exception as e:
            logger.error("Ollama answer call failed: %s", e)
            return (None, "")
        parsed = self._safe_parse_answer(raw_content)
        return (parsed, raw_content)


def main() -> None:
    """CLI: ingest <pdf_path> | ask <question>."""
    import sys

    rag = LocalContextRAG()
    if len(sys.argv) < 2:
        print("Usage: python local_context_rag.py ingest <path/to/file.pdf>")
        print("       python local_context_rag.py ask \"Your question?\"")
        sys.exit(1)
    cmd = sys.argv[1].lower()
    if cmd == "ingest":
        if len(sys.argv) < 3:
            print("Usage: python local_context_rag.py ingest <path/to/file.pdf>")
            sys.exit(1)
        rag.ingest_document(sys.argv[2])
    elif cmd == "ask":
        if len(sys.argv) < 3:
            print("Usage: python local_context_rag.py ask \"Your question?\"")
            sys.exit(1)
        result, _raw = rag.ask_question(sys.argv[2])
        if result is not None:
            print("Answer:\n")
            print(result.answer)
            if result.citations:
                print("\nCitations:")
                for i, c in enumerate(result.citations, 1):
                    print(f"  [{i}] {c.source}: {c.quote[:200]}{'...' if len(c.quote) > 200 else ''}")
            else:
                print("\n(No citations returned.)")
        else:
            print("(Failed to generate answer.)")
    else:
        print("Unknown command. Use 'ingest' or 'ask'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
