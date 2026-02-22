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

# Optional: OpenCV + pytesseract for enhanced OCR preprocessing pipeline.
# If unavailable the enhanced fallback is silently skipped and pymupdf4llm handles OCR.
try:
    import cv2 as _cv2
    import numpy as _np
    _OPENCV_AVAILABLE = True
except ImportError:
    _cv2 = None  # type: ignore[assignment]
    _np = None   # type: ignore[assignment]
    _OPENCV_AVAILABLE = False

try:
    import pytesseract as _pytesseract
    _PYTESSERACT_AVAILABLE = True
except ImportError:
    _pytesseract = None  # type: ignore[assignment]
    _PYTESSERACT_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Prompts (tune here without touching logic)
# -----------------------------------------------------------------------------

PROMPTS = {
    "knowledge_extract_system": """You are a precise knowledge extractor. Given document content in markdown, you must output a JSON object with exactly three keys: "glossary", "topic_index", and "domain_rules".

- "glossary": list of objects, each with "term" (string) and "definition" (string). Extract key terms and their definitions from the document.
- "topic_index": list of strings. List the main topics or section themes in order of appearance.
- "domain_rules": list of strings. Extract domain rules that should apply when answering questions from this material. Include rules that help an expert in this domain: how to phrase search queries (terminology, key concepts to target), how to reason (what to consider first, what not to assume), how to structure answers (what to include, order, caveats), and terminology/citation constraints (e.g. cite policy numbers, use specific units). Each rule should be one short, actionable sentence.

Output only valid JSON matching the schema. No markdown, no explanation.""",
    "knowledge_extract_user": """Extract a structured glossary, topic index, and domain rules from the following document content. Output valid JSON only.

Document content (markdown):
---
{markdown_preview}
---

Use the exact JSON schema: glossary (list of {{"term": "...", "definition": "..."}}), topic_index (list of topic strings), domain_rules (list of rule strings).""",
    "query_expand_system": """You are a search query expander. Given a knowledge base (glossary, topic index, and domain rules) and a user question, you must output exactly 3 different search queries that would find relevant passages in a vector database.

Use the domain rules and glossary to phrase expert-like queries: use domain terminology, target the right concepts, and follow any query-related guidance. Each query should be a short, specific phrase (3-10 words) targeting different aspects: one conceptual, one keyword-focused, one rephrased. Output only valid JSON with a single key "queries" whose value is a list of exactly 3 strings.""",
    "query_expand_user": """Knowledge base summary:
---
{knowledge_content}
---

User question: {question}

Generate exactly 3 vector search queries (as a JSON object with key "queries", list of 3 strings). No other text.""",
    "answer_synthesize_system": """You are a precise assistant. Answer the user's question using ONLY the provided context. Follow these domain rules when reasoning and structuring your answer:
---
{domain_rules}
---

Each context block is labeled with [Source: filename | Section: h1 > h2 > h3]. You must cite sources using ONLY the exact source filenames listed as "Valid source filenames" in the user message—do not invent or use any other filenames. In the "citations" array, use only those exact strings for "source". Do not invent facts. Keep the answer concise and grounded in the context. Output only valid JSON: "answer" (string) and "citations" (list of {{"source": "filename", "quote": "short excerpt"}}).""",
    "answer_synthesize_user": """Valid source filenames (use ONLY these exact strings in citations, no others): {valid_sources}

Context from the knowledge base (each block is labeled with source file and section path):
---
{context}
---

User question: {question}

Provide your response as JSON only: "answer" (full answer text, citing sources by filename where relevant), "citations" (list of objects with "source" = one of the valid filenames above and "quote" = a short excerpt from that source that supports the answer). Do not use any source name that is not in the valid list.""",
    "knowledge_compact_system": """You are a knowledge compactor. Given a knowledge base (glossary, topic index, and domain rules), you must output a JSON object with exactly three keys: "glossary", "topic_index", and "domain_rules".

Your goal is to keep only what is essential for understanding the corpus, for rewriting user questions into good vector-search queries, and for expert-like answering. You must:
- Deduplicate: merge terms/topics/rules that mean the same thing; keep one canonical form.
- Prioritize: keep domain-specific terms, central concepts, and the most important domain rules (query guidance, reasoning, answer structure, citation/terminology).
- Compress: shorten every definition to at most one clear sentence; keep domain rules as short actionable sentences.
- Keep the topic_index as a concise list of high-level topics and domain_rules as a concise list of rule strings.

Output only valid JSON matching the schema. The total output must stay well under the requested character limit. No markdown, no explanation.""",
    "knowledge_compact_user": """Current knowledge base (may be truncated):
---
{knowledge_preview}
---

Produce a condensed glossary, topic index, and domain rules as JSON. Keep only essential terms, topics, and rules. Each definition must be one sentence; each domain rule one short sentence. Output valid JSON only: glossary (list of {{"term": "...", "definition": "..."}}), topic_index (list of topic strings), domain_rules (list of rule strings).""",
}

# -----------------------------------------------------------------------------
# Pydantic schemas for Ollama structured output
# -----------------------------------------------------------------------------


class GlossaryEntry(BaseModel):
    """Single glossary entry: term and definition."""

    term: str
    definition: str


class KnowledgeSchema(BaseModel):
    """Structured output for knowledge extraction: glossary, topic index, and domain rules."""

    glossary: list[GlossaryEntry] = Field(default_factory=list)
    topic_index: list[str] = Field(default_factory=list)
    domain_rules: list[str] = Field(default_factory=list)


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
KNOWLEDGE_DOMAIN_RULES_HEADER = "\n## Domain rules\n\n"
DEFAULT_KNOWLEDGE_CONTENT = (
    KNOWLEDGE_MARKDOWN_HEADER + "(empty)\n"
    + KNOWLEDGE_TOPIC_HEADER + "(empty)\n"
    + KNOWLEDGE_DOMAIN_RULES_HEADER + "(empty)\n"
)
MAX_MARKDOWN_FOR_KNOWLEDGE = 12000  # chars per window sent to LLM for knowledge extraction
KNOWLEDGE_WINDOW_OVERLAP = 500      # overlap between consecutive extraction windows
# 50% of 128K context: ~64K tokens * 4 chars/token ≈ 256K chars
THRESHOLD_KNOWLEDGE_CHARS = 256000
MAX_KNOWLEDGE_FOR_COMPACT = 80000  # max chars per compaction window sent to LLM
N_QUERY_RESULTS = 8
# Reranking: retrieve more, then cross-encoder rerank and take top K
RETRIEVE_K = 50          # total chunks to retrieve (before rerank); n_results per query = ceil(RETRIEVE_K/3)
RERANK_TOP_K = 15        # chunks to keep after reranking
TOP_N_FOR_PARENTS = 5     # only resolve parent for top N chunks (good parent-to-child ratio)
PARENT_MIN_CHARS = 100    # only add parent to context if longer than this
PARENT_MAX_CHARS = 500    # truncate parent to this length when adding to context
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
MAX_CHUNK_SIZE = 1500   # max chars per vector-store chunk (secondary splitter)
CHUNK_OVERLAP = 150     # overlap for RecursiveCharacterTextSplitter
PDF_PAGE_BATCH_SIZE = 50  # pages per batch for converting large PDFs
# OCR language(s) for scanned PDFs (Tesseract codes; only used when pymupdf.layout is imported)
OCR_LANGUAGE = "eng+hin"  # English + Hindi (install Tesseract hin data for Hindi)
OCR_DPI = 300             # Render resolution for OCR; 300 DPI is the industry standard for good accuracy
# Pages with fewer native chars AND at least one image are treated as scanned and sent through the
# enhanced OpenCV + pytesseract pipeline in addition to pymupdf4llm.
OCR_NATIVE_TEXT_THRESHOLD = 30
# Tesseract engine/page-seg flags: OEM 3 = LSTM (best accuracy), PSM 3 = fully automatic layout.
OCR_TESSERACT_CONFIG = r"--oem 3 --psm 3"


def _preprocess_image_for_ocr(img_bgr: Any) -> Any:
    """
    Preprocess a BGR image to improve OCR accuracy on scanned/messy documents.

    Pipeline:
      1. Grayscale conversion
      2. Non-local means denoising  (removes scanner noise / JPEG artefacts)
      3. Adaptive Gaussian binarization  (handles uneven illumination / shadows)
      4. Deskew  (corrects page rotation up to ~45 °)

    Requires OpenCV (cv2) and NumPy.  Returns the preprocessed single-channel
    image as a NumPy array.  If OpenCV is unavailable the input is returned
    unchanged so callers can still attempt OCR on the raw image.
    """
    if not _OPENCV_AVAILABLE:
        return img_bgr
    cv2 = _cv2
    np = _np
    # 1. Grayscale
    gray: Any = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # 2. Denoise — h=10 is a good balance between noise removal and detail retention
    denoised: Any = cv2.fastNlMeansDenoising(gray, h=10)
    # 3. Adaptive binarization — better than global threshold for uneven lighting
    binary: Any = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )
    # 4. Deskew: estimate rotation from the distribution of dark (text) pixels
    coords: Any = np.column_stack(np.where(binary < 128))
    if len(coords) > 100:
        angle: float = cv2.minAreaRect(coords)[-1]
        # minAreaRect returns angles in [-90, 0); convert to a meaningful rotation
        angle = -(90 + angle) if angle < -45 else -angle
        if abs(angle) > 0.3:  # skip negligible rotation
            h, w = binary.shape
            M: Any = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            binary = cv2.warpAffine(
                binary, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
    return binary


def _ocr_page_enhanced(page: Any, ocr_language: str) -> str:
    """
    High-quality OCR fallback for scanned / image-heavy PDF pages.

    Steps:
      1. Render the page to a pixmap at OCR_DPI (300 DPI by default).
      2. Preprocess with :func:`_preprocess_image_for_ocr`.
      3. Run pytesseract with OEM 3 (LSTM) + PSM 3 (auto layout detection).

    Returns the extracted text, or an empty string if the enhanced pipeline
    is unavailable (missing cv2 / pytesseract) or encounters an error.

    The LSTM engine (OEM 3) is significantly more accurate than the legacy
    engine for non-Latin scripts such as Hindi / Devanagari.
    """
    if not (_OPENCV_AVAILABLE and _PYTESSERACT_AVAILABLE):
        return ""
    try:
        from PIL import Image as _PILImage  # Pillow is required by pytesseract
        np = _np
        cv2 = _cv2
        mat = pymupdf.Matrix(OCR_DPI / 72, OCR_DPI / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=pymupdf.csRGB)
        arr: Any = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        img_bgr: Any = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        preprocessed = _preprocess_image_for_ocr(img_bgr)
        pil_img = _PILImage.fromarray(preprocessed)
        text: str = _pytesseract.image_to_string(
            pil_img,
            lang=ocr_language,
            config=OCR_TESSERACT_CONFIG,
        )
        return text.strip()
    except Exception as exc:
        logger.warning("Enhanced OCR failed on page %d: %s", getattr(page, "number", "?") + 1, exc)
        return ""


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


def _section_label(meta: dict[str, Any]) -> str:
    """Build a section path from header metadata (h1, h2, h3) for context labels."""
    parts = []
    for key in ("h1", "h2", "h3"):
        v = meta.get(key)
        if v is not None and str(v).strip():
            parts.append(str(v).strip())
    return " > ".join(parts) if parts else "—"


def _chunk_index_from_meta(meta: dict[str, Any]) -> int | None:
    """Return chunk_index from metadata as int, or None if missing/invalid."""
    v = meta.get("chunk_index")
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


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
        chunks_with_sources: list[tuple[str, str, dict[str, Any]]],
        top_k: int = RERANK_TOP_K,
    ) -> list[tuple[str, str, dict[str, Any]]]:
        """
        Rerank (text, source, meta) chunks by relevance to the question using a cross-encoder.
        Returns the top_k chunks in descending score order. If cross-encoder is not
        available, returns the first top_k chunks unchanged.
        """
        if not chunks_with_sources or len(chunks_with_sources) <= top_k:
            return chunks_with_sources
        model = self._get_reranker()
        if model is None:
            logger.warning("Reranking skipped (sentence_transformers not installed); using first %d chunks.", top_k)
            return chunks_with_sources[:top_k]
        pairs = [(question, text) for text, _s, _m in chunks_with_sources]
        scores = model.predict(pairs)
        indexed = list(zip(scores, chunks_with_sources))
        indexed.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in indexed[:top_k]]

    def _get_parent_chunk(
        self,
        source: str,
        meta: dict[str, Any],
    ) -> tuple[str, str, dict[str, Any]] | None:
        """
        Resolve the section-root (parent) chunk at query time.
        For a chunk with (h1, h2, h3), parent is the chunk with same source, h1, h2 and min chunk_index.
        For (h1, h2), parent is the chunk with same source, h1 and min chunk_index.
        Returns (doc_text, source, meta) or None if not found or no parent level.
        """
        h1 = meta.get("h1")
        h2 = meta.get("h2")
        if h1 is None or not str(h1).strip():
            return None
        where: dict[str, Any] = {"source": source, "h1": str(h1).strip()}
        if h2 is not None and str(h2).strip():
            where["h2"] = str(h2).strip()
        try:
            result = self._collection.get(
                where=where,
                include=["documents", "metadatas"],
            )
        except Exception as e:
            logger.debug("Parent chunk lookup failed for %s: %s", where, e)
            return None
        ids_list = result.get("ids") or []
        docs = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        if not ids_list or not docs:
            return None
        # Chroma get returns flat lists; get first doc and meta per id
        best_idx = 0
        best_index = _chunk_index_from_meta(metadatas[0] if metadatas else {})
        for i in range(1, len(ids_list)):
            ci = _chunk_index_from_meta(metadatas[i] if i < len(metadatas) else {})
            if ci is not None and (best_index is None or ci < best_index):
                best_idx = i
                best_index = ci
        doc_text = docs[best_idx] if best_idx < len(docs) else ""
        parent_meta = metadatas[best_idx] if best_idx < len(metadatas) else {}
        if not doc_text:
            return None
        return (doc_text, source, parent_meta)

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

    def _get_domain_rules_text(self, knowledge_content: str) -> str:
        """
        Extract the Domain rules section body from knowledge.md content for use in prompts.
        :return: The rules section text, or "None specified." if missing or empty.
        """
        marker = "## Domain rules"
        idx = knowledge_content.find(marker)
        if idx == -1:
            return "None specified."
        start = idx + len(marker)
        rest = knowledge_content[start:]
        # Stop at next ## header
        next_h2 = rest.find("\n## ")
        body = rest[:next_h2].strip() if next_h2 != -1 else rest.strip()
        if not body or body.lower() == "(empty)":
            return "None specified."
        return body

    def _merge_knowledge(self, parsed: KnowledgeSchema) -> None:
        """
        Merge parsed glossary, topic_index, and domain_rules into existing knowledge.md.
        Appends new entries to each section; deduplicates by normalized text.
        """
        content = self._ensure_knowledge_file()
        glossary_marker = "## Glossary"
        topic_marker = "## Topic Index"
        rules_marker = "## Domain rules"
        idx_glossary = content.find(glossary_marker)
        idx_topic = content.find(topic_marker)
        idx_rules = content.find(rules_marker)

        # Section bounds: from this section's header until the next ## or end
        end_glossary = idx_topic if idx_topic != -1 else (idx_rules if idx_rules != -1 else len(content))
        if idx_topic != -1:
            end_glossary = min(end_glossary, idx_topic)
        end_topic = idx_rules if idx_rules != -1 else len(content)
        if idx_rules != -1:
            end_topic = min(end_topic, idx_rules)
        # Body = content after the header line (so we don't include "## X" in parsed bullets)
        def body_after(section_start: int, section_end: int, header: str) -> str:
            if section_start == -1:
                return ""
            start = section_start + len(header)
            return content[start:section_end].strip()

        glossary_body = body_after(idx_glossary, end_glossary, glossary_marker) if idx_glossary != -1 else ""
        topic_body = body_after(idx_topic, end_topic, topic_marker) if idx_topic != -1 else ""
        rules_body = body_after(idx_rules, len(content), rules_marker) if idx_rules != -1 else ""

        # Parse existing items from each section only (so topic vs rule bullets don't mix)
        existing_terms: set[str] = set()
        for line in glossary_body.split("\n"):
            m = re.match(r"^- \*\*(.+?)\*\*:", line)
            if m:
                existing_terms.add(m.group(1).lower().strip())

        existing_topics: set[str] = set()
        for line in topic_body.split("\n"):
            if line.startswith("- ") and "**" not in line:
                existing_topics.add(line[2:].strip().lower())

        existing_rules: set[str] = set()
        for line in rules_body.split("\n"):
            if line.startswith("- "):
                existing_rules.add(line[2:].strip().lower())

        # Build new lines (skip duplicates)
        new_glossary_lines = []
        for e in parsed.glossary:
            if e.term.lower().strip() not in existing_terms:
                new_glossary_lines.append(f"- **{e.term}**: {e.definition}")
        new_topic_lines = []
        for t in parsed.topic_index:
            if t.lower().strip() not in existing_topics:
                new_topic_lines.append(f"- {t}")
        new_rule_lines = []
        for r in parsed.domain_rules:
            key = r.strip().lower()
            if key and key not in existing_rules:
                new_rule_lines.append(f"- {r.strip()}")

        # Build sections (keep existing body, append new)
        if idx_glossary == -1:
            glossary_section = glossary_marker + "\n\n" + ("\n".join(new_glossary_lines) if new_glossary_lines else "(empty)\n")
        else:
            existing_glossary = glossary_body
            if new_glossary_lines:
                existing_glossary += "\n\n" + "\n".join(new_glossary_lines)
            glossary_section = glossary_marker + "\n\n" + existing_glossary

        if idx_topic == -1:
            topic_section = topic_marker + "\n\n" + ("\n".join(new_topic_lines) if new_topic_lines else "(empty)\n")
        else:
            existing_topic = topic_body
            if new_topic_lines:
                existing_topic += "\n\n" + "\n".join(new_topic_lines)
            topic_section = topic_marker + "\n\n" + existing_topic

        if idx_rules == -1:
            rules_section = rules_marker + "\n\n" + ("\n".join(new_rule_lines) if new_rule_lines else "(empty)\n")
        else:
            existing_rules_content = rules_body
            if new_rule_lines:
                existing_rules_content += "\n\n" + "\n".join(new_rule_lines)
            rules_section = rules_marker + "\n\n" + existing_rules_content

        new_content = "# Knowledge\n\n" + glossary_section + "\n\n" + topic_section + "\n\n" + rules_section + "\n"
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
            all_rules: list[str] = []
            seen_terms: set[str] = set()
            seen_topics: set[str] = set()
            seen_rules: set[str] = set()
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
                for r in (result.domain_rules or []):
                    key = r.strip().lower()
                    if key and key not in seen_rules:
                        seen_rules.add(key)
                        all_rules.append(r.strip())
            parsed = KnowledgeSchema(glossary=all_glossary, topic_index=all_topics, domain_rules=all_rules) if (all_glossary or all_topics or all_rules) else None

        if parsed is None:
            logger.warning("Compaction parse failed; knowledge.md unchanged.")
            return
        glossary_lines = [f"- **{e.term}**: {e.definition}" for e in parsed.glossary]
        topic_lines = [f"- {t}" for t in parsed.topic_index]
        rule_lines = [f"- {r}" for r in (parsed.domain_rules or [])]
        glossary_section = "## Glossary\n\n" + ("\n".join(glossary_lines) if glossary_lines else "(empty)")
        topic_section = "## Topic Index\n\n" + ("\n".join(topic_lines) if topic_lines else "(empty)")
        rules_section = "## Domain rules\n\n" + ("\n".join(rule_lines) if rule_lines else "(empty)")
        new_content = "# Knowledge\n\n" + glossary_section + "\n\n" + topic_section + "\n\n" + rules_section + "\n"
        self.knowledge_path.write_text(new_content, encoding="utf-8")
        logger.info(
            "Compacted knowledge.md: %d glossary entries, %d topics, %d domain rules (%d chars).",
            len(parsed.glossary),
            len(parsed.topic_index),
            len(parsed.domain_rules or []),
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
        Deduplicates glossary terms, topics, and domain rules across all windows.
        """
        all_glossary: list[GlossaryEntry] = []
        all_topics: list[str] = []
        all_rules: list[str] = []
        seen_terms: set[str] = set()
        seen_topics: set[str] = set()
        seen_rules: set[str] = set()

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
            for r in getattr(parsed, "domain_rules", []) or []:
                key = r.strip().lower()
                if key and key not in seen_rules:
                    seen_rules.add(key)
                    all_rules.append(r.strip())

        logger.info(
            "Knowledge extraction complete: %d windows, %d terms, %d topics, %d domain rules.",
            window_count,
            len(all_glossary),
            len(all_topics),
            len(all_rules),
        )
        return KnowledgeSchema(glossary=all_glossary, topic_index=all_topics, domain_rules=all_rules)

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
        #    Primary path  : pymupdf4llm at OCR_DPI (300 DPI) with Tesseract via pymupdf.layout.
        #    Enhanced path : for pages that contain images but yield fewer than
        #                    OCR_NATIVE_TEXT_THRESHOLD native characters, a second pass using
        #                    OpenCV preprocessing + pytesseract (OEM 3 LSTM) recovers text that
        #                    the primary Tesseract pass missed (skewed scans, low contrast, etc.).
        logger.info("Parsing PDF to markdown: %s", pdf_path)
        try:
            doc = pymupdf.open(pdf_path)
            try:
                total_pages = len(doc)
                if total_pages <= PDF_PAGE_BATCH_SIZE:
                    md = pymupdf4llm.to_markdown(doc, ocr_language=OCR_LANGUAGE, dpi=OCR_DPI)
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
                        parts.append(
                            pymupdf4llm.to_markdown(
                                doc, pages=batch_pages, ocr_language=OCR_LANGUAGE, dpi=OCR_DPI
                            )
                        )
                    md = "\n\n".join(parts)

                # Enhanced OCR pass — identify scanned/image-heavy pages and run the
                # OpenCV-preprocessed pytesseract pipeline on them.  The recovered text
                # is appended only when it is meaningfully longer than the native extraction,
                # so purely textual pages are never duplicated.
                if _OPENCV_AVAILABLE and _PYTESSERACT_AVAILABLE:
                    recovered: list[str] = []
                    for pg_num in range(total_pages):
                        pg = doc[pg_num]
                        native_text = pg.get_text("text").strip()
                        has_images = bool(pg.get_images(full=False))
                        if len(native_text) < OCR_NATIVE_TEXT_THRESHOLD and has_images:
                            enhanced = _ocr_page_enhanced(pg, OCR_LANGUAGE)
                            # Only use the enhanced result if it recovered substantially more text
                            if enhanced and len(enhanced) > len(native_text) + 20:
                                recovered.append(
                                    f"<!-- Enhanced OCR — page {pg_num + 1} -->\n\n{enhanced}"
                                )
                    if recovered:
                        logger.info(
                            "Enhanced OCR pipeline recovered text from %d scanned page(s).",
                            len(recovered),
                        )
                        md = md + "\n\n" + "\n\n".join(recovered)
                else:
                    if not (_OPENCV_AVAILABLE and _PYTESSERACT_AVAILABLE):
                        logger.debug(
                            "Enhanced OCR pipeline unavailable "
                            "(cv2=%s, pytesseract=%s); using pymupdf4llm OCR only.",
                            _OPENCV_AVAILABLE,
                            _PYTESSERACT_AVAILABLE,
                        )
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
        if parsed.glossary or parsed.topic_index or getattr(parsed, "domain_rules", None):
            self._merge_knowledge(parsed)
            logger.info(
                "Merged knowledge (glossary=%d, topics=%d, domain_rules=%d).",
                len(parsed.glossary),
                len(parsed.topic_index),
                len(getattr(parsed, "domain_rules", []) or []),
            )
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
        chunks_with_sources: list[tuple[str, str, dict[str, Any]]] = []  # (doc_text, source_filename, meta)
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
                chunks_with_sources.append((doc_text, source, meta))
        # Rerank with cross-encoder and take top K
        chunks_with_sources = self._rerank_chunks(question, chunks_with_sources, top_k=RERANK_TOP_K)
        # Build context: 15 main chunks, then parent chunks (for reference / better context mapping) for top 5 only
        context_parts = [
            f"[Source: {source} | Section: {_section_label(meta)}]\n{text}"
            for text, source, meta in chunks_with_sources
        ]
        set_of_15 = {
            (src, _chunk_index_from_meta(meta))
            for _t, src, meta in chunks_with_sources
        }
        seen_parent_keys: set[tuple[str, str]] = set()
        parent_blocks: list[tuple[str, str]] = []  # (section_label, truncated_text)
        for text, source, meta in chunks_with_sources[:TOP_N_FOR_PARENTS]:
            parent = self._get_parent_chunk(source, meta)
            if parent is None:
                continue
            p_text, p_source, p_meta = parent
            p_key = (p_source, _section_label(p_meta))
            if p_key in seen_parent_keys:
                continue
            if (p_source, _chunk_index_from_meta(p_meta)) in set_of_15:
                continue
            if len(p_text) <= PARENT_MIN_CHARS:
                continue
            seen_parent_keys.add(p_key)
            truncated = p_text if len(p_text) <= PARENT_MAX_CHARS else p_text[:PARENT_MAX_CHARS].rstrip() + "..."
            parent_blocks.append((_section_label(p_meta), truncated))
        if parent_blocks:
            context_parts.append(
                "Parent chunks (for reference / better context mapping):\n\n"
                + "\n\n".join(
                    f"[Section: {section}]\n{truncated}" for section, truncated in parent_blocks
                )
            )
        context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant context found."
        valid_sources = ", ".join(sorted({source for _t, source, _m in chunks_with_sources}))

        # 4) Structured answer with citations (non-streaming for JSON); inject domain rules into system
        domain_rules_text = self._get_domain_rules_text(knowledge_content)
        answer_system = PROMPTS["answer_synthesize_system"].replace("{domain_rules}", domain_rules_text)
        user_msg = PROMPTS["answer_synthesize_user"].format(
            context=context, question=question, valid_sources=valid_sources
        )
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": answer_system},
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
