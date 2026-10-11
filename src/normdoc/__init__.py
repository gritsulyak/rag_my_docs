from normdoc.assistant import NormDocAssistant, RAGAnswer, build_assistant
from normdoc.config import COLLECTION_NAME
from normdoc.indexing import build_index, load_index, retrieve
from normdoc.llm import build_llm
from normdoc.parsing import (
    Chunk,
    PageContent,
    load_all_chunks,
    load_manifest,
    parse_pdf,
    parse_pdf_to_chunks,
    sanitize_links,
    save_chunks,
    split_text,
)
from normdoc.prompts import (
    SYSTEM_PROMPT,
    format_context,
    ground_citations,
    source_label,
    user_message,
)

__all__ = [
    "Chunk",
    "PageContent",
    "RAGAnswer",
    "SYSTEM_PROMPT",
    "NormDocAssistant",
    "build_assistant",
    "build_index",
    "build_llm",
    "format_context",
    "ground_citations",
    "load_all_chunks",
    "load_index",
    "load_manifest",
    "parse_pdf",
    "parse_pdf_to_chunks",
    "retrieve",
    "sanitize_links",
    "save_chunks",
    "source_label",
    "split_text",
    "user_message",
    "COLLECTION_NAME",
]

__version__ = "0.2.0"