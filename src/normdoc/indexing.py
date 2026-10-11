"""Векторизация чанков и сборка векторной базы (Qdrant) через LlamaIndex."""

from __future__ import annotations

import time

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from normdoc.config import (
    COLLECTION_NAME,
    EMBEDDING_DEVICE,
    EMBEDDING_MAX_LENGTH,
    EMBEDDING_MODEL,
    EMBEDDING_PASSAGE_PREFIX,
    EMBEDDING_QUERY_PREFIX,
    EMBEDDING_TRUST_REMOTE_CODE,
    MIN_SCORE,
    QDRANT_URL,
    RERANK_ENABLED,
    RETRIEVE_CANDIDATES,
    TOP_K,
)
from normdoc.llm import build_llm
from normdoc.parsing import Chunk
from normdoc.reranking import get_reranker


def select_device() -> str:
    """Выбор устройства для эмбеддингов: auto -> cuda/mps/cpu."""
    if EMBEDDING_DEVICE != "auto":
        return EMBEDDING_DEVICE
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    try:
        import torch  # noqa: F811

        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def make_embed_model():
    device = select_device()
    print(f"[indexing] эмбеддинги: {EMBEDDING_MODEL} на устройстве {device}")
    kwargs = {
        "model_name": EMBEDDING_MODEL,
        "device": device,
        "normalize": True,
        "query_instruction": EMBEDDING_QUERY_PREFIX,
        "text_instruction": EMBEDDING_PASSAGE_PREFIX,
    }
    if EMBEDDING_TRUST_REMOTE_CODE:
        kwargs["trust_remote_code"] = True
    if EMBEDDING_MAX_LENGTH:
        kwargs["max_length"] = EMBEDDING_MAX_LENGTH
    return HuggingFaceEmbedding(**kwargs)


def configure_settings() -> None:
    Settings.embed_model = make_embed_model()
    Settings.llm = build_llm()
    Settings.node_parser = SimpleNodeParser.from_defaults()


def chunks_to_nodes(chunks: list[Chunk]) -> list[TextNode]:
    return [TextNode(text=chunk.text, metadata=dict(chunk.metadata)) for chunk in chunks]


def build_index(
    chunks: list[Chunk],
    *,
    reset: bool = False,
    collection_name: str = COLLECTION_NAME,
) -> VectorStoreIndex:
    """Собирает Qdrant-коллекцию: векторы + payload с метаданными каждого чанка."""
    client = qdrant_client()

    if client.collection_exists(collection_name):
        if not reset:
            print(f"[indexing] коллекция {collection_name!r} уже существует; пересборка пропущена (--reset).")
        else:
            print(f"[indexing] удаляю коллекцию {collection_name!r}...")
            client.delete_collection(collection_name)

    configure_settings()
    nodes = chunks_to_nodes(chunks)
    print(f"[indexing] чанков к индексации: {len(nodes)}")

    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    started = time.monotonic()
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True,
    )
    elapsed = time.monotonic() - started
    print(f"[indexing] индекс собран за {elapsed:.1f} c (коллекция={collection_name!r})")
    return index


_INDEX_CACHE: dict[str, VectorStoreIndex] = {}


def load_index(collection_name: str = COLLECTION_NAME) -> VectorStoreIndex:
    """Подключение к существующей коллекции без повторной индексации.

    Результат кэшируется по имени коллекции: повторные запросы не пересоздают
    модель эмбеддингов (иначе латентность каждого запроса растёт на секунды).
    """
    if collection_name in _INDEX_CACHE:
        return _INDEX_CACHE[collection_name]
    client = qdrant_client()
    if not client.collection_exists(collection_name):
        raise RuntimeError(
            f"Коллекция {collection_name!r} не найдена. Сначала запустите build_index."
        )
    configure_settings()
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    _INDEX_CACHE[collection_name] = index
    return index


def retrieve(
    query: str,
    top_k: int = TOP_K,
    collection_name: str = COLLECTION_NAME,
    min_score: float = MIN_SCORE,
    rerank: bool | None = None,
    candidates: int = RETRIEVE_CANDIDATES,
) -> list:
    """Retriever: топ-K релевантных чанков с метаданными.

    Если включён реранкер, сначала достаём широкий пул `candidates`
    bi-encoder'ом, затем cross-encoder переупорядочивает и оставляет top_k.
    """
    use_rerank = RERANK_ENABLED if rerank is None else rerank
    index = load_index(collection_name=collection_name)
    pool = max(top_k, candidates) if use_rerank else top_k
    nodes = index.as_retriever(similarity_top_k=pool).retrieve(query)

    if not use_rerank:
        if min_score and min_score > 0:
            nodes = [n for n in nodes if getattr(n, "score", None) is not None and n.score >= min_score]
        return nodes

    reranker = get_reranker()
    if reranker is None:
        return nodes[:top_k]
    return reranker.rerank(query, nodes, top_n=top_k)