"""Конфигурация RAG-ассистента по нормативным документам.

Порядок загрузки:
1) `.env` в корне проекта;
2) профиль железа  `configs/hw/<HW_PROFILE>.env`   (например amd_7840hs, ai395_strix_hala, intel_rtx3060);
3) профиль LLM     `configs/llms/<LLM_PROFILE>.env` (например bonsai, fast_local).
Более поздние переопределяют более ранние, поэтому конкретный запуск можно
тонко настроить, не трогая общие значения.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


_PROFILE_DIRS = {"hw": "hw", "llm": "llms", "embeddings": "embeddings"}
_PROFILE_ENV = {"hw": "HW_PROFILE", "llm": "LLM_PROFILE", "embeddings": "EMBEDDING_PROFILE"}


def _load_profile(kind: str) -> None:
    """Подгружает профиль `configs/{kind}/{name}.env`, если он задан."""
    env_var = _PROFILE_ENV.get(kind, f"{kind.upper()}_PROFILE")
    name = os.getenv(env_var, "").strip()
    if not name:
        return
    dirname = _PROFILE_DIRS.get(kind, kind)
    path = PROJECT_ROOT / "configs" / dirname / f"{name}.env"
    if path.exists():
        load_dotenv(path, override=True)
        print(f"[config] профиль {kind}: {name} -> {path}")
    else:
        print(f"[config] предупреждение: профиль {kind}='{name}' не найден ({path})")


_load_profile("hw")
_load_profile("llm")
_load_profile("embeddings")

# ---------------------------------------------------------------------------
# Пути данных
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
# Оригинальные файлы (pdf/rtf/odt). Можно указать внешний каталог через SOURCES_DIR.
SOURCES_DIR = Path(os.getenv("SOURCES_DIR", str(RAW_DIR / "sources")))
PDF_DIR = RAW_DIR / "pdf"              # подготовленные PDF (rtf/odt конвертированы)
OCR_DIR = RAW_DIR / "ocr"              # sidecar OCR-текст для сканов (<stub>.json)
MANIFEST_PATH = RAW_DIR / "manifest.json"
CHUNKS_JSON = DATA_DIR / "chunks" / "chunks.json"
QA_DIR = DATA_DIR / "qa"
RESULTS_DIR = PROJECT_ROOT / "results"

# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "normdoc_architects")

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
# Провайдер генерации:
#   "openai"   — OpenAI-совместимый HTTP API: локальный llama.cpp, Ollama, vLLM,
#                а также локальный GigaChat/другие РФ-модели под llama-server;
#   "gigachat" — облачный API Sber GigaChat (OAuth + REST, сертификаты Минцифры).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "bonsai")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "65536"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "600"))
# 'none' | 'low' | 'medium' | 'high' | 'auto' (auto = не передавать параметр)
LLM_REASONING = os.getenv("LLM_REASONING", "none").strip().lower()

# --- GigaChat API (Sber) — используется при LLM_PROVIDER=gigachat ---
GIGACHAT_AUTH_URL = os.getenv(
    "GIGACHAT_AUTH_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
)
GIGACHAT_API_URL = os.getenv("GIGACHAT_API_URL", "https://gigachat.devices.sberbank.ru/api/v1").rstrip("/")
# base64(client_id:client_secret) из личного кабинета developers.sber.ru.
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS", "")
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat")
# Проверка TLS. Для облака Sber нужен корневой сертификат Минцифры; при его
# отсутствии можно временно выключить (GIGACHAT_VERIFY_SSL=0) — только для тестов.
GIGACHAT_VERIFY_SSL = os.getenv("GIGACHAT_VERIFY_SSL", "1") not in {"0", "false", "False", ""}


# ---------------------------------------------------------------------------
# Эмбеддинги
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto").strip().lower()
# Префиксы инструкций. Для e5 обязательны "query: "/"passage: "; для BGE-M3 — пустые.
EMBEDDING_QUERY_PREFIX = os.getenv("EMBEDDING_QUERY_PREFIX", "query: ")
EMBEDDING_PASSAGE_PREFIX = os.getenv("EMBEDDING_PASSAGE_PREFIX", "passage: ")
# Для моделей с кастомным кодом (Giga-Embeddings) нужен trust_remote_code=1.
EMBEDDING_TRUST_REMOTE_CODE = os.getenv("EMBEDDING_TRUST_REMOTE_CODE", "0") not in {"0", "false", "False", ""}
# Максимальная длина входа эмбеддера (0 = дефолт модели).
EMBEDDING_MAX_LENGTH = int(os.getenv("EMBEDDING_MAX_LENGTH", "0"))

# ---------------------------------------------------------------------------
# Чанкинг и ретривер
# ---------------------------------------------------------------------------
# Режим чанкинга: "semantic" — по смысловым границам (предложения/абзацы),
# 300..500 токенов с перекрытием 10..15%; "chars" — простой по символам.
CHUNKING_MODE = os.getenv("CHUNKING_MODE", "semantic").strip().lower()
# Параметры семантического чанкинга (в токенах tiktoken).
CHUNK_TARGET_TOKENS = int(os.getenv("CHUNK_TARGET_TOKENS", "400"))
CHUNK_MIN_TOKENS = int(os.getenv("CHUNK_MIN_TOKENS", "300"))
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "500"))
CHUNK_OVERLAP_RATIO = float(os.getenv("CHUNK_OVERLAP_RATIO", "0.12"))
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "cl100k_base")
# Параметры символьного чанкинга (fallback).
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "60"))
TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.0"))

# ---------------------------------------------------------------------------
# Реранкер (bge-reranker-v2-m3) — переупорядочивает кандидатов ретривера.
# ---------------------------------------------------------------------------
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "1") not in {"0", "false", "False", ""}
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "auto").strip().lower()
RERANK_BATCH = int(os.getenv("RERANK_BATCH", "8"))
# Сколько кандидатов достаём из Qdrant перед реранкингом (recall шире, чем top_k).
RETRIEVE_CANDIDATES = int(os.getenv("RETRIEVE_CANDIDATES", "20"))

# ---------------------------------------------------------------------------
# Парсер
# ---------------------------------------------------------------------------
PARSE_TABLES = os.getenv("PARSE_TABLES", "1") not in {"0", "false", "False", ""}
PARSE_FIGURES = os.getenv("PARSE_FIGURES", "1") not in {"0", "false", "False", ""}
SANITIZE_LINKS = os.getenv("SANITIZE_LINKS", "1") not in {"0", "false", "False", ""}

# ---------------------------------------------------------------------------
# Langfuse (мониторинг)
# ---------------------------------------------------------------------------
LANGFUSE_ENABLED = os.getenv("LANGFUSE_ENABLED", "0") not in {"0", "false", "False", ""}
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")


def effective_llm_ctx() -> int:
    """Реальный размер контекста llama.cpp (у bonsai - 65536, у быстрых моделей меньше)."""
    return int(os.getenv("LLM_CONTEXT_WINDOW", str(LLM_CONTEXT_WINDOW)))