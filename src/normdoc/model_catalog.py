"""Каталог рекомендованных РФ-моделей для локального RAG.

Сводка лучших русскоязычных моделей «малого» класса (ориентир — 5-8B
параметров, плюс MoE с малым числом активных параметров), которые реально
запускаются на потребительском железе (CPU / iGPU / 8-12 ГБ VRAM) и хорошо
себя показывают в RAG-задачах: генерация ответа строго по контексту, русский
язык, цитирование источников.

Каталог используется в отчёте (``scripts/report.py``), в CLI
(``scripts/list_models.py``) и как справочник при выборе профиля в
``configs/llms`` и ``configs/embeddings``.

Поля:
    key           — короткий идентификатор;
    name          — человекочитаемое имя;
    org           — организация/вендор;
    kind          — llm | embedding | reranker;
    params_b      — общее число параметров, млрд (может быть диапазоном);
    active_b      — активных параметров на токен (для MoE), иначе None;
    context       — контекстное окно, токенов;
    languages     — поддерживаемые языки (ru/en/multi);
    license       — лицензия (если известна);
    hf            — репозиторий на HuggingFace;
    gguf          — есть ли GGUF-кванты (для llama.cpp/Ollama);
    serve         — рекомендуемый способ запуска (llama.cpp / ollama / api);
    note          — короткое примечание (особенно про RAG/локальное железо).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelInfo:
    key: str
    name: str
    org: str
    kind: str
    params_b: str
    active_b: str | None
    context: int
    languages: str
    license: str
    hf: str
    gguf: bool
    serve: str
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


CATALOG: list[ModelInfo] = [
    # ---- GigaChat (Sber, ai-sage) -----------------------------------------
    ModelInfo(
        key="gigachat3-10b-a1.8b",
        name="GigaChat3.1-10B-A1.8B",
        org="Sber / ai-sage",
        kind="llm",
        params_b="10",
        active_b="1.8",
        context=262144,
        languages="ru,en",
        license="см. HF",
        hf="ai-sage/GigaChat3.1-10B-A1.8B",
        gguf=True,
        serve="llama.cpp / ollama",
        note="MoE: 10B всего, ~1.8B активных — лучший локальный баланс качества и скорости для РФ-RAG.",
    ),
    ModelInfo(
        key="gigachat-20b-a3b",
        name="GigaChat-20B-A3B-instruct-v1.5",
        org="Sber / ai-sage",
        kind="llm",
        params_b="21",
        active_b="3",
        context=131072,
        languages="ru,en",
        license="см. HF",
        hf="ai-sage/GigaChat-20B-A3B-instruct-v1.5",
        gguf=True,
        serve="llama.cpp / ollama",
        note="Более сильная инструкт-версия; 20B/3B активных, влезает в 12 ГБ VRAM при Q4.",
    ),
    ModelInfo(
        key="gigachat-api",
        name="GigaChat (облачный API)",
        org="Sber",
        kind="llm",
        params_b="—",
        active_b=None,
        context=131072,
        languages="ru,en",
        license="проприетарная (API)",
        hf="",
        gguf=False,
        serve="api",
        note="LLM_PROVIDER=gigachat, OAuth + REST. Не требует локального железа.",
    ),
    # ---- Avito ------------------------------------------------------------
    ModelInfo(
        key="avito-a-vibe",
        name="Avito A-vibe",
        org="Avito (AvitoTech)",
        kind="llm",
        params_b="—",
        active_b=None,
        context=32768,
        languages="ru",
        license="проприетарная (веса не опубликованы)",
        hf="",
        gguf=False,
        serve="api / OpenAI-совместимый эндпоинт",
        note="Дообученная под задачи Авито РФ-модель. Веса закрыты — подключается через внутренний OpenAI-совместимый API (см. configs/llms/avito.env).",
    ),
    # ---- Специализированные RAG-модели 5-8B -------------------------------
    ModelInfo(
        key="meno-lite",
        name="Meno-Lite-0.1",
        org="Meno AI",
        kind="llm",
        params_b="7",
        active_b=None,
        context=32768,
        languages="ru,en",
        license="см. HF",
        hf="Menlo/Meno-Lite-0.1",
        gguf=True,
        serve="llama.cpp / ollama",
        note="Специализирована под RAG: короткие ответы строго по контексту. Уже настроена в проекте (порт 8081).",
    ),
    ModelInfo(
        key="t-lite-it-2.1",
        name="T-lite-it-2.1",
        org="T-Bank (t-tech)",
        kind="llm",
        params_b="8",
        active_b=None,
        context=40960,
        languages="ru",
        license="см. HF",
        hf="t-tech/T-lite-it-2.1",
        gguf=True,
        serve="llama.cpp / ollama",
        note="Qwen3-8B база, сильный русский инструкт; хорошо держит формат цитат.",
    ),
    ModelInfo(
        key="qvikhr-3-8b",
        name="QVikhr-3-8B-Instruction",
        org="Vikhrmodels",
        kind="llm",
        params_b="8",
        active_b=None,
        context=40960,
        languages="ru,en",
        license="Apache-2.0",
        hf="Vikhrmodels/QVikhr-3-8B-Instruction",
        gguf=True,
        serve="llama.cpp / ollama",
        note="Qwen3-8B + русский SFT/DPO. Один из лучших открытых 8B для РФ-задач.",
    ),
    ModelInfo(
        key="ruadapt-qwen3-8b",
        name="RuadaptQwen3-8B-Hybrid",
        org="RefalMachine",
        kind="llm",
        params_b="8",
        active_b=None,
        context=40960,
        languages="ru,en",
        license="Apache-2.0",
        hf="RefalMachine/RuadaptQwen3-8B-Hybrid",
        gguf=True,
        serve="llama.cpp / ollama",
        note="Адаптация словаря/эмбеддингов Qwen3 под русский — дешевле в токенах на русском тексте.",
    ),
    ModelInfo(
        key="ruadapt-qwen2.5-7b",
        name="ruadapt_qwen2.5_7B_instruct",
        org="RefalMachine",
        kind="llm",
        params_b="7",
        active_b=None,
        context=32768,
        languages="ru,en",
        license="Apache-2.0",
        hf="RefalMachine/ruadapt_qwen2.5_7B_ext_u48_instruct",
        gguf=True,
        serve="llama.cpp / ollama",
        note="Проверенный 7B-бейзлайн для русского, много GGUF-квантов.",
    ),
    ModelInfo(
        key="saiga-llama3-8b",
        name="Saiga Llama3 8B",
        org="IlyaGusev",
        kind="llm",
        params_b="8",
        active_b=None,
        context=8192,
        languages="ru",
        license="см. HF",
        hf="IlyaGusev/saiga_llama3_8b_gguf",
        gguf=True,
        serve="llama.cpp / ollama",
        note="Классическая русская 8B-инструкт модель, стабильный GGUF.",
    ),
    ModelInfo(
        key="vikhr-yandexgpt-5-lite-8b",
        name="Vikhr-YandexGPT-5-Lite-8B-it",
        org="Vikhrmodels",
        kind="llm",
        params_b="8",
        active_b=None,
        context=32768,
        languages="ru,en",
        license="см. HF",
        hf="Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it",
        gguf=True,
        serve="llama.cpp / ollama",
        note="Открытая 8B на базе YandexGPT-5-Lite, адаптированная Vikhr.",
    ),
    ModelInfo(
        key="qwen2.5-7b-instruct",
        name="Qwen2.5-7B-Instruct",
        org="Alibaba",
        kind="llm",
        params_b="7",
        active_b=None,
        context=32768,
        languages="multi (включая ru)",
        license="Apache-2.0",
        hf="Qwen/Qwen2.5-7B-Instruct",
        gguf=True,
        serve="llama.cpp / ollama",
        note="Нейтральный мультиязычный бейзлайн для сравнения с РФ-моделями.",
    ),
    ModelInfo(
        key="bonsai-2-27b",
        name="Ternary Bonsai 2 27B",
        org="PrismML",
        kind="llm",
        params_b="27",
        active_b=None,
        context=65536,
        languages="multi",
        license="см. HF",
        hf="",
        gguf=True,
        serve="llama.cpp",
        note="Текущая дефолтная модель проекта (ternary 2.13 бит/вес, ~7 ГБ). Сильнее на сложных вопросах, но ~9 ток/с на CPU.",
    ),
    # ---- Эмбеддинги -------------------------------------------------------
    ModelInfo(
        key="bge-m3",
        name="BAAI/bge-m3",
        org="BAAI",
        kind="embedding",
        params_b="0.57",
        active_b=None,
        context=8192,
        languages="multi",
        license="MIT",
        hf="BAAI/bge-m3",
        gguf=False,
        serve="sentence-transformers",
        note="Мультиязычный SOTA-бейзлайн ретривера, 1024d. Дефолт для bge-профиля.",
    ),
    ModelInfo(
        key="e5-small",
        name="multilingual-e5-small",
        org="intfloat",
        kind="embedding",
        params_b="0.12",
        active_b=None,
        context=512,
        languages="multi",
        license="MIT",
        hf="intfloat/multilingual-e5-small",
        gguf=False,
        serve="sentence-transformers",
        note="Быстрый CPU-бейзлайн (384d). Дефолт для e5-профиля.",
    ),
    ModelInfo(
        key="giga-embeddings",
        name="Giga-Embeddings-instruct",
        org="Sber / ai-sage",
        kind="embedding",
        params_b="—",
        active_b=None,
        context=8192,
        languages="ru,en",
        license="см. HF",
        hf="ai-sage/Giga-Embeddings-instruct",
        gguf=False,
        serve="sentence-transformers (trust_remote_code)",
        note="Русскоязычный SOTA-эмбеддер Sber; 3B-вариант качественнее bge-m3 на ru, но тяжелее.",
    ),
    ModelInfo(
        key="user-bge-m3",
        name="USER-bge-m3",
        org="deepvk",
        kind="embedding",
        params_b="0.57",
        active_b=None,
        context=8194,
        languages="ru,en",
        license="MIT",
        hf="deepvk/USER-bge-m3",
        gguf=False,
        serve="sentence-transformers",
        note="Русская дообученная BGE-M3 (1024d) — дешёвая замена bge-m3 для ru-корпуса.",
    ),
    ModelInfo(
        key="roberta-ru-en",
        name="ru-en-RoSBERTa",
        org="ai-forever",
        kind="embedding",
        params_b="0.4",
        active_b=None,
        context=512,
        languages="ru,en",
        license="MIT",
        hf="ai-forever/ru-en-RoSBERTa",
        gguf=False,
        serve="sentence-transformers",
        note="Классический русский ретривер (512 токенов) — только для коротких чанков.",
    ),
    # ---- Реранкеры --------------------------------------------------------
    ModelInfo(
        key="bge-reranker-v2-m3",
        name="BAAI/bge-reranker-v2-m3",
        org="BAAI",
        kind="reranker",
        params_b="0.57",
        active_b=None,
        context=8192,
        languages="multi",
        license="Apache-2.0",
        hf="BAAI/bge-reranker-v2-m3",
        gguf=False,
        serve="sentence-transformers CrossEncoder",
        note="Дефолтный cross-encoder проекта; заметно повышает ContextPrecision.",
    ),
    ModelInfo(
        key="giga-retrieval",
        name="Giga-Retrieval-instruct",
        org="Sber / ai-sage",
        kind="reranker",
        params_b="—",
        active_b=None,
        context=8192,
        languages="ru,en",
        license="см. HF",
        hf="ai-sage/Giga-Retrieval-instruct",
        gguf=False,
        serve="transformers (trust_remote_code)",
        note="Русский retrieval/rerank Sber — альтернатива bge-reranker для ru-корпуса.",
    ),
]


def by_kind(kind: str) -> list[ModelInfo]:
    return [m for m in CATALOG if m.kind == kind]


def get(key: str) -> ModelInfo | None:
    for model in CATALOG:
        if model.key == key:
            return model
    return None
