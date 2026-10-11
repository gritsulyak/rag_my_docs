# normdoc-assistant

Приватный **RAG-ассистент по нормативным документам пользователя**
(национальные стандарты, регламенты, приказы, методические указания). Отвечает
на вопросы по **локальному корпусу документов со ссылками на источники**,
работает **полностью офлайн** на своём `llama.cpp` (Vulkan/CUDA/ROCm/CPU) и
поставляется как **docker-compose**-стек с самохостящимся **Langfuse** для
трейсинга.

> English version — in [`README.md`](README.md). Этот файл — русская
> документация.

---

## Содержание

1. [Зачем это нужно](#1-зачем-это-нужно)
2. [Архитектура](#2-архитектура)
3. [Быстрый старт (Docker)](#3-быстрый-старт-docker)
4. [Профили железа: AI MAX+ 395 и RTX 3060](#4-профили-железа-ai-max-395-и-rtx-3060)
5. [Локальный запуск без Docker](#5-локальный-запуск-без-docker)
6. [Модели: GGUF, GigaChat, каталог РФ-моделей](#6-модели-gguf-gigachat-каталог-рф-моделей)
7. [Корпус и индекс](#7-корпус-и-индекс)
8. [HTTP API и примеры запросов](#8-http-api-и-примеры-запросов)
9. [Мониторинг Langfuse](#9-мониторинг-langfuse)
10. [Оценка качества (RAGAS) и quality gate](#10-оценка-качества-ragas-и-quality-gate)
11. [Нагрузочное тестирование и графики](#11-нагрузочное-тестирование-и-графики)
12. [Тесты](#12-тесты)
13. [CI/CD](#13-cicd)
14. [Отчёт](#14-отчёт)
15. [Структура проекта](#15-структура-проекта)
16. [Конфигурация](#16-конфигурация)
17. [Решение проблем](#17-решение-проблем)
18. [Ограничения и допущения](#18-ограничения-и-допущения)

---

## 1. Зачем это нужно

**Задача.** Специалистам, работающим с нормативными документами (стандарты,
приказы, регламенты, методические указания), нужно быстро получать точные
ответы по базе документов, **не отправляя внутренние или закрытые документы в
облачную LLM**. Ответ должен быть **проверяемым**: с указанием документа и
страницы.

Ассистент **не привязан к конкретной предметной области** — он работает с
*вашим* корпусом документов. Направьте его на свои документы, и он будет
отвечать строго по ним.

**Что делает система:**

- парсит корпус (PDF/RTF/ODT, включая сканы через OCR), чистит и чанкует текст;
- строит векторный индекс в Qdrant (эмбеддинги — локальные, напр. `e5-small`);
- опционально реранкует кандидатов (`bge-reranker-v2-m3`);
- генерирует ответ локальной LLM через `llama.cpp` со строгим системным
  промптом «отвечай только по контексту»;
- **граундит цитаты**: каждая ссылка `[n]` сверяется с реальными метаданными
  извлечённых чанков, выдуманные источники удаляются;
- трейсит запросы в Langfuse (ретрив, генерация, latency, токены);
- измеряет качество (RAGAS) и производительность (perf + нагрузочные тесты с
  графиками).

---

## 2. Архитектура

```
                 ┌──────────────────────────────────────────────────────┐
   HTTP/CLI ───▶ │  app (FastAPI / scripts/chat.py)                     │
                 │  retrieval → rerank → prompt → LLM → cite-grounding   │
                 └───────┬───────────────┬───────────────────┬──────────┘
                         │               │                   │
                 ┌───────▼──────┐  ┌─────▼──────────┐  ┌─────▼──────────┐
                 │   Qdrant     │  │  llama.cpp     │  │   Langfuse     │
                 │ (векторы)    │  │  (Vulkan/CUDA/ │  │ (трейсы,       │
                 │ 6333         │  │   ROCm/CPU)    │  │  latency)      │
                 └──────────────┘  └────────────────┘  └────────────────┘
```

- **Слой LLM** (`src/normdoc/llm.py`) — единая точка входа `call_llm`,
  провайдеры `openai` (OpenAI-совместимый: llama.cpp/Ollama/vLLM/LM Studio) и
  `gigachat` (облако Sber, OAuth+REST).
- **Ретрив** (`src/normdoc/indexing.py`) — Qdrant + LlamaIndex, реранкер.
- **Ассистент** (`src/normdoc/assistant.py`) — оркестрация + замер latency.
- **Парсинг/чанкинг** (`src/normdoc/parsing.py`) — PyMuPDF, OCR-сайдкары,
  семантический чанкинг по токенам.
- **Цитаты** (`src/normdoc/prompts.py`) — `ground_citations` убирает
  галлюцинированные источники.

---

## 3. Быстрый старт (Docker)

Требуется Docker 24+ и Docker Compose v2. Стек — в [`docker-compose.yml`](docker-compose.yml).

```bash
# 1) Конфигурация
cp .env.example .env
# отредактируйте .env: как минимум укажите каталог моделей и LLM_MODEL

# 2) Скачайте GGUF-модель в ./models (пример — Meno-Lite; см. §6)
mkdir -p models
# ... положите, например, models/meno-lite-0.1-Q4_K_M.gguf

# 3) Поднимите стек под своё железо (см. §4)
docker compose --profile cpu --profile monitoring up -d --build

# 4) Подготовьте корпус и индекс
docker compose --profile tools run --rm app python scripts/prepare_docs.py
docker compose --profile tools run --rm app python scripts/build_index.py --reset \
    --collection normdoc_e5_small

# 5) Проверьте
curl -s localhost:8000/health
curl -s localhost:8000/answer -H 'Content-Type: application/json' \
     -d '{"question":"Что такое описание архитектуры?"}' | jq

# 6) Откройте Langfuse (трейсы) — http://localhost:3000
#    API — http://localhost:8000/docs
```

`Makefile` — короткие команды: `make up-ai395`, `make index`, `make test`,
`make ragas`, `make load`, `make report`.

---

## 4. Профили железа: AI MAX+ 395 и RTX 3060

`llama.cpp` — единственный движок инференса; бэкенд ускорения выбирается
**профилем docker-compose** (см. [`docker/llama/Dockerfile`](docker/llama/Dockerfile)).

| Профиль | Железо | Бэкенд llama.cpp | Команда |
|---|---|---|---|
| `ai395` | **AMD Ryzen AI MAX+ 395** / Radeon 8060S (Strix Halo, gfx1151) | **Vulkan** (`/dev/dri`, `/dev/kfd`) | `docker compose --profile ai395 up -d --build` |
| `rtx3060` | **NVIDIA RTX 3060 12 ГБ** (отдельный бокс, sm_86) | **CUDA** (nvidia runtime) | `docker compose --profile rtx3060 up -d --build` |
| `rocm` | AMD через ROCm/HIP | **HIP/ROCm** | `docker compose --profile rocm up -d --build` |
| `cpu` | любая машина | CPU | `docker compose --profile cpu up -d --build` |
| `monitoring` | — | self-hosted **Langfuse** | `docker compose --profile monitoring up -d` |
| `tools` | — | разовые задачи (индексация/RAGAS/load) | `docker compose --profile tools run --rm app …` |

**Vulkan vs ROCm на AMD.** По умолчанию для AI MAX+ 395 используется **Vulkan**:
он даёт стабильный запуск на RADV/Mesa без возни с версиями ROCm; ROCm-профиль
оставлен как альтернатива (нужен `AMDGPU_TARGETS=gfx1151`, задаётся в `.env`).

**RTX 3060.** Куда бы ни была вставлена карта (в тот же хост или в отдельный
«карточный» бокс), запуск одинаков: `--profile rtx3060`. На хосте нужен
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Если карта в отдельном боксе — запускайте compose на том боксе и направьте
`app` на него (`LLM_BASE_URL=http://<бокс>:8080/v1`).

Параметры llama.cpp задаются переменными: `MODELS_DIR`, `MODEL_FILE`,
`MODEL_ALIAS`, `N_CTX`, `N_GPU_LAYERS` (99 — все слои на GPU, 0 — CPU),
`N_PARALLEL`, `THREADS`.

---

## 5. Локальный запуск без Docker

Нужны Python 3.12+, [`uv`](https://github.com/astral-sh/uv), Qdrant и
`llama-server` на хосте.

```bash
uv sync --extra perf                # зависимости + графики
cp .env.example .env                # QDRANT_URL=http://localhost:6333, LLM_BASE_URL=...

# Qdrant (если не запущен)
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:v1.13.3

# локальный llama-server (Vulkan-сборка llama.cpp)
scripts/serve_llm.sh meno           # или: scripts/start_meno.sh
# либо любой OpenAI-совместимый сервер на LLM_BASE_URL

# корпус + индекс
uv run python scripts/prepare_docs.py
uv run python scripts/build_index.py --reset --collection normdoc_e5_small

# интерактивный чат
uv run python scripts/chat.py -q "Что такое описание архитектуры?"

# HTTP-сервис
uv run python scripts/serve_api.py --port 8000
```

---

## 6. Модели: GGUF, GigaChat, каталог РФ-моделей

**Каталог рекомендованных моделей** (с размерами, лицензиями и ссылками):

```bash
uv run python scripts/list_models.py --kind llm
uv run python scripts/list_models.py --kind embedding --format markdown
```

В каталоге — локальные РФ-модели для GGUF (`llama.cpp`) и облачные:
GigaChat, T-lite, QVikhr, RuadaptQwen, Saiga, Qwen, Meno-Lite и др.
Полное описание — в разделе «Каталог» итогового отчёта.

**Локальный запуск (GGUF).** Положите `.gguf` в `./models` (или в каталог из
`MODELS_DIR`) и задайте `MODEL_FILE`/`MODEL_ALIAS`. Профили LLM —
в [`configs/llms/`](configs/llms/) (`bonsai`, `meno`, `t_lite`, `qvikhr`,
`ruadapt`, `saiga`, `qwen`, `gigachat`, `gigachat_api`, `avito`, `fast_local`).

> **Про Avito A-vibe.** Веса модели закрыты, публичного GGUF нет. Поэтому она
> представлена в каталоге как **OpenAI-совместимый эндпоинт** (профиль
> `configs/llms/avito.env`): если у вас есть доступ к API/весам, укажите
> `LLM_BASE_URL` и `LLM_MODEL`, код менять не нужно.

**Облачный GigaChat (Sber).**

```bash
LLM_PROVIDER=gigachat
GIGACHAT_CREDENTIALS=<base64(client_id:client_secret)>
GIGACHAT_SCOPE=GIGACHAT_API_PERS
GIGACHAT_VERIFY_SSL=1     # нужен корневой сертификат Минцифры; 0 — для тестов
```

---

## 7. Корпус и индекс

**Примеры входных данных.** Корпус документов **приватный и не публикуется** в
репозитории: положите свои документы в `data/raw/sources/` (PDF/RTF/ODT; сканы
распознаются OCR). В репозитории есть примеры *вопросов* и HTTP-запросов:

- вопросы: [`data/qa/golden_qa.jsonl`](data/qa/golden_qa.jsonl) — 12 пар
  «вопрос → эталонный ответ» (на русском, по документам примерного корпуса);
- HTTP-примеры: [`examples/requests.http`](examples/requests.http),
  [`examples/README.md`](examples/README.md).

```bash
# конвертация (RTF/ODT→PDF), OCR сканов, чистка, manifest
uv run python scripts/prepare_docs.py           # --force для повтора

# построение индекса
uv run python scripts/build_index.py --reset --collection normdoc_e5_small
#   --mode chars --chunk-size 600 --chunk-overlap 60   # символьный чанкинг
#   --from-cache                                       # из data/chunks/chunks.json
```

Эмбеддинги переключаются профилем `configs/embeddings/*.env`
(`e5_small`, `bge_m3`, `giga_embeddings`, `user_bge_m3`, `roberta`).

---

## 8. HTTP API и примеры запросов

Сервис — `src/normdoc/api.py` (FastAPI). Интерактивная документация:
`http://localhost:8000/docs`.

| Метод | Путь | Назначение |
|---|---|---|
| `GET` | `/health` | статус, коллекция, модель |
| `POST`| `/answer` | вопрос → ответ, источники, latency, usage |
| `GET` | `/metrics` | метрики Prometheus (latency/throughput) |
| `POST`| `/warmup` | прогрев (загрузка моделей в память) |

```bash
curl -s localhost:8000/answer -H 'Content-Type: application/json' -d '{
  "question": "Каким приказом утверждён ГОСТ Р 57100—2025?",
  "top_k": 5
}' | jq
```

Ответ:

```json
{
  "answer": "Стандарт утверждён приказом Росстандарта от 25.03.2025 N 215-ст и введён в действие с 30 июня 2025 года [ГОСТ Р 57100—2025, стр. 27].",
  "sources": [{"source": "ГОСТ Р 57100—2025", "page": 27, "score": 0.83, "snippet": "…"}],
  "latency_s": 14.2, "retrieval_s": 0.4, "generation_s": 13.8,
  "usage": {"prompt_tokens": 1200, "completion_tokens": 64, "total_tokens": 1264},
  "dropped_citations": [], "collection": "normdoc_e5_small", "model": "meno"
}
```

CLI-чат: `uv run python scripts/chat.py` (команды `:sources`, `:quit`).

---

## 9. Мониторинг Langfuse

Самохостящийся Langfuse поднимается профилем `monitoring` (Postgres,
ClickHouse, Redis, MinIO, web, worker). Организация/проект/ключи создаются
автоматически (`LANGFUSE_INIT_*` в compose), ключи берутся из `.env`.

```bash
docker compose --profile monitoring up -d
# UI: http://localhost:3000  (admin@example.com / password123 — из .env)

# включите трейсинг в приложении
# .env: LANGFUSE_ENABLED=1; LANGFUSE_HOST=http://langfuse-web:3000 (в compose)
#       LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY — как в compose-проекте
docker compose --profile cpu up -d app

# экспорт последних трейсов в отчёт
uv run python scripts/langfuse_report.py --limit 50
```

Каждый запрос = один trace `rag_answer` с дочерними спанами `retrieve`
(тип `retriever`) и `generate` (тип `generation`, с токенами). В отчёт
(`results/langfuse_traces.json` → `results/REPORT.md`) попадают id трейса,
вопрос, латентность и разбивка по спанам.

**Fallback без сервера.** Если Langfuse-сервер недоступен (например, на
слабой dev-машине), включите запись трейсов в JSONL — данные те же
(вопрос, спаны, latency, токены):

```bash
TRACE_JSONL=results/traces_local.jsonl uv run python scripts/chat.py -q "…"
uv run python scripts/langfuse_report.py        # возьмёт локальный JSONL
```

---

## 10. Оценка качества (RAGAS) и quality gate

Оценка — на **golden-наборе** `data/qa/golden_qa.jsonl`, судья — локальная LLM
(`--judge-url`, по умолчанию второй llama-server на `:8081`).

```bash
# одиночный прогон (все 12 вопросов)
uv run python scripts/evaluate_ragas.py --limit 12

# с quality gate (ненулевой код возврата, если пороги не достигнуты)
uv run python scripts/evaluate_ragas.py --gate
# пороги в .env: THRESH_FAITHFULNESS / THRESH_CONTEXT_PRECISION / THRESH_CONTEXT_RECALL

# сравнение конфигураций (chunking × эмбеддинги × реранкер × LLM)
uv run python scripts/compare_configs.py --limit 6 --with-perf
```

Метрики: **Faithfulness**, **ContextPrecision**, **ContextRecall**.
Результаты: `results/comparison_ragas.json` (+ `results/ragas_*.json`).
`--gate` удобно использовать в CI как объективный критерий приёмки.

---

## 11. Нагрузочное тестирование и графики

```bash
# поднять API (docker или локально), затем:
uv run python scripts/load_test.py --url http://localhost:8000 \
    --requests 40 --concurrency 4 --name local
```

Скрипт измеряет **RPS**, перцентили латентности (p50/p90/p95/p99), долю ошибок
и токены/с, и **строит графики** (латентность по запросам, распределение,
перцентили) → `results/load_local.json` и `results/load_local.png`. Графики
встраиваются в итоговый отчёт.

Perf-тесты (ретрив/генерация/ток-с по конфигурациям):

```bash
uv run python scripts/benchmark_rag.py --limit 6 --repeat 2 --name meno
```

---

## 12. Тесты

```bash
uv run python scripts/run_tests.py            # → results/pytest.txt
uv run python scripts/run_tests.py -k TestApi -v
```

Покрытие (`tests/`): парсинг/OCR/таблицы, чанкинг, цитирование и grounding,
payload LLM (`reasoning_effort`), провайдеры LLM (`openai`/`gigachat`, кэш
токена), golden-QA и статистика, отчётность, Langfuse-мониторинг (no-op и
offline), HTTP-API (FastAPI `TestClient`).

---

## 13. CI/CD

[`.github/workflows/ci.yml`](.github/workflows/ci.yml):

- **tests** — линт (ruff, не блокирующий) + pytest, артефакт `results/pytest.txt`;
- **docker** — матричная сборка образов `app` и `llama-cpu`, публикация в GHCR
  с тегами (`branch`, `sha`, `latest`);
- **ragas-load** — на **self-hosted** раннере (GPU/CPU): поднимает стек,
  строит индекс, гоняет RAGAS с `--gate`, нагрузочный тест и отчёт; выгружает
  артефакты. Запуск вручную (`workflow_dispatch`).

---

## 14. Отчёт

```bash
uv run python scripts/langfuse_report.py      # трейсы (если мониторинг включён)
uv run python scripts/report.py               # → results/REPORT.md
```

`results/REPORT.md` объединяет: стенд, корпус, RAGAS по конфигурациям,
perf-таблицу, нагрузочные тесты с графиками, трейсы Langfuse и latency,
тесты, каталог РФ-моделей и раздел про Docker/сервинг.

---

## 15. Структура проекта

```
├── docker-compose.yml          # qdrant + llama (профили) + app + langfuse + tools
├── Makefile                    # короткие команды запуска
├── docker/
│   ├── app/Dockerfile          # образ RAG-приложения (FastAPI)
│   └── llama/{Dockerfile,entrypoint.sh}   # llama.cpp: Vulkan/CUDA/ROCm/CPU
├── configs/
│   ├── hw/                     # профили железа (ai395, amd_7840hs, rtx3060)
│   ├── llms/                   # профили LLM (bonsai, meno, gigachat, avito, …)
│   └── embeddings/             # профили эмбеддингов (e5_small, bge_m3, …)
├── src/normdoc/                # код: assistant, llm, api, indexing, parsing,
│                               #   prompts, monitoring, reporting, qa, model_catalog
├── scripts/                    # prepare_docs, build_index, chat, serve_api,
│                               #   evaluate_ragas, compare_configs, benchmark_rag,
│                               #   load_test, langfuse_report, report, run_tests, list_models
├── data/
│   ├── raw/sources/            # ваши документы (приватные, не публикуются)
│   └── qa/golden_qa.jsonl      # примеры тестовых вопросов/эталонов
├── examples/                   # примеры HTTP-запросов
├── tests/                      # pytest: ядро + llmops
└── results/                    # артефакты: ragas, perf, load, langfuse, REPORT.md
```

---

## 16. Конфигурация

Все настройки — через `.env` (шаблон: [`.env.example`](.env.example)); секреты в
git не попадают. Профили `configs/{hw,llms,embeddings}` выбираются переменными
`HW_PROFILE` / `LLM_PROFILE` / `EMBEDDING_PROFILE`.

Ключевые группы: Qdrant, корпус (`SOURCES_DIR`), провайдер LLM
(`LLM_PROVIDER/LLM_BASE_URL/…`, GigaChat), эмбеддинги, реранкер, чанкинг,
парсер, Langfuse, Docker (`MODELS_DIR`, `MODEL_FILE`, `APP_PORT`), пороги RAGAS.

---

## 17. Решение проблем

| Симптом | Причина / решение |
|---|---|
| `GGUF не найден` | пусто в `MODELS_DIR`; задайте `MODEL_FILE=/models/<файл>.gguf` |
| `app` не видит LLM | не поднят llama-профиль (`ai395`/`rtx3060`/`cpu`); проверьте `docker compose ps` |
| Нет GPU в контейнере | для Vulkan/ROCm — `group_add: video,render` и устройства `/dev/dri`,`/dev/kfd`; для CUDA — nvidia-container-toolkit |
| Langfuse пуст | `LANGFUSE_ENABLED=1`, `LANGFUSE_HOST=http://langfuse-web:3000`, ключи совпадают с `LANGFUSE_INIT_*` |
| RAGAS низкий recall | увеличьте `TOP_K`/`RETRIEVE_CANDIDATES`, включите реранкер, проверьте чанкинг |
| Медленная генерация | вынесите больше слоёв на GPU (`N_GPU_LAYERS`), уменьшите `N_CTX`, поднимите `N_PARALLEL` |

---

## 18. Ограничения и допущения

- **Железо разработки** — AMD Ryzen 7 7840HS + Radeon 780M (CPU-инференс).
  Профили `ai395` (Vulkan) и `rtx3060` (CUDA) поставляются как конфигурации и
  собираются на целевом железе; фактические замеры в отчёте сделаны на доступном
  железе.
- **Avito A-vibe** — закрытые веса → представлена как OpenAI-совместимый
  эндпоинт-профиль.
- **Наименьшие открытые веса GigaChat** — MoE 10B/1.8B и 20B/3B (не строго
  5–8B); в каталоге это отражено.
- **Langfuse на dev-машине.** Полный self-hosted стек Langfuse (Postgres +
  ClickHouse + Redis + MinIO + web + worker) описан в `docker-compose.yml`
  (профиль `monitoring`) и рассчитан на целевые боксы (AI MAX+ 395, 128 ГБ).
  На слабой dev-машине он не поднимался (недостаток RAM); трейсы для отчёта
  собраны через встроенный JSONL-сток (`TRACE_JSONL`), формат совпадает с
  `scripts/langfuse_report.py`.
- **Корпус документов** — корпус принадлежит пользователю и не публикуется в
  репозитории; положите свои документы в `data/raw/sources/`.
- Сканы распознаются OCR (tesseract); качество зависит от исходника.
