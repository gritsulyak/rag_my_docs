# normdoc-assistant

A private **RAG assistant over a user's own normative documents** (national
standards, regulations, government orders, industry guidance). It answers
questions about a **local document corpus with references to the sources**,
runs **fully offline** on its own `llama.cpp` backend (Vulkan / CUDA / ROCm /
CPU), and ships as a **docker-compose** stack with a self-hosted **Langfuse**
for tracing.

> Русская версия — в [`README_RU.md`](README_RU.md). This file is the English
> documentation.

---

## Table of contents

1. [Why](#1-why)
2. [Architecture](#2-architecture)
3. [Quick start (Docker)](#3-quick-start-docker)
4. [Hardware profiles: AI MAX+ 395 and RTX 3060](#4-hardware-profiles-ai-max-395-and-rtx-3060)
5. [Local run without Docker](#5-local-run-without-docker)
6. [Models: GGUF, GigaChat, RU model catalog](#6-models-gguf-gigachat-ru-model-catalog)
7. [Corpus and index](#7-corpus-and-index)
8. [HTTP API and example requests](#8-http-api-and-example-requests)
9. [Langfuse monitoring](#9-langfuse-monitoring)
10. [Quality evaluation (RAGAS) and quality gate](#10-quality-evaluation-ragas-and-quality-gate)
11. [Load testing and graphs](#11-load-testing-and-graphs)
12. [Tests](#12-tests)
13. [CI/CD](#13-cicd)
14. [Report](#14-report)
15. [Project structure](#15-project-structure)
16. [Configuration](#16-configuration)
17. [Troubleshooting](#17-troubleshooting)
18. [Limitations and assumptions](#18-limitations-and-assumptions)

---

## 1. Why

**The task.** People who work with regulations and standards need fast, precise
answers from a document base (standards, orders, regulations, guidance notes)
**without sending internal or restricted documents to a cloud LLM**. Every
answer must be **verifiable**: it must point to the exact document and page.

The assistant is **domain-agnostic** — it is bound to *your* document corpus,
not to any particular subject area. Point it at your own documents and it
answers strictly from them.

**What the system does:**

- parses a corpus (PDF/RTF/ODT, including scans via OCR), cleans and chunks text;
- builds a vector index in Qdrant (embeddings are local, e.g. `e5-small`);
- optionally reranks candidates (`bge-reranker-v2-m3`);
- generates an answer with a local LLM via `llama.cpp` and a strict
  "answer only from the context" system prompt;
- **grounds citations**: every `[n]` reference is checked against the real
  metadata of the retrieved chunks, and invented sources are dropped;
- traces requests in Langfuse (retrieval, generation, latency, tokens);
- measures quality (RAGAS) and performance (perf + load tests with graphs).

---

## 2. Architecture

```
                 ┌──────────────────────────────────────────────────────┐
   HTTP/CLI ───▶ │  app (FastAPI / scripts/chat.py)                     │
                 │  retrieval → rerank → prompt → LLM → cite-grounding   │
                 └───────┬───────────────┬───────────────────┬──────────┘
                         │               │                   │
                 ┌───────▼──────┐  ┌─────▼──────────┐  ┌─────▼──────────┐
                 │   Qdrant     │  │  llama.cpp     │  │   Langfuse     │
                 │  (vectors)   │  │  (Vulkan/CUDA/ │  │ (traces,       │
                 │  6333        │  │   ROCm/CPU)    │  │  latency)      │
                 └──────────────┘  └────────────────┘  └────────────────┘
```

- **LLM layer** (`src/normdoc/llm.py`) — a single entry point `call_llm` with
  providers `openai` (OpenAI-compatible: llama.cpp/Ollama/vLLM/LM Studio) and
  `gigachat` (Sber cloud, OAuth+REST).
- **Retrieval** (`src/normdoc/indexing.py`) — Qdrant + LlamaIndex, reranker.
- **Assistant** (`src/normdoc/assistant.py`) — orchestration + latency metering.
- **HTTP & web UI** (`src/normdoc/api.py`, `src/normdoc/webui.py`) — FastAPI
  `POST /answer` plus a built-in chat page (`/`) and a no-JS HTML fallback
  (`GET /ask`) for console browsers.
- **Parsing/chunking** (`src/normdoc/parsing.py`) — PyMuPDF, OCR sidecars,
  token-based semantic chunking.
- **Citations** (`src/normdoc/prompts.py`) — `ground_citations` removes
  hallucinated sources.

---

## 3. Quick start (Docker)

Requires Docker 24+ and Docker Compose v2. Stack: [`docker-compose.yml`](docker-compose.yml).

```bash
# 1) Configuration
cp .env.example .env
# edit .env: at minimum set the models directory and LLM_MODEL

# 2) Download a GGUF model into ./models (example — Meno-Lite; see §6)
mkdir -p models
# ... e.g. put models/meno-lite-0.1-Q4_K_M.gguf there

# 3) Bring up the stack for your hardware (see §4)
docker compose --profile cpu --profile monitoring up -d --build

# 4) Prepare the corpus and the index
docker compose --profile tools run --rm app python scripts/prepare_docs.py
docker compose --profile tools run --rm app python scripts/build_index.py --reset \
    --collection normdoc_e5_small

# 5) Check it
curl -s localhost:8000/health
curl -s localhost:8000/answer -H 'Content-Type: application/json' \
     -d '{"question":"What is an architecture description?"}' | jq

# 6) Open the web chat — http://localhost:8000/
#    Langfuse (traces) — http://localhost:3000
#    API docs — http://localhost:8000/docs
```

The `Makefile` offers short commands: `make up-ai395`, `make up-7840hs`,
`make index`, `make test`, `make ragas`, `make load`, `make report`.

---

## 4. Hardware profiles: AI MAX+ 395, 7840HS and RTX 3060

`llama.cpp` is the only inference engine; the acceleration backend is chosen by
the **docker-compose profile** (see [`docker/llama/Dockerfile`](docker/llama/Dockerfile)).

| Profile | Hardware | llama.cpp backend | Command |
|---|---|---|---|
| `ai395` | **AMD Ryzen AI MAX+ 395** / Radeon 8060S (Strix Halo, gfx1151) | **Vulkan**, built from source (`/dev/dri`, `/dev/kfd`) | `docker compose --profile ai395 up -d --build` |
| `amd7840hs` | **AMD Ryzen 7 7840HS** / Radeon 780M iGPU (gfx1103) | **Vulkan**, prebuilt PrismML binaries (`/dev/dri`) | `docker compose --profile amd7840hs up -d --build` |
| `rtx3060` | **NVIDIA RTX 3060 12 GB** (separate box, sm_86) | **CUDA** (nvidia runtime) | `docker compose --profile rtx3060 up -d --build` |
| `rocm` | AMD via ROCm/HIP | **HIP/ROCm** | `docker compose --profile rocm up -d --build` |
| `cpu` | any machine | CPU | `docker compose --profile cpu up -d --build` |
| `monitoring` | — | self-hosted **Langfuse** | `docker compose --profile monitoring up -d` |
| `tools` | — | one-off jobs (indexing/RAGAS/load) | `docker compose --profile tools run --rm app …` |

**Vulkan vs ROCm on AMD.** The AI MAX+ 395 defaults to **Vulkan**: it starts
reliably on RADV/Mesa without chasing ROCm versions; the ROCm profile is kept as
an alternative (needs `AMDGPU_TARGETS=gfx1151`, set in `.env`).

**`amd7840hs` — prebuilt PrismML Vulkan (no source build).** On a laptop/mini-PC
with the Radeon 780M iGPU, compiling `llama.cpp` from source is slow, so this
profile reuses a **prebuilt PrismML Vulkan release** of `llama.cpp`. Point
`PRISM_VULKAN_DIR` (`.env`, default `../../llm_eng/prism-vulkan`) at a directory
that contains `llama-prism-*/llama-server` (or the extracted
`prism-vulkan.tar.gz`); it is mounted read-only into the container. The runtime
image only ships the Vulkan loader and Mesa RADV
([`docker/llama/Dockerfile.prism`](docker/llama/Dockerfile.prism)); the entrypoint
locates `llama-server` and adds its `.so` files to `LD_LIBRARY_PATH`. The same
host binaries are used by `scripts/serve_llm.sh` / `scripts/start_meno.sh` for
local (non-Docker) runs.

**RTX 3060.** Wherever the card lives (same host or a separate "card" box), the
launch is identical: `--profile rtx3060`. The host needs
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
If the card is in a separate box, run compose on that box and point `app` at it
(`LLM_BASE_URL=http://<box>:8080/v1`).

llama.cpp parameters are set by variables: `MODELS_DIR`, `MODEL_FILE`,
`MODEL_ALIAS`, `N_CTX`, `N_GPU_LAYERS` (99 — all layers on GPU, 0 — CPU),
`N_PARALLEL`, `THREADS`.

---

## 5. Local run without Docker

You need Python 3.12+, [`uv`](https://github.com/astral-sh/uv), Qdrant and
`llama-server` on the host.

```bash
uv sync --extra perf                # dependencies + plotting
cp .env.example .env                # QDRANT_URL=http://localhost:6333, LLM_BASE_URL=...

# Qdrant (if not running)
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:v1.13.3

# local llama-server (Vulkan build of llama.cpp)
scripts/serve_llm.sh meno           # or: scripts/start_meno.sh
# or any OpenAI-compatible server on LLM_BASE_URL

# corpus + index
uv run python scripts/prepare_docs.py
uv run python scripts/build_index.py --reset --collection normdoc_e5_small

# interactive chat
uv run python scripts/chat.py -q "What is an architecture description?"

# HTTP service
uv run python scripts/serve_api.py --port 8000
```

---

## 6. Models: GGUF, GigaChat, RU model catalog

**Recommended-model catalog** (sizes, licenses and links):

```bash
uv run python scripts/list_models.py --kind llm
uv run python scripts/list_models.py --kind embedding --format markdown
```

The catalog covers local RU models for GGUF (`llama.cpp`) and cloud ones:
GigaChat, T-lite, QVikhr, RuadaptQwen, Saiga, Qwen, Meno-Lite, and more. The
full description is in the "Catalog" section of the final report.

**Local run (GGUF).** Put a `.gguf` into `./models` (or the directory from
`MODELS_DIR`) and set `MODEL_FILE`/`MODEL_ALIAS`. LLM profiles live in
[`configs/llms/`](configs/llms/) (`bonsai`, `meno`, `t_lite`, `qvikhr`,
`ruadapt`, `saiga`, `qwen`, `gigachat`, `gigachat_api`, `avito`, `fast_local`).

> **About Avito A-vibe.** The weights are closed and there is no public GGUF.
> It is therefore represented in the catalog as an **OpenAI-compatible
> endpoint** (profile `configs/llms/avito.env`): if you have API/weights access,
> set `LLM_BASE_URL` and `LLM_MODEL` — no code changes needed.

**Cloud GigaChat (Sber).**

```bash
LLM_PROVIDER=gigachat
GIGACHAT_CREDENTIALS=<base64(client_id:client_secret)>
GIGACHAT_SCOPE=GIGACHAT_API_PERS
GIGACHAT_VERIFY_SSL=1     # requires the Ministry of Digital Development root cert; 0 — for tests
```

---

## 7. Corpus and index

**Sample input data.** The document corpus is **private and not published** in
the repository: put your own documents into `data/raw/sources/` (PDF/RTF/ODT;
scans are OCR'd). The repository ships the sample *questions* and HTTP requests:

- questions: [`data/qa/golden_qa.jsonl`](data/qa/golden_qa.jsonl) — 12
  `question → ground truth` pairs (in Russian, grounded in the sample standards);
- HTTP examples: [`examples/requests.http`](examples/requests.http),
  [`examples/README.md`](examples/README.md).

```bash
# conversion (RTF/ODT→PDF), OCR of scans, cleanup, manifest
uv run python scripts/prepare_docs.py           # --force to repeat

# build the index
uv run python scripts/build_index.py --reset --collection normdoc_e5_small
#   --mode chars --chunk-size 600 --chunk-overlap 60   # character chunking
#   --from-cache                                       # from data/chunks/chunks.json
```

Embeddings are switched by the `configs/embeddings/*.env` profile (`e5_small`,
`bge_m3`, `giga_embeddings`, `user_bge_m3`, `roberta`).

---

## 8. HTTP API and example requests

Service: `src/normdoc/api.py` (FastAPI). Interactive docs:
`http://localhost:8000/docs`.

**Web chat (browser):** open `http://localhost:8000/` — a built-in chat UI
(ask questions, browse the answer history, see sources, latency and dropped
citations; no external CDN, works offline). The same page falls back to a plain
HTML form (`GET /ask`) when JavaScript is disabled, so console browsers also work:

```bash
w3m  "http://localhost:8000/"                       # console browser (UI)
curl -s "http://localhost:8000/ask?q=What+is+an+architecture+description?"
lynx -dump "http://localhost:8000/ask?q=…"           # text answer in terminal
```

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | web chat UI (browser; `/ui` is an alias) |
| `GET` | `/ask` | answer as an HTML page for `?q=…` (no-JS / console browsers) |
| `GET` | `/info` | service info (JSON) |
| `GET` | `/health` | status, collection, model |
| `POST`| `/answer` | question → answer, sources, latency, usage |
| `GET` | `/metrics` | Prometheus metrics (latency/throughput) |
| `POST`| `/warmup` | warm-up (load models into memory) |

```bash
curl -s localhost:8000/answer -H 'Content-Type: application/json' -d '{
  "question": "Which order approved GOST R 57100—2025?",
  "top_k": 5
}' | jq
```

Response:

```json
{
  "answer": "The standard was approved by Rosstandart order No. 215-st and takes effect on 30 June 2025 [ГОСТ Р 57100—2025, стр. 27].",
  "sources": [{"source": "ГОСТ Р 57100—2025", "page": 27, "score": 0.83, "snippet": "…"}],
  "latency_s": 14.2, "retrieval_s": 0.4, "generation_s": 13.8,
  "usage": {"prompt_tokens": 1200, "completion_tokens": 64, "total_tokens": 1264},
  "dropped_citations": [], "collection": "normdoc_e5_small", "model": "meno"
}
```

CLI chat: `uv run python scripts/chat.py` (commands `:sources`, `:quit`).

---

## 9. Langfuse monitoring

The self-hosted Langfuse comes up with the `monitoring` profile (Postgres,
ClickHouse, Redis, MinIO, web, worker). The organization/project/keys are
created automatically (`LANGFUSE_INIT_*` in compose); keys are taken from `.env`.

```bash
docker compose --profile monitoring up -d
# UI: http://localhost:3000  (admin@example.com / password123 — from .env)

# enable tracing in the app
# .env: LANGFUSE_ENABLED=1; LANGFUSE_HOST=http://langfuse-web:3000 (in compose)
#       LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY — as in the compose project
docker compose --profile cpu up -d app

# export the latest traces into the report
uv run python scripts/langfuse_report.py --limit 50
```

Each request = one `rag_answer` trace with child spans `retrieve`
(type `retriever`) and `generate` (type `generation`, with tokens). The report
(`results/langfuse_traces.json` → `results/REPORT.md`) includes the trace id,
question, latency and per-span breakdown.

**Fallback without a server.** If the Langfuse server is unavailable (e.g. on a
weak dev machine), enable writing traces to JSONL — the data is the same
(question, spans, latency, tokens):

```bash
TRACE_JSONL=results/traces_local.jsonl uv run python scripts/chat.py -q "…"
uv run python scripts/langfuse_report.py        # will pick up the local JSONL
```

---

## 10. Quality evaluation (RAGAS) and quality gate

Evaluation runs on the **golden set** `data/qa/golden_qa.jsonl`; the judge is a
local LLM (`--judge-url`, by default a second llama-server on `:8081`).

```bash
# single run (all 12 questions)
uv run python scripts/evaluate_ragas.py --limit 12

# with the quality gate (non-zero exit code if thresholds are not met)
uv run python scripts/evaluate_ragas.py --gate
# thresholds in .env: THRESH_FAITHFULNESS / THRESH_CONTEXT_PRECISION / THRESH_CONTEXT_RECALL

# compare configurations (chunking × embeddings × reranker × LLM)
uv run python scripts/compare_configs.py --limit 6 --with-perf
```

Metrics: **Faithfulness**, **ContextPrecision**, **ContextRecall**.
Results: `results/comparison_ragas.json` (+ `results/ragas_*.json`). The `--gate`
mode is convenient in CI as an objective acceptance criterion.

---

## 11. Load testing and graphs

```bash
# start the API (docker or local), then:
uv run python scripts/load_test.py --url http://localhost:8000 \
    --requests 40 --concurrency 4 --name local
```

The script measures **RPS**, latency percentiles (p50/p90/p95/p99), the error
rate and tokens/s, and **plots graphs** (latency per request, distribution,
percentiles) → `results/load_local.json` and `results/load_local.png`. The
graphs are embedded in the final report.

Perf tests (retrieval/generation/tokens-s across configurations):

```bash
uv run python scripts/benchmark_rag.py --limit 6 --repeat 2 --name meno
```

---

## 12. Tests

```bash
uv run python scripts/run_tests.py            # → results/pytest.txt
uv run python scripts/run_tests.py -k TestApi -v
```

Coverage (`tests/`): parsing/OCR/tables, chunking, citations and grounding, LLM
payload (`reasoning_effort`), LLM providers (`openai`/`gigachat`, token cache),
golden QA and statistics, reporting, Langfuse monitoring (no-op and offline),
HTTP API (FastAPI `TestClient`).

---

## 13. CI/CD

[`.github/workflows/ci.yml`](.github/workflows/ci.yml):

- **tests** — lint (ruff, non-blocking) + pytest, artifact `results/pytest.txt`;
- **docker** — matrix build of the `app` and `llama-cpu` images, published to
  GHCR with tags (`branch`, `sha`, `latest`);
- **ragas-load** — on a **self-hosted** runner (GPU/CPU): brings up the stack,
  builds the index, runs RAGAS with `--gate`, a load test and the report;
  uploads artifacts. Triggered manually (`workflow_dispatch`).

---

## 14. Report

```bash
uv run python scripts/langfuse_report.py      # traces (if monitoring is on)
uv run python scripts/report.py               # → results/REPORT.md
```

`results/REPORT.md` combines: the test bench, corpus, RAGAS across
configurations, the perf table, load tests with graphs, Langfuse traces and
latency, tests, the RU model catalog and the Docker/serving section.

---

## 15. Project structure

```
├── docker-compose.yml          # qdrant + llama (profiles) + app + langfuse + tools
├── Makefile                    # short run commands
├── docker/
│   ├── app/Dockerfile          # RAG application image (FastAPI)
│   ├── llama/{Dockerfile,entrypoint.sh}   # llama.cpp: Vulkan/CUDA/ROCm/CPU (from source)
│   └── llama/Dockerfile.prism              # llama.cpp: prebuilt PrismML Vulkan (7840HS)
├── configs/
│   ├── hw/                     # hardware profiles (ai395, amd_7840hs, rtx3060)
│   ├── llms/                   # LLM profiles (bonsai, meno, gigachat, avito, …)
│   └── embeddings/             # embedding profiles (e5_small, bge_m3, …)
├── src/normdoc/                # code: assistant, llm, api, webui, indexing, parsing,
│                               #   prompts, monitoring, reporting, qa, model_catalog
├── scripts/                    # prepare_docs, build_index, chat, serve_api,
│                               #   evaluate_ragas, compare_configs, benchmark_rag, compare_plots,
│                               #   load_test, langfuse_report, report, run_tests, list_models
├── data/
│   ├── raw/sources/            # your documents (private, not published)
│   └── qa/golden_qa.jsonl      # sample test questions/ground truths
├── examples/                   # example HTTP requests
├── tests/                      # pytest: core + llmops
└── results/                    # artifacts: ragas, perf, load, langfuse, REPORT.md
```

---

## 16. Configuration

All settings go through `.env` (template: [`.env.example`](.env.example));
secrets never land in git. The `configs/{hw,llms,embeddings}` profiles are
selected by `HW_PROFILE` / `LLM_PROFILE` / `EMBEDDING_PROFILE`.

Key groups: Qdrant, corpus (`SOURCES_DIR`), LLM provider
(`LLM_PROVIDER/LLM_BASE_URL/…`, GigaChat), embeddings, reranker, chunking,
parser, Langfuse, Docker (`MODELS_DIR`, `MODEL_FILE`, `APP_PORT`), RAGAS
thresholds.

---

## 17. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `GGUF not found` | `MODELS_DIR` is empty; set `MODEL_FILE=/models/<file>.gguf` |
| `app` cannot see the LLM | the llama profile is not up (`ai395`/`amd7840hs`/`rtx3060`/`cpu`); check `docker compose ps` |
| No GPU inside the container | for Vulkan/ROCm — `group_add: video,render` and `/dev/dri`,`/dev/kfd`; for CUDA — nvidia-container-toolkit |
| `llama-server не найден в PRISM_DIR` | `PRISM_VULKAN_DIR` does not contain `llama-prism-*/llama-server`; extract `prism-vulkan.tar.gz` and point the variable at it |
| Prebuilt Vulkan crashes on start | host/container Mesa mismatch: rebuild `normdoc-llama:prism-vulkan` (`docker compose --profile amd7840hs build --no-cache llama-7840hs`) — the image pins Ubuntu 24.04 + Mesa to match the host |
| Langfuse is empty | `LANGFUSE_ENABLED=1`, `LANGFUSE_HOST=http://langfuse-web:3000`, keys match `LANGFUSE_INIT_*` |
| Low RAGAS recall | increase `TOP_K`/`RETRIEVE_CANDIDATES`, enable the reranker, review chunking |
| Slow generation | offload more layers to the GPU (`N_GPU_LAYERS`), reduce `N_CTX`, raise `N_PARALLEL` |

---

## 18. Limitations and assumptions

- **Dev hardware** — AMD Ryzen 7 7840HS + Radeon 780M iGPU. The `amd7840hs`
  profile serves the LLM on the iGPU through **Vulkan** (prebuilt PrismML
  binaries, no source build); the `ai395` (Vulkan from source) and `rtx3060`
  (CUDA) profiles ship as configurations and are built on the target hardware.
  The actual measurements in the report were taken on the available hardware
  (the numbers in `results/` were CPU-only; re-run on the iGPU profile to update
  them).
- **Avito A-vibe** — closed weights → represented as an OpenAI-compatible
  endpoint profile.
- **Smallest open GigaChat weights** — MoE 10B/1.8B and 20B/3B (not strictly
  5–8B); the catalog reflects this.
- **Langfuse on the dev machine.** The full self-hosted Langfuse stack (Postgres
  + ClickHouse + Redis + MinIO + web + worker) is described in
  `docker-compose.yml` (the `monitoring` profile) and targets the production
  boxes (AI MAX+ 395, 128 GB). It was not started on the weak dev machine
  (insufficient RAM); the traces for the report were collected through the
  built-in JSONL sink (`TRACE_JSONL`), whose format matches
  `scripts/langfuse_report.py`.
- **Document corpus** — the corpus is the user's own and is not published in the
  repository; bring your own documents in `data/raw/sources/`.
- Scans are recognized with OCR (tesseract); quality depends on the source.
