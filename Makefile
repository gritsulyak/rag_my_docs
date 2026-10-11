# normdoc-assistant — удобные команды запуска и оценки.
#
# Основные сценарии:
#   make up-ai395      # поднять стек на AMD Ryzen AI MAX+ 395 (llama.cpp Vulkan)
#   make up-rtx3060    # поднять стек на NVIDIA RTX 3060 (llama.cpp CUDA)
#   make up-cpu        # поднять стек без GPU
#   make monitoring    # поднять self-hosted Langfuse (трейсы + latency)
#   make index         # подготовить корпус и собрать векторный индекс
#   make test          # unit-тесты (pytest) -> results/pytest.txt
#   make ragas         # оценка качества RAGAS -> results/comparison_ragas.json
#   make load          # нагрузочный тест с графиками -> results/load_*.png
#   make report        # собрать итоговый отчёт results/REPORT.md
#
# Переменные: MODELS_DIR (каталог с GGUF), MODEL_FILE, MODEL_ALIAS, QDRANT_COLLECTION.

SHELL := /bin/bash
COMPOSE := docker compose
MODELS_DIR ?= ./models
QDRANT_COLLECTION ?= normdoc_e5_small
export MODELS_DIR
export QDRANT_COLLECTION

.PHONY: help
help:
	@grep -E '^[a-zA-Z0-9_-]+:' Makefile | sed 's/:.*//' | sort -u | sed 's/^/  make /'

# ---------------------------------------------------------------- инфраструктура
.PHONY: up-ai395 up-rtx3060 up-rocm up-cpu monitoring down logs ps build
up-ai395:
	$(COMPOSE) --profile ai395 --profile monitoring up -d --build

up-rtx3060:
	$(COMPOSE) --profile rtx3060 --profile monitoring up -d --build

up-rocm:
	$(COMPOSE) --profile rocm --profile monitoring up -d --build

up-cpu:
	$(COMPOSE) --profile cpu up -d --build

monitoring:
	$(COMPOSE) --profile monitoring up -d

down:
	$(COMPOSE) --profile ai395 --profile rtx3060 --profile rocm --profile cpu --profile monitoring --profile tools down

logs:
	$(COMPOSE) logs -f --tail=100

ps:
	$(COMPOSE) ps

build:
	$(COMPOSE) --profile ai395 --profile rtx3060 --profile cpu build

# ------------------------------------------------------------------- пайплайн
.PHONY: prepare index
prepare:
	$(COMPOSE) --profile tools run --rm app python scripts/prepare_docs.py

index:
	$(COMPOSE) --profile tools run --rm app python scripts/prepare_docs.py
	$(COMPOSE) --profile tools run --rm app python scripts/build_index.py --reset \
		--collection $(QDRANT_COLLECTION)

# -------------------------------------------------------------------- качество
.PHONY: test ragas ragas-gate compare load report
test:
	$(COMPOSE) --profile tools run --rm app python scripts/run_tests.py

ragas:
	$(COMPOSE) --profile tools run --rm app python scripts/evaluate_ragas.py --limit 12

ragas-gate:
	$(COMPOSE) --profile tools run --rm app python scripts/evaluate_ragas.py --gate

compare:
	$(COMPOSE) --profile tools run --rm app python scripts/compare_configs.py --limit 6

load:
	$(COMPOSE) --profile tools run --rm app python scripts/load_test.py \
		--url http://app:8000 --requests 40 --concurrency 4 --name docker

report:
	$(COMPOSE) --profile tools run --rm app python scripts/report.py

# --------------------------------------------------------------------- локально
.PHONY: local-test local-report
local-test:
	uv run python scripts/run_tests.py

local-report:
	uv run python scripts/report.py
