#!/usr/bin/env bash
# Запуск llama-server внутри контейнера.
#
# Переменные:
#   MODEL_FILE      — путь к GGUF (например, /models/gigachat.gguf). Если не задан,
#                     берётся единственный *.gguf в MODEL_DIR.
#   MODEL_DIR       — каталог с моделями (по умолчанию /models).
#   MODEL_ALIAS     — имя модели в API (по умолчанию basename файла).
#   N_GPU_LAYERS    — число слоёв на GPU (-ngl): 0 = CPU, 99 = максимум.
#   N_CTX           — размер контекста.
#   N_PARALLEL      — число слотов параллельных запросов.
#   THREADS         — число CPU-потоков.
#   LLAMA_EXTRA_ARGS— дополнительные аргументы llama-server.
#   PRISM_DIR       — каталог с ПРЕДСОБРАННЫМИ бинарниками llama.cpp (PrismML,
#                     Vulkan). Если задан, llama-server ищется здесь (вместо
#                     бинарника из PATH), а каталог добавляется в LD_LIBRARY_PATH.
set -euo pipefail

PORT="${PORT:-8080}"
MODEL_DIR="${MODEL_DIR:-/models}"
MODEL_FILE="${MODEL_FILE:-}"

if [[ -z "$MODEL_FILE" ]]; then
    MODEL_FILE="$(find "$MODEL_DIR" -maxdepth 2 -name '*.gguf' -type f | head -n 1 || true)"
fi
if [[ -z "$MODEL_FILE" || ! -f "$MODEL_FILE" ]]; then
    echo "GGUF не найден. Смонтируйте каталог моделей в $MODEL_DIR" >&2
    echo "или задайте MODEL_FILE=/models/<файл>.gguf (см. README → «Модели»)." >&2
    ls -la "$MODEL_DIR" 2>/dev/null || true
    exit 1
fi

ALIAS="${MODEL_ALIAS:-$(basename "$MODEL_FILE" .gguf)}"
N_GL="${N_GPU_LAYERS:-99}"
N_CTX="${N_CTX:-32768}"
N_PARALLEL="${N_PARALLEL:-4}"
THREADS="${THREADS:-$(nproc)}"

# Источник бинарника: source-сборка (llama-server из PATH) или предсобранный
# PrismML-релиз в PRISM_DIR (монтируется в контейнер, см. Dockerfile.prism).
if [[ -n "${PRISM_DIR:-}" ]]; then
    LLAMA_SERVER="$(find "$PRISM_DIR" -maxdepth 2 -name llama-server -type f | head -n 1 || true)"
    if [[ -z "$LLAMA_SERVER" ]]; then
        echo "llama-server не найден в PRISM_DIR=$PRISM_DIR" >&2
        ls -la "$PRISM_DIR" 2>/dev/null || true
        exit 1
    fi
    LLAMA_BIN_DIR="$(dirname "$LLAMA_SERVER")"
    export LD_LIBRARY_PATH="$LLAMA_BIN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "[llama] prebuilt prism binary: $LLAMA_SERVER"
else
    LLAMA_SERVER="llama-server"
fi

echo "[llama] model=$MODEL_FILE alias=$ALIAS ctx=$N_CTX ngl=$N_GL parallel=$N_PARALLEL port=$PORT"
exec "$LLAMA_SERVER" \
    -m "$MODEL_FILE" \
    --host 0.0.0.0 --port "$PORT" \
    -c "$N_CTX" -ngl "$N_GL" -t "$THREADS" -np "$N_PARALLEL" \
    --alias "$ALIAS" \
    ${LLAMA_EXTRA_ARGS:-}
