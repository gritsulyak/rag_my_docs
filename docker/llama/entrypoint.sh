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

echo "[llama] model=$MODEL_FILE alias=$ALIAS ctx=$N_CTX ngl=$N_GL parallel=$N_PARALLEL port=$PORT"
exec llama-server \
    -m "$MODEL_FILE" \
    --host 0.0.0.0 --port "$PORT" \
    -c "$N_CTX" -ngl "$N_GL" -t "$THREADS" -np "$N_PARALLEL" \
    --alias "$ALIAS" \
    ${LLAMA_EXTRA_ARGS:-}
