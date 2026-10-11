#!/usr/bin/env bash
# Запуск Meno-Lite-0.1 (7B RAG) на втором инстансе llama.cpp (порт 8081).
# Не трогает bonsai на 8080. Использует бинарник PrismML (Vulkan/CPU).
set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:-$HOME/github/llm_eng/prism-vulkan/llama-prism-b10770-6684606}"
MODEL="${MENO_GGUF:-$HOME/models/meno-lite/meno-lite-0.1-Q4_K_M.gguf}"
PORT="${MENO_PORT:-8081}"
CTX="${MENO_CTX:-16384}"
THREADS="${MENO_THREADS:-6}"
LOG="${MENO_LOG:-/tmp/opencode/meno-server.log}"

if [[ ! -x "$LLAMA_DIR/llama-server" ]]; then
  echo "llama-server не найден: $LLAMA_DIR/llama-server" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "GGUF не найден: $MODEL" >&2
  exit 1
fi

export LD_LIBRARY_PATH="$LLAMA_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
setsid nohup "$LLAMA_DIR/llama-server" \
  -m "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  -c "$CTX" -ngl 99 -t "$THREADS" --no-warmup \
  > "$LOG" 2>&1 < /dev/null &

echo "Meno-Lite запускается на http://127.0.0.1:$PORT (лог: $LOG)"
