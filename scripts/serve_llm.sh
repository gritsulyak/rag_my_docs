#!/usr/bin/env bash
# Запуск локальной LLM (GGUF) через llama-server под профиль configs/llms/<name>.env.
#
# Скрипт читает из профиля порт (LLM_BASE_URL), алиас (LLM_MODEL) и контекст
# (LLM_CONTEXT_WINDOW), а путь к GGUF берёт из --model, $LLM_GGUF или карты
# популярных РФ-моделей ниже.
#
# Примеры:
#   scripts/serve_llm.sh gigachat
#   scripts/serve_llm.sh t_lite --model ~/models/t-lite/T-lite-it-2.1-Q4_K_M.gguf
#   LLM_GGUF=/data/qvikhr.gguf scripts/serve_llm.sh qvikhr --port 8090
#
# llama-server поднимается в фоне (setsid nohup), лог — в /tmp/opencode.
set -euo pipefail

PROFILE="${1:-}"
if [[ -z "$PROFILE" ]]; then
  echo "usage: $0 <profile> [--model path.gguf] [--port N] [--ctx N] [--threads N] [--ngl N] [--alias NAME]" >&2
  exit 2
fi
shift || true

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$ROOT/configs/llms/$PROFILE.env"
LLAMA_DIR="${LLAMA_DIR:-$HOME/github/llm_eng/prism-vulkan/llama-prism-b10770-6684606}"
LLAMA="$LLAMA_DIR/llama-server"

_port_from_url() {
  local url="$1"
  echo "$url" | sed -nE 's#.*:([0-9]{2,5})(/.*)?$#\1#p'
}

PORT=""
CTX=""
ALIAS="$PROFILE"
if [[ -f "$ENV_FILE" ]]; then
  PORT="$(_port_from_url "$(grep -E '^LLM_BASE_URL=' "$ENV_FILE" | head -1 | cut -d= -f2- || true)")"
  CTX="$(grep -E '^LLM_CONTEXT_WINDOW=' "$ENV_FILE" | head -1 | cut -d= -f2- || true)"
  ALIAS="$(grep -E '^LLM_MODEL=' "$ENV_FILE" | head -1 | cut -d= -f2- || echo "$PROFILE")"
fi
PORT="${PORT:-8082}"
CTX="${CTX:-32768}"

default_gguf() {
  case "$1" in
    gigachat) echo "$HOME/models/gigachat/GigaChat3.1-10B-A1.8B-Q4_K_M.gguf" ;;
    t_lite)   echo "$HOME/models/t-lite/T-lite-it-2.1-Q4_K_M.gguf" ;;
    qvikhr)   echo "$HOME/models/qvikhr/QVikhr-3-8B-Instruction-Q4_K_M.gguf" ;;
    ruadapt)  echo "$HOME/models/ruadapt/RuadaptQwen3-8B-Hybrid-Q4_K_M.gguf" ;;
    saiga)    echo "$HOME/models/saiga/saiga_llama3_8b-Q4_K_M.gguf" ;;
    qwen)     echo "$HOME/models/qwen/Qwen2.5-7B-Instruct-Q4_K_M.gguf" ;;
    meno)     echo "$HOME/models/meno-lite/meno-lite-0.1-Q4_K_M.gguf" ;;
    *)        echo "" ;;
  esac
}

MODEL="${LLM_GGUF:-$(default_gguf "$PROFILE")}"
THREADS="${LLM_THREADS:-$(nproc 2>/dev/null || echo 6)}"
NGL="${LLM_NGL:-99}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)   MODEL="$2"; shift 2 ;;
    --port)    PORT="$2"; shift 2 ;;
    --ctx)     CTX="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --ngl)     NGL="$2"; shift 2 ;;
    --alias)   ALIAS="$2"; shift 2 ;;
    *) echo "неизвестный аргумент: $1" >&2; exit 2 ;;
  esac
done

LOG="${LLM_LOG:-/tmp/opencode/${PROFILE}-server.log}"

if [[ ! -x "$LLAMA" ]]; then
  echo "llama-server не найден: $LLAMA (задайте LLAMA_DIR)" >&2
  exit 1
fi
if [[ -z "$MODEL" || ! -f "$MODEL" ]]; then
  echo "GGUF не найден для профиля '$PROFILE': '$MODEL'." >&2
  echo "Укажите --model /path/to.gguf или LLM_GGUF=..., либо скачайте GGUF (см. scripts/list_models.py)." >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG")"

export LD_LIBRARY_PATH="$LLAMA_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
setsid nohup "$LLAMA" \
  -m "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  -c "$CTX" -ngl "$NGL" -t "$THREADS" \
  --alias "$ALIAS" --no-warmup \
  > "$LOG" 2>&1 < /dev/null &

echo "Профиль '$PROFILE': $MODEL"
echo "llama-server запускается на http://127.0.0.1:$PORT/v1 (alias=$ALIAS, ctx=$CTX, лог: $LOG)"
