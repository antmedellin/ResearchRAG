#!/bin/sh
set -eu

PLATFORM="${DECISION_PLATFORM:-vllm}"
MODEL="${DECISION_MODEL:-clef-flash:9b}"

case "$PLATFORM" in
  ollama)
    echo "[decision] platform=ollama model=$MODEL"

    exec sh /app/ollama-keeper.sh \
      "$MODEL" \
      systemone
    ;;

  vllm)
    echo "[decision] platform=vllm model=$MODEL"

    exec vllm-wrapper serve \
      --vllm-url "${VLLM_DECISION_URL:-http://vllm-decision:8000}" \
      --listen ":${PORT}" \
      --wait-timeout 600s \
      -- /bin/sh -c "echo Decision vLLM daemon is externally managed; exit 1"
    ;;

  *)
    echo "ERROR: DECISION_PLATFORM must be 'ollama' or 'vllm'" >&2
    exit 1
    ;;
esac
