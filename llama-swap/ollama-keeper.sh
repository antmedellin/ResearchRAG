#!/bin/sh
# Usage: ollama-keeper.sh <ollama-model-tag> <chat|embed|systemone>
MODEL="$1"
KIND="${2:-chat}"
OLLAMA="${OLLAMA_URL:-http://ollama:11434}"

post() {  # post <endpoint> <json>
  curl -fsS -X POST "$OLLAMA$1" -H 'Content-Type: application/json' -d "$2" >/dev/null 2>&1
}

is_loaded() {
  curl -fsS "$OLLAMA/api/ps" 2>/dev/null | grep -q "\"name\":\"$MODEL\""
}

load() {
  case "$KIND" in
    embed)
      post /api/embed "{\"model\":\"$MODEL\",\"input\":\"warmup\",\"keep_alive\":-1}" ;;
    systemone)
      post /v1/systemone "{\"model\":\"$MODEL\",\"keep_alive\":-1,\"state\":\"warmup\",\"questions\":{\"w\":{\"type\":\"noul\",\"instructions\":\"Is this text?\",\"criteria\":{\"true\":\"yes\",\"false\":\"no\"}}}}" ;;
    *)
      post /api/generate "{\"model\":\"$MODEL\",\"keep_alive\":-1}" ;;
  esac
}

unload() {
  echo "[keeper] unloading $MODEL"
  # try several unload paths; stop as soon as the model is gone
  for attempt in \
    "/api/generate|{\"model\":\"$MODEL\",\"keep_alive\":0}" \
    "/api/embed|{\"model\":\"$MODEL\",\"input\":\"\",\"keep_alive\":0}" \
    "/v1/systemone|{\"model\":\"$MODEL\",\"keep_alive\":0,\"state\":\"x\",\"questions\":{\"w\":{\"type\":\"noul\",\"instructions\":\"x\",\"criteria\":{\"true\":\"y\",\"false\":\"n\"}}}}"
  do
    post "${attempt%%|*}" "${attempt#*|}"
    sleep 1
    is_loaded || { echo "[keeper] unloaded via ${attempt%%|*}"; exit 0; }
  done
  echo "[keeper] WARNING: $MODEL still loaded after unload attempts"
  exit 0
}
trap unload TERM INT

echo "[keeper] loading $MODEL ($KIND)"
load || echo "[keeper] preload failed (will load on first request)"

# Re-load models that Ollama may have evicted while loading this one.
# REPIN is a space-separated list of embedding model tags.
for m in $REPIN; do
  if ! curl -fsS "$OLLAMA/api/ps" | grep -q "\"name\":\"$m\""; then
    echo "[keeper] re-pinning $m"
    post /api/embed "{\"model\":\"$m\",\"input\":\"warmup\",\"keep_alive\":-1}"
  fi
done

while :; do sleep 1 & wait $!; done