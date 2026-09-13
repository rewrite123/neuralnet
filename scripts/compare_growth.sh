#!/usr/bin/env bash
# Equal-budget comparison of growth strategies. Every run starts from the same seed model and
# sees the same number of tokens, so differences come from the growth policy rather than compute.
set -u
cd "$(dirname "$0")/.."
BIN=target/release/neuralnet
CORPUS=data/text/shakespeare.txt
EPOCHS=${EPOCHS:-2}
MAX_SEQUENCES=${MAX_SEQUENCES:-0}
SEQ=128
LR=0.0003
LOG_EVERY=${LOG_EVERY:-128}
mkdir -p runs
if [ "$MAX_SEQUENCES" -gt 0 ]; then LIMIT=(--max-sequences "$MAX_SEQUENCES"); else LIMIT=(); fi

for RUN in none atgt ang mixture; do
  OUT=models/grow-$RUN.gguf
  LOG=runs/$RUN.log
  cp models/seed.gguf "$OUT"
  rm -f "$OUT.optimizer.bin"
  if [ "$RUN" = none ]; then GROWTH=(); else GROWTH=(--growth "$RUN"); fi
  echo "### $RUN starting $(date +%H:%M:%S)" | tee "$LOG"
  "$BIN" train-text --model "$OUT" --output "$OUT" --text "$CORPUS" \
    --sequence $SEQ --epochs "$EPOCHS" --learning-rate $LR --log-every $LOG_EVERY \
    --validation-fraction 0.1 "${GROWTH[@]}" "${LIMIT[@]}" --cuda >>"$LOG" 2>&1
  echo "### $RUN done $(date +%H:%M:%S)" >>"$LOG"
  grep FINAL "$LOG" | tail -1
done

echo "=== summary ==="
for RUN in none atgt ang mixture; do printf '%-8s %s\n' "$RUN" "$(grep FINAL runs/$RUN.log | tail -1)"; done
