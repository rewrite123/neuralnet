#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

vocab=data/gpt2/vocab.json
merges=data/gpt2/merges.txt
if [[ ! -f "$vocab" || ! -f "$merges" ]]; then
    echo "GPT-2 tokenizer files are required at data/gpt2/." >&2
    exit 1
fi

run_id="cuda-math-$(date +%Y%m%d-%H%M%S)"
corpus="runs/$run_id.txt"
model="models/$run_id.gguf"
log="runs/$run_id.log"
gpu_profile="runs/$run_id.gpu.csv"
cpu_profile="runs/$run_id.cpu.txt"
mkdir -p runs models

for number in $(seq 0 99); do
    printf '%s plus %s equals %s.\n' "$number" "$number" "$((number + number))"
    printf '%s minus %s equals zero.\n' "$number" "$number"
    printf '%s times two equals %s.\n' "$number" "$((number * 2))"
done > "$corpus"

bash scripts/run_cuda.sh new --architecture transformer --vocab 50257 --hidden 128 --heads 4 --ff-hidden 256 --blocks 1 --max-sequence 16 --output "$model"

bash scripts/run_cuda.sh train-text \
    --model "$model" --output "$model" --text "$corpus" --vocab "$vocab" --merges "$merges" \
    --sequence 16 --epochs 1 --max-sequences 256 --batch-size 4 --log-every 64 --checkpoint-every 1 \
    --learning-rate 0.001 --validation-fraction 0.1 --cuda > "$log" 2>&1 &
pid=$!

nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits --loop-ms=200 > "$gpu_profile" 2>&1 &
gpu_sampler=$!
top -b -d 1 -n 120 -p "$pid" > "$cpu_profile" 2>&1 &
cpu_sampler=$!

set +e
wait "$pid"
status=$?
set -e
kill "$gpu_sampler" "$cpu_sampler" 2>/dev/null || true
wait "$gpu_sampler" "$cpu_sampler" 2>/dev/null || true

if [[ "$status" -ne 0 ]]; then
    cat "$log" >&2
    exit "$status"
fi
if ! awk -F, -v pid="$pid" '$1 ~ "^" pid " " { found=1 } END { exit !found }' "$gpu_profile"; then
    echo "CUDA test failed: training PID $pid never appeared in nvidia-smi." >&2
    echo "GPU profile: $gpu_profile" >&2
    exit 1
fi

echo "CUDA arithmetic test passed."
echo "Training log: $log"
echo "GPU profile: $gpu_profile"
echo "CPU profile: $cpu_profile"
grep -F 'FINAL:' "$log" | tail -n 1