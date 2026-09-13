#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <neuralnet arguments...>" >&2
    exit 2
fi

nvrtc_dir=$(find "$HOME/.local/lib" -path '*/nvidia/cuda_nvrtc/lib' -type d -print -quit)
if [[ -z "$nvrtc_dir" || ! -f "$nvrtc_dir/libnvrtc.so" ]]; then
    echo "Compatible user-local NVRTC runtime not found under $HOME/.local/lib." >&2
    exit 1
fi

export LD_LIBRARY_PATH="$nvrtc_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$(dirname "$0")/../target/release/neuralnet" "$@"