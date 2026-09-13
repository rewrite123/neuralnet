#!/usr/bin/env bash
set -euo pipefail

host=${DEEPLEARN_HOST:-deeplearn}
remote_root=${DEEPLEARN_ROOT:-/home/ihormel/projects/neuralnet}
branch=${1:-grownn}

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [branch]" >&2
    exit 2
fi

git push origin "$branch"

ssh "$host" 'bash -s' -- "$remote_root" "$branch" <<'REMOTE'
set -euo pipefail
remote_root=$1
branch=$2
source "$HOME/.cargo/env"
export PATH="$HOME/.local/opt/cmake-3.31.8-linux-x86_64/bin:$PATH"
cd "$remote_root"
git fetch origin "$branch"
git checkout "$branch"
git pull --ff-only origin "$branch"
CARGO_BUILD_JOBS=1 cargo build --release --features gpu
git rev-parse HEAD
REMOTE