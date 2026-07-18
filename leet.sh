#!/usr/bin/env bash
# Local LeetGPU harness. Runs leetgpu_local.py with the machine's CUDA venv. Usage:
#   ./leet.sh check                       # verify the setup after cloning
#   ./leet.sh list [easy|medium|hard] [--search conv]
#   ./leet.sh setup challenges/easy/1_vector_add
#   ./leet.sh test 1_vector_add
#   ./leet.sh profile 1_vector_add        # generate standalone profile.cu + data, then build.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Pick the venv for this machine's arch (home is shared NFS across x86_64 + aarch64 nodes).
py=""
for cand in ".venv-cuda-$(uname -m)" ".venv-cuda"; do
    if [[ -x "$root/$cand/bin/python" ]]; then py="$root/$cand/bin/python"; break; fi
done
if [[ -z "$py" ]]; then
    echo "No CUDA venv for $(uname -m). Create one, e.g.:" >&2
    echo "  uv venv .venv-cuda-$(uname -m) --python 3.12 && \\" >&2
    echo "  uv pip install --python .venv-cuda-$(uname -m) torch numpy --index-url https://download.pytorch.org/whl/cu130" >&2
    exit 1
fi
exec "$py" "$root/scripts/leetgpu_local.py" "$@"
