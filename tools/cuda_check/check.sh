#!/usr/bin/env bash
# TEST-ONLY: type-check .cu files with clang where no CUDA toolkit is installed (the sandbox). Host and device sides,
# sm_89, warnings shown. Usage: tools/cuda_check/check.sh FILE.cu ...   (run from the repository root)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
rc=0
for f in "$@"; do
  for side in --cuda-host-only --cuda-device-only; do
    clang++ -x cuda -std=c++20 -fsyntax-only -nocudainc -nocudalib --cuda-gpu-arch=sm_89 $side \
      -I "$here" -I "$root/src" -I "$root/include" -I "$root/build/generated" \
      -Wall -Wextra -Wno-unknown-cuda-version -Wno-unused-parameter "$f" || rc=1
  done
done
exit $rc
