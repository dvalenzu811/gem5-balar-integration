#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" &&
    pwd
)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_env.sh"

: "${GPGPU_BUILD_JOBS:=16}"
: "${GPU_ARCH:=sm_70}"

export GPU_ARCH
export CUDA_INSTALL_PATH
export PTXAS_CUDA_INSTALL_PATH="${PTXAS_CUDA_INSTALL_PATH:-$CUDA_INSTALL_PATH}"

cd "$GPGPUSIM_ROOT"

echo "============================================================"
echo " Building modified GPGPU-Sim"
echo "============================================================"

echo "GPGPUSIM_ROOT=$GPGPUSIM_ROOT"
echo "CUDA_INSTALL_PATH=$CUDA_INSTALL_PATH"
echo "GPU_ARCH=$GPU_ARCH"
echo "JOBS=$GPGPU_BUILD_JOBS"

# setup_environment must run after CUDA_INSTALL_PATH is defined.
# The SST argument is the mode used by the Balar integration.
# shellcheck disable=SC1091
source ./setup_environment sst

make -j"$GPGPU_BUILD_JOBS"

echo
echo "===== BUILT LIBCUDART ====="

mapfile -t libs < <(
    find "$GPGPUSIM_ROOT/lib" \
        -type f \
        -name libcudart.so \
        2>/dev/null
)

if [[ ${#libs[@]} -eq 0 ]]; then
    echo "ERROR: GPGPU-Sim build completed but libcudart.so was not found."
    exit 1
fi

printf '%s\n' "${libs[@]}"

echo
echo "GPGPU-Sim build complete."
