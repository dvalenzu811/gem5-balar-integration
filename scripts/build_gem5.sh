#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" &&
    pwd
)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_env.sh"

: "${GEM5_BUILD_JOBS:=2}"

if [[ ! -f "${GPGPUSIM_LIBDIR}/libcudart.so" ]]; then
    echo "ERROR: modified GPGPU-Sim libcudart.so not found:"
    echo "  ${GPGPUSIM_LIBDIR}/libcudart.so"
    echo
    echo "Build GPGPU-Sim first:"
    echo "  ./scripts/build.sh gpgpu"
    exit 1
fi

# Known-good gem5/Balar build environment on BSC.
# This is the same toolchain used by the validated M3.7 build:
#   Python 3.12 / Miniforge
#   SCons 4.0.1
#   GCC/G++ 11.5
export PYTHONNOUSERSITE=1

export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"

if [[ ! -x "$CC" ]]; then
    echo "ERROR: C compiler is not executable: $CC"
    exit 1
fi

if [[ ! -x "$CXX" ]]; then
    echo "ERROR: C++ compiler is not executable: $CXX"
    exit 1
fi

# SCons FindTool() searches PATH for gcc/clang.
# Ensure the selected compiler directory is visible there.
export PATH="$(dirname "$CC"):$PATH"

SCONS_BIN="${SCONS_BIN:-$(command -v scons || true)}"

if [[ -z "$SCONS_BIN" ]]; then
    echo "ERROR: scons was not found in PATH."
    echo
    echo "Load/configure the build environment and retry."
    echo "You may also set SCONS_BIN explicitly in configs/local.env."
    exit 1
fi

cd "$GEM5_DIR"

echo "============================================================"
echo " Building gem5 + Gem5Balar"
echo "============================================================"
echo "GEM5_DIR=$GEM5_DIR"
echo "GPGPUSIM_LIBDIR=$GPGPUSIM_LIBDIR"
echo "SCONS_BIN=$SCONS_BIN"
echo "JOBS=$GEM5_BUILD_JOBS"

echo
echo "===== STEP 1/2 - Gem5Balar object ====="

"$SCONS_BIN" \
    --no-compress-debug \
    --ignore-style \
    -j"$GEM5_BUILD_JOBS" \
    build/BSC/dev/balar/gem5_balar.o

test -f build/BSC/dev/balar/gem5_balar.o

echo
echo "===== STEP 2/2 - gem5.opt ====="

"$SCONS_BIN" \
    --no-compress-debug \
    --ignore-style \
    -j"$GEM5_BUILD_JOBS" \
    build/BSC/gem5.opt

test -x build/BSC/gem5.opt

echo
echo "===== GEM5BALAR MEM_FETCH HOOKS ====="

nm -C build/BSC/gem5.opt \
    | grep -E 'is_SST_buffer_full|send_(read|write)_request_SST' \
    || true

echo
echo "===== RESULT ====="
ls -lh build/BSC/gem5.opt

echo
echo "gem5 build complete."
