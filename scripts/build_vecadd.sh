#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" &&
    pwd
)"

ROOT="$(
    cd "${SCRIPT_DIR}/.." &&
    pwd
)"

# Integration repo normally lives beside sst-elements.
DEFAULT_WORK="$(dirname "$ROOT")"

: "${SST_ELEMENTS_DIR:=${DEFAULT_WORK}/sst-elements}"

: "${GUEST_RISCV_GXX:=$(command -v riscv64-linux-gnu-g++-11 || true)}"
: "${GUEST_RISCV_GCC:=$(command -v riscv64-linux-gnu-gcc-11 || true)}"
: "${GUEST_RISCV_AR:=$(command -v riscv64-linux-gnu-ar || true)}"

: "${GUEST_CLANGXX:=${HOME}/balar/llvm-install/bin/clang++}"
: "${GUEST_CUDA_TOOLKIT:=${HOME}/balar/cuda-toolkit}"

: "${GUEST_RISCV_SYSROOT:=/usr/riscv64-linux-gnu}"
: "${GUEST_RISCV_CXX_HEADERS:=${GUEST_RISCV_SYSROOT}/include/c++/11}"

: "${GPU_ARCH:=sm_70}"

BUILD_ROOT="${ROOT}/build/vecadd"
HOST_BUILD="${BUILD_ROOT}/host"
GUEST_BUILD="${BUILD_ROOT}/guest"

TESTS="${SST_ELEMENTS_DIR}/src/sst/elements/balar/tests"
LIBCUDART="${TESTS}/libcudart"
VECADD_DIR="${TESTS}/vanadis_llvm_rv64"

VECADD_SRC="${VECADD_DIR}/vecadd.cu"
VANADIS_RUNTIME="${LIBCUDART}/cuda_runtime_api_vanadis.cc"
LINUX_RUNTIME="${LIBCUDART}/cuda_runtime_api_linux.cc"

CUOBJDUMP="${GUEST_CUDA_TOOLKIT}/bin/cuobjdump"

required_files=(
    "$GUEST_RISCV_GXX"
    "$GUEST_RISCV_GCC"
    "$GUEST_RISCV_AR"
    "$GUEST_CLANGXX"
    "$CUOBJDUMP"
    "$GUEST_CUDA_TOOLKIT/include/cuda_runtime_api.h"
    "$GUEST_RISCV_CXX_HEADERS/cmath"
    "$GUEST_RISCV_CXX_HEADERS/riscv64-linux-gnu/bits/c++config.h"
    "$VECADD_SRC"
    "$VANADIS_RUNTIME"
    "$LINUX_RUNTIME"
)

for f in "${required_files[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required file not found:"
        echo "  $f"
        exit 1
    fi
done

mkdir -p "$HOST_BUILD" "$GUEST_BUILD"

GCC_DIR="$(
    dirname "$("$GUEST_RISCV_GCC" -print-libgcc-file-name)"
)"

compile_runtime()
{
    local source="$1"
    local object="$2"
    local archive="$3"

    "$GUEST_RISCV_GXX" \
        -march=rv64gc \
        -mabi=lp64d \
        -fPIC \
        -idirafter "${GUEST_CUDA_TOOLKIT}/include" \
        -I"$LIBCUDART" \
        '-DBALAR_CUDA_VERSION="libcudart_vanadis.a.11"' \
        -c "$source" \
        -o "$object"

    rm -f "$archive"

    "$GUEST_RISCV_AR" \
        rcs \
        "$archive" \
        "$object"
}

link_vecadd()
{
    local library_dir="$1"
    local library_name="$2"
    local output="$3"
    local include_ptx="${4:-no}"

    local cuda_ptx_args=()

    if [[ "$include_ptx" == "yes" ]]; then
        cuda_ptx_args=(
            "--cuda-include-ptx=${GPU_ARCH}"
        )
    fi

    "$GUEST_CLANGXX" \
        "$VECADD_SRC" \
        -o "$output" \
        --target=riscv64-linux-gnu \
        --gcc-install-dir="$GCC_DIR" \
        --sysroot="$GUEST_RISCV_SYSROOT" \
        -march=rv64gc \
        -mabi=lp64d \
        --cuda-path="$GUEST_CUDA_TOOLKIT" \
        --cuda-gpu-arch="$GPU_ARCH" \
        "${cuda_ptx_args[@]}" \
        -include __clang_cuda_runtime_wrapper.h \
        -nostdinc++ \
        -isystem "$GUEST_RISCV_CXX_HEADERS" \
        -isystem "$GUEST_RISCV_CXX_HEADERS/riscv64-linux-gnu" \
        -isystem "$GUEST_RISCV_CXX_HEADERS/backward" \
        -isystem "$GUEST_RISCV_SYSROOT/include" \
        -static \
        -static-libgcc \
        -static-libstdc++ \
        -L"$library_dir" \
        "-l${library_name}" \
        -ldl \
        -lrt \
        -pthread
}

echo "============================================================"
echo " Building host-visible CUDA ELF"
echo "============================================================"

compile_runtime \
    "$VANADIS_RUNTIME" \
    "$HOST_BUILD/cuda_runtime_api_vanadis.o" \
    "$HOST_BUILD/libcudart_vanadis.a"

link_vecadd \
    "$HOST_BUILD" \
    cudart_vanadis \
    "$HOST_BUILD/vecadd-ubuntu" \
    yes

echo
echo "===== VALIDATING HOST EMBEDDED PTX ====="

HOST_PTX_CHECK="${HOST_BUILD}/embedded.ptx.check"

"$CUOBJDUMP" \
    --dump-ptx \
    "$HOST_BUILD/vecadd-ubuntu" \
    > "$HOST_PTX_CHECK"

if ! grep -q '_Z6vecAddPdS_S_i' "$HOST_PTX_CHECK"; then
    echo "ERROR: host CUDA ELF does not contain vecAdd embedded PTX."
    exit 1
fi

if ! grep -q "\.target ${GPU_ARCH}" "$HOST_PTX_CHECK"; then
    echo "ERROR: host CUDA ELF does not contain PTX for ${GPU_ARCH}."
    exit 1
fi

echo "Host embedded PTX validation: OK"

grep -E \
    '\.version|\.target|\.entry _Z6vecAddPdS_S_i' \
    "$HOST_PTX_CHECK" \
    | head -10

rm -f "$HOST_PTX_CHECK"

echo
echo "============================================================"
echo " Building Linux guest CUDA ELF"
echo "============================================================"

compile_runtime \
    "$LINUX_RUNTIME" \
    "$GUEST_BUILD/cuda_runtime_api_linux.o" \
    "$GUEST_BUILD/libcudart_balar_linux.a"

# IMPORTANT:
# Do not add --cuda-include-ptx here.
# This recipe reproduces the validated Linux guest ELF byte-for-byte.
link_vecadd \
    "$GUEST_BUILD" \
    cudart_balar_linux \
    "$GUEST_BUILD/vecadd-ubuntu" \
    no

echo
echo "===== HOST ELF ====="

file "$HOST_BUILD/vecadd-ubuntu"

strings "$HOST_BUILD/vecadd-ubuntu" \
    | grep -E \
      'libcudart_vanadis\.a|/proc/self/pagemap|/dev/mem' \
    | sort -u \
    || true

echo
echo "===== HOST PTX LIST ====="

"$CUOBJDUMP" \
    --list-ptx \
    "$HOST_BUILD/vecadd-ubuntu"

echo
echo "===== GUEST ELF ====="

file "$GUEST_BUILD/vecadd-ubuntu"

strings "$GUEST_BUILD/vecadd-ubuntu" \
    | grep -E \
      'libcudart_vanadis\.a|/proc/self/pagemap|/dev/mem' \
    | sort -u \
    || true

echo
echo "===== SHA256 ====="

sha256sum \
    "$HOST_BUILD/vecadd-ubuntu" \
    "$GUEST_BUILD/vecadd-ubuntu"

echo
echo "Build outputs:"
echo "  host:  $HOST_BUILD/vecadd-ubuntu"
echo "  guest: $GUEST_BUILD/vecadd-ubuntu"
