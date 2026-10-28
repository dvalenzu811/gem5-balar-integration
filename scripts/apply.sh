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

LOCAL_ENV="${BALAR_LOCAL_ENV:-${ROOT}/configs/local.env}"

if [[ ! -f "$LOCAL_ENV" ]]; then
    echo "ERROR: missing local configuration:"
    echo "  $LOCAL_ENV"
    echo
    echo "Create it from:"
    echo "  ${ROOT}/configs/local.env.example"
    exit 1
fi

# shellcheck disable=SC1090
source "$LOCAL_ENV"

required_vars=(
    GEM5_DIR
    GPGPUSIM_ROOT
    SST_ELEMENTS_DIR
    GEM5_INFRA_DIR
)

for var in "${required_vars[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: $var is not defined"
        exit 1
    fi
done

require_git_repo()
{
    local repo="$1"
    local name="$2"

    if ! git -C "$repo" rev-parse --is-inside-work-tree \
        >/dev/null 2>&1; then
        echo "ERROR: $name is not a Git repository:"
        echo "  $repo"
        exit 1
    fi
}

expected_commit()
{
    local component="$1"

    awk -v section="[$component]" '
        $0 == section {
            inside = 1
            next
        }

        /^\[/ {
            inside = 0
        }

        inside && /^commit = / {
            sub(/^commit = /, "")
            print
            exit
        }
    ' "${ROOT}/evidence/versions.txt"
}

check_revision()
{
    local name="$1"
    local repo="$2"

    local expected
    local actual

    expected="$(expected_commit "$name")"
    actual="$(git -C "$repo" rev-parse HEAD)"

    if [[ -z "$expected" ]]; then
        echo "WARNING: no reference revision recorded for $name"
        return
    fi

    if [[ "$expected" != "$actual" ]]; then
        echo
        echo "ERROR: revision mismatch for $name"
        echo "  expected: $expected"
        echo "  actual:   $actual"
        echo

        if [[ "${BALAR_ALLOW_REVISION_MISMATCH:-0}" != "1" ]]; then
            echo "Set BALAR_ALLOW_REVISION_MISMATCH=1 to override."
            exit 1
        fi

        echo "WARNING: continuing despite revision mismatch."
    fi
}

apply_patch()
{
    local repo="$1"
    local patch="$2"

    echo
    echo "Applying:"
    echo "  $patch"

    if git -C "$repo" apply --check "$patch" 2>/dev/null; then
        git -C "$repo" apply "$patch"
        echo "  applied"
        return
    fi

    if git -C "$repo" apply --reverse --check "$patch" 2>/dev/null; then
        echo "  already applied"
        return
    fi

    echo
    echo "ERROR: patch cannot be applied cleanly:"
    echo "  $patch"
    echo
    git -C "$repo" apply --check "$patch" || true
    exit 1
}

install_file()
{
    local source="$1"
    local destination="$2"

    mkdir -p "$(dirname "$destination")"
    install -m 0644 "$source" "$destination"

    echo "Installed:"
    echo "  $destination"
}

echo "============================================================"
echo " gem5-Balar integration installer"
echo "============================================================"

require_git_repo "$GEM5_DIR"       "gem5"
require_git_repo "$GPGPUSIM_ROOT"  "gpgpu-sim"
require_git_repo "$SST_ELEMENTS_DIR" "sst-elements"
require_git_repo "$GEM5_INFRA_DIR" "gem5-infra"

echo
echo "Checking reference revisions..."

check_revision gem5         "$GEM5_DIR"
check_revision gpgpu-sim    "$GPGPUSIM_ROOT"
check_revision sst-elements "$SST_ELEMENTS_DIR"
check_revision gem5-infra   "$GEM5_INFRA_DIR"

# ============================================================
# gem5: Gem5Balar implementation
# ============================================================

echo
echo "===== gem5 / Gem5Balar ====="

for file in \
    Gem5Balar.py \
    gem5_balar.cc \
    gem5_balar.hh \
    balar_gpu_api.hh \
    SConscript
do
    install_file \
        "${ROOT}/src/gem5/dev/balar/${file}" \
        "${GEM5_DIR}/src/dev/balar/${file}"
done

apply_patch \
    "$GEM5_DIR" \
    "${ROOT}/patches/gem5/current-gem5-integration.patch"

# ============================================================
# GPGPU-Sim
# ============================================================

echo
echo "===== GPGPU-Sim ====="

apply_patch \
    "$GPGPUSIM_ROOT" \
    "${ROOT}/patches/gpgpu-sim/0001-balar-gpgpusim-integration.patch"

apply_patch \
    "$GPGPUSIM_ROOT" \
    "${ROOT}/patches/gpgpu-sim/0002-build-portability.patch"

# ============================================================
# gem5-infra
# ============================================================

echo
echo "===== gem5-infra ====="

apply_patch \
    "$GEM5_INFRA_DIR" \
    "${ROOT}/patches/gem5-infra/current-balar-benchmark-workflow.patch"

install_file \
    "${ROOT}/src/gem5-infra/exploration/benchmarks/vecadd.py" \
    "${GEM5_INFRA_DIR}/exploration/benchmarks/vecadd.py"

# ============================================================
# SST Elements guest runtime
# ============================================================

echo
echo "===== SST Elements / Linux runtime ====="

apply_patch \
    "$SST_ELEMENTS_DIR" \
    "${ROOT}/patches/sst-elements/0001-linux-runtime-build.patch"

install_file \
    "${ROOT}/src/sst-elements/balar/tests/libcudart/balar_linux.h" \
    "${SST_ELEMENTS_DIR}/src/sst/elements/balar/tests/libcudart/balar_linux.h"

install_file \
    "${ROOT}/src/sst-elements/balar/tests/libcudart/cuda_runtime_api_linux.cc" \
    "${SST_ELEMENTS_DIR}/src/sst/elements/balar/tests/libcudart/cuda_runtime_api_linux.cc"

install_file \
    "${ROOT}/src/sst-elements/balar/tests/vanadis_llvm_rv64/vecadd.cu" \
    "${SST_ELEMENTS_DIR}/src/sst/elements/balar/tests/vanadis_llvm_rv64/vecadd.cu"

# Canonical GPGPU-Sim configuration used by Gem5Balar.
install_file \
    "${ROOT}/configs/gpgpusim.config" \
    "${SST_ELEMENTS_DIR}/src/sst/elements/balar/tests/gpgpusim.config"

echo
echo "============================================================"
echo " Integration installed successfully"
echo "============================================================"

echo
echo "Reference-only patches were NOT applied:"
echo "  patches/reference/sst-balar-memfetch-tracing.patch"

echo
echo "Next steps:"
echo "  source scripts/setup_env.sh"
echo "  build GPGPU-Sim and gem5"
echo "  ./scripts/generate_vecadd.sh"
echo "  ./scripts/run_vecadd.sh"
