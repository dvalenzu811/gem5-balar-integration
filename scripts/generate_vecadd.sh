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

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_env.sh"

: "${GEM5_BIN:=${GEM5_DIR}/build/BSC/gem5.opt}"
: "${GEM5_SCRIPT:=${GEM5_INFRA_DIR}/gem5/simulate.py}"

# ------------------------------------------------------------------
# Portable vecadd artifacts.
#
# These deliberately use VECADD_* variables instead of the historical
# HOSTBIN / BENCHDISK / CHECKPOINT_DIR variables that may still exist
# in older local environments.
# ------------------------------------------------------------------

: "${VECADD_HOST_DIR:=${ROOT}/build/vecadd/host}"

: "${VECADD_BENCHDISK:=${BALAR_WORK}/generated/benchdisk/benchmarks-balar-portable.img}"

: "${VECADD_CHECKPOINT_DIR:=${BALAR_WORK}/checkpoints/linux-balar-base-portable}"

: "${GENERATED_DIR:=${BALAR_WORK}/generated/vecadd-portable}"

required_files=(
    "$GEM5_BIN"
    "$GEM5_SCRIPT"
    "$BENCH_CONFIG"
    "$KERNEL"
    "$ROOTDISK"
    "$VECADD_BENCHDISK"
    "$VECADD_HOST_DIR/vecadd-ubuntu"
)

for f in "${required_files[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required file not found:"
        echo "  $f"
        exit 1
    fi
done

if [[ ! -d "$VECADD_CHECKPOINT_DIR" ]]; then
    echo "ERROR: checkpoint directory not found:"
    echo "  $VECADD_CHECKPOINT_DIR"
    exit 1
fi

echo "============================================================"
echo " Portable vecadd inputs"
echo "============================================================"
echo
echo "gem5:"
echo "  $GEM5_BIN"
echo
echo "host CUDA ELF:"
echo "  $VECADD_HOST_DIR/vecadd-ubuntu"
echo
echo "benchmark disk:"
echo "  $VECADD_BENCHDISK"
echo
echo "checkpoint:"
echo "  $VECADD_CHECKPOINT_DIR"
echo
echo "output:"
echo "  $GENERATED_DIR"

rm -rf "$GENERATED_DIR"
mkdir -p "$GENERATED_DIR"

cd "${GEM5_INFRA_DIR}/exploration"

python3 exploration.py \
    --generate \
    --benchmark vecadd \
    --riscv \
    --restore-checkpoint \
    --checkpoint-directory "$VECADD_CHECKPOINT_DIR" \
    --config-files "$BENCH_CONFIG" \
    --gem5-bin "$GEM5_BIN" \
    --gem5-script "$GEM5_SCRIPT" \
    --binary-dir "$VECADD_HOST_DIR" \
    --kernel "$KERNEL" \
    --rootdisk "$ROOTDISK" \
    --benchdisk "$VECADD_BENCHDISK" \
    --results-dir "$GENERATED_DIR"

VECOUT="$(
    find "$GENERATED_DIR" \
        -mindepth 1 \
        -maxdepth 1 \
        -type d \
        | head -n1
)"

if [[ -z "$VECOUT" ]]; then
    echo "ERROR: gem5-infra did not create a vecadd run directory"
    exit 1
fi

printf '%s\n' "$VECOUT" > "${ROOT}/.last_vecadd_run"

echo
echo "============================================================"
echo " Generated vecadd run"
echo "============================================================"
echo
echo "  $VECOUT"

echo
echo "===== GUEST WORKLOAD ====="

grep -nE \
    'vecadd|resetstats|dumpstats|m5 exit' \
    "$VECOUT/work.sh" || true

echo
echo "===== GENERATED GEM5 INPUTS ====="

grep -nE \
    'checkpoint-restore|checkpoint-directory|cuda-executable|benchmarks-balar|benchdisk' \
    "$VECOUT/srun.sh" || true

echo
echo "Run with:"
echo "  ./scripts/run_vecadd.sh"
