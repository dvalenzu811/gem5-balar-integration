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

# Optional machine-specific configuration.
if [[ -f "${ROOT}/configs/local.env" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT}/configs/local.env"
fi

: "${SLURM_ACCOUNT:=bsc18}"
: "${SLURM_QOS:=ngp_debug}"
: "${SLURM_MEM:=8G}"

LAST_RUN="${ROOT}/.last_vecadd_run"

if [[ ! -f "$LAST_RUN" ]]; then
    echo "ERROR: no generated vecadd run recorded."
    echo "Run first:"
    echo "  ./scripts/generate_vecadd.sh"
    exit 1
fi

VECOUT="$(cat "$LAST_RUN")"

if [[ ! -d "$VECOUT" ]]; then
    echo "ERROR: generated run directory no longer exists:"
    echo "  $VECOUT"
    exit 1
fi

SRUN="${VECOUT}/srun.sh"

if [[ ! -f "$SRUN" ]]; then
    echo "ERROR: generated Slurm script not found:"
    echo "  $SRUN"
    exit 1
fi

if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch not available."
    exit 1
fi

echo "============================================================"
echo " Submitting portable vecadd"
echo "============================================================"
echo
echo "run directory:"
echo "  $VECOUT"
echo
echo "Slurm script:"
echo "  $SRUN"
echo
echo "account:"
echo "  $SLURM_ACCOUNT"
echo
echo "qos:"
echo "  $SLURM_QOS"
echo
echo "memory:"
echo "  $SLURM_MEM"
echo

SBATCH_ARGS=(
    --parsable
    --account="$SLURM_ACCOUNT"
    --mem="$SLURM_MEM"
)

if [[ -n "$SLURM_QOS" ]]; then
    SBATCH_ARGS+=(--qos="$SLURM_QOS")
fi

JOB_ID="$(
    sbatch "${SBATCH_ARGS[@]}" "$SRUN"
)"

printf '%s\n' "$JOB_ID" > "${ROOT}/.last_vecadd_job"

echo "Submitted vecadd job:"
echo "  $JOB_ID"
echo
echo "Check with:"
echo "  squeue -j $JOB_ID"
