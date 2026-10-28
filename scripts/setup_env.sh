#!/usr/bin/env bash

# This file is intended to be sourced.
#
#     source scripts/setup_env.sh
#

_BALAR_SETUP_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" &&
    pwd
)"

export BALAR_INTEGRATION_ROOT="$(
    cd "${_BALAR_SETUP_DIR}/.." &&
    pwd
)"

BALAR_LOCAL_ENV="${BALAR_LOCAL_ENV:-${BALAR_INTEGRATION_ROOT}/configs/local.env}"

if [[ ! -f "$BALAR_LOCAL_ENV" ]]; then
    echo "ERROR: local configuration not found:"
    echo "  $BALAR_LOCAL_ENV"
    echo
    echo "Create it from:"
    echo "  ${BALAR_INTEGRATION_ROOT}/configs/local.env.example"
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "$BALAR_LOCAL_ENV"

required_vars=(
    BALAR_WORK
    GEM5_INFRA_DIR
    GEM5_DIR
    GPGPUSIM_ROOT
    SST_ELEMENTS_DIR
)

for var in "${required_vars[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: $var is not defined in $BALAR_LOCAL_ENV"
        return 1 2>/dev/null || exit 1
    fi
done

# ------------------------------------------------------------
# Optional environment-modules setup.
# ------------------------------------------------------------

if [[ -n "${PYTHON_MODULE:-}" ]]; then
    if type module >/dev/null 2>&1; then
        module load "$PYTHON_MODULE"
    else
        echo "WARNING: PYTHON_MODULE=$PYTHON_MODULE but 'module' is unavailable"
    fi
fi

# ------------------------------------------------------------
# Find GPGPU-Sim libcudart when not explicitly configured.
# ------------------------------------------------------------

if [[ -z "${GPGPUSIM_LIBDIR:-}" ]]; then

    mapfile -t _gpgpu_libs < <(
        find -L "${GPGPUSIM_ROOT}/lib" \
            -type f \
            -name libcudart.so \
            2>/dev/null
    )

    if [[ ${#_gpgpu_libs[@]} -eq 0 ]]; then
        echo "ERROR: no libcudart.so found below:"
        echo "  ${GPGPUSIM_ROOT}/lib"
        echo
        echo "Build GPGPU-Sim or set GPGPUSIM_LIBDIR explicitly."
        return 1 2>/dev/null || exit 1
    fi

    if [[ ${#_gpgpu_libs[@]} -gt 1 ]]; then
        echo "ERROR: multiple libcudart.so files found:"
        printf '  %s\n' "${_gpgpu_libs[@]}"
        echo
        echo "Set GPGPUSIM_LIBDIR explicitly in configs/local.env."
        return 1 2>/dev/null || exit 1
    fi

    export GPGPUSIM_LIBDIR
    GPGPUSIM_LIBDIR="$(dirname "${_gpgpu_libs[0]}")"
fi

export GPGPUSIM_ROOT
export GPGPUSIM_LIBDIR

# ------------------------------------------------------------
# gem5-infra workspace directories
# ------------------------------------------------------------
#
# gem5-infra still expects these environment variables when
# constructing its default paths.
#
: "${SRCDIR:=${GEM5_INFRA_DIR}/sources}"
: "${IMGDIR:=${GEM5_INFRA_DIR}/images}"
: "${RUNDIR:=${GEM5_INFRA_DIR}/runs}"

export SRCDIR
export IMGDIR
export RUNDIR

# ------------------------------------------------------------
# CUDA tools used by GPGPU-Sim.
# ------------------------------------------------------------

if [[ -n "${CUDA_INSTALL_PATH:-}" ]]; then
    export CUDA_INSTALL_PATH
    export PTXAS_CUDA_INSTALL_PATH="$CUDA_INSTALL_PATH"
    export PATH="$CUDA_INSTALL_PATH/bin:$PATH"
fi

# ------------------------------------------------------------
# Python / dynamic libraries.
# ------------------------------------------------------------

unset PYTHONHOME

if command -v python3 >/dev/null 2>&1; then
    PYLIB="$(python3 -c \
        'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"
else
    PYLIB=""
fi

if [[ -n "$PYLIB" ]]; then
    export LD_LIBRARY_PATH="${PYLIB}:${GPGPUSIM_LIBDIR}:${LD_LIBRARY_PATH:-}"
else
    export LD_LIBRARY_PATH="${GPGPUSIM_LIBDIR}:${LD_LIBRARY_PATH:-}"
fi

if [[ -n "${PYDEPS_DIR:-}" && -d "${PYDEPS_DIR}" ]]; then
    export PYTHONPATH="${PYDEPS_DIR}:${GEM5_INFRA_DIR}/gem5:${PYTHONPATH:-}"
else
    export PYTHONPATH="${GEM5_INFRA_DIR}/gem5:${PYTHONPATH:-}"
fi

# ------------------------------------------------------------
# Python used by gem5-infra to parse simulation results.
# ------------------------------------------------------------

if [[ -z "${GEM5INFRA_PARSER_PYTHON:-}" ]]; then
    if [[ -x "${BALAR_WORK}/env-gem5infra/bin/python" ]]; then
        export GEM5INFRA_PARSER_PYTHON="${BALAR_WORK}/env-gem5infra/bin/python"
    else
        export GEM5INFRA_PARSER_PYTHON="$(command -v python3)"
    fi
else
    export GEM5INFRA_PARSER_PYTHON
fi

export PYTHONUNBUFFERED=1

# This path is embedded by the benchmark into generated srun.sh.
export BALAR_HOST_SETUP_SCRIPT="${BALAR_INTEGRATION_ROOT}/scripts/setup_env.sh"

export BALAR_TESTS="${SST_ELEMENTS_DIR}/src/sst/elements/balar/tests"

# Directory from which GPGPU-Sim loads gpgpusim.config.
export BALAR_GPGPUSIM_WORKDIR="${BALAR_GPGPUSIM_WORKDIR:-${BALAR_TESTS}}"

echo "===== Gem5Balar environment ====="
echo "BALAR_WORK=$BALAR_WORK"
echo "GEM5_DIR=$GEM5_DIR"
echo "GPGPUSIM_ROOT=$GPGPUSIM_ROOT"
echo "GPGPUSIM_LIBDIR=$GPGPUSIM_LIBDIR"
echo "SST_ELEMENTS_DIR=$SST_ELEMENTS_DIR"
echo "GEM5_INFRA_DIR=$GEM5_INFRA_DIR"
echo "SRCDIR=$SRCDIR"
echo "IMGDIR=$IMGDIR"
echo "RUNDIR=$RUNDIR"
