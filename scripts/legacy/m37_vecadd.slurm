#!/bin/bash
#SBATCH --account=bsc18
#SBATCH --qos=ngp_debug
#SBATCH --job-name=m37-probe-debug
#SBATCH --output=m37-probe-debug-%j.out
#SBATCH --error=m37-probe-debug-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00

set -u

WORK=/gpfs/scratch/bsc18/$USER/gem5-balar
INFRA=/home/bsc/$USER/gem5-infra

BIN="$WORK/gem5/build/BSC/gem5.opt"
SCRIPT="$INFRA/gem5/simulate.py"
OUT="$WORK/runs/m37-vecadd-real"

CONF="$INFRA/rvbench_gemm/rvbench_gemm-conf_ruby-1numa-tc2-dse6-cpus_8-threads_1-vlen_4096-elen_64-pagesize_4kb-size_512/rvbench_gemm-conf_ruby-1numa-tc2-dse6-cpus_8-threads_1-vlen_4096-elen_64-pagesize_4kb-size_512.yaml"

KERNEL="$INFRA/images/objects/RISCV/binaries/riscv-vmlinux-6.6.14.numa"
DISK1="$INFRA/images/objects/RISCV/disks/gem5-ubuntu24.04-riscv64.img"
DISK2="$INFRA/images/objects/RISCV/disks/benchmarks.img"
BOOTLOADER="$INFRA/images/objects/RISCV/binaries/riscv-opensbi-bootloader"

BOOT="$WORK/after_restore_vecadd_raw.sh"
CPT="$WORK/checkpoints/linux-balar-base"

rm -rf "$OUT"
mkdir -p "$OUT"

echo "========================================"
echo " M3.7 PYTHON / WIRING PROBE"
echo "========================================"

#
# IMPORTANT: gem5 was built with Python 3.12.
#
module purge 2>/dev/null || true
module load miniforge/24.11.3-2

#
# Load GPGPU-Sim/Balar environment.
#
source "$WORK/env_balar.sh"

export PYTHONPATH="/usr/lib64/python3.9/site-packages:${PYTHONPATH:-}"

#
# Do NOT allow system Python 3.9 modules to contaminate
# gem5's embedded Python 3.12 interpreter.
#
unset PYTHONHOME

PYLIB=$(python3 -c \
'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')

export GPGPUSIM_ROOT="$WORK/gpgpu-sim"
export GPGPUSIM_LIBDIR="$GPGPUSIM_ROOT/lib/gcc-11.5.0/cuda-11080/release"

export LD_LIBRARY_PATH="$PYLIB:$GPGPUSIM_LIBDIR:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

echo "===== HOST ENV ====="
echo "python=$(which python3)"
python3 --version
echo "PYLIB=$PYLIB"
echo "PYTHONHOME=${PYTHONHOME-<unset>}"
echo "PYTHONPATH=${PYTHONPATH-<unset>}"
echo

BALAR_TESTS="$WORK/sst-elements/src/sst/elements/balar/tests"

echo "===== GPGPU-SIM CONFIG ====="
echo "BALAR_TESTS=$BALAR_TESTS"

test -f "$BALAR_TESTS/gpgpusim.config" || {
    echo "ERROR: missing $BALAR_TESTS/gpgpusim.config"
    exit 1
}

grep -n -- '-gpgpu_n_clusters' "$BALAR_TESTS/gpgpusim.config"

cd "$BALAR_TESTS"

echo "PWD=$(pwd)"
echo "gpgpusim.config=$(readlink -f gpgpusim.config)"

timeout 1000s "$BIN" \
    --outdir="$OUT" \
    "$SCRIPT" \
    --config-file "$CONF" \
    --kernel "$KERNEL" \
    --disk "$DISK1" \
    --disk "$DISK2" \
    --root-partition /dev/vda1 \
    --bootloader "$BOOTLOADER" \
    --bootscript "$BOOT" \
    --checkpoint-restore 1 \
    --checkpoint-directory "$CPT" \
    --riscv

RC=$?

echo
echo "PROBE_RC=$RC"

exit 0
