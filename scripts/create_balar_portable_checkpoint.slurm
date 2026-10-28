#!/bin/bash
#SBATCH --account=bsc18
#SBATCH --qos=ngp_debug
#SBATCH --job-name=balar-cpt-portable
#SBATCH --output=balar-cpt-portable-%j.out
#SBATCH --error=balar-cpt-portable-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00

set -u

WORK=/gpfs/scratch/bsc18/$USER/gem5-balar
INFRA=/home/bsc/$USER/gem5-infra

GEM5="$WORK/gem5/build/BSC/gem5.opt"
SCRIPT="$INFRA/gem5/simulate.py"

CONF="$INFRA/rvbench_gemm/rvbench_gemm-conf_ruby-1numa-tc2-dse6-cpus_8-threads_1-vlen_4096-elen_64-pagesize_4kb-size_512/rvbench_gemm-conf_ruby-1numa-tc2-dse6-cpus_8-threads_1-vlen_4096-elen_64-pagesize_4kb-size_512.yaml"

KERNEL="$INFRA/images/objects/RISCV/binaries/riscv-vmlinux-6.6.14.numa"
ROOTDISK="$INFRA/images/objects/RISCV/disks/gem5-ubuntu24.04-riscv64.img"

# IMPORTANT: new benchmark disk
BENCHDISK="$WORK/generated/benchdisk/benchmarks-balar-portable.img"

BOOTLOADER="$INFRA/images/objects/RISCV/binaries/riscv-opensbi-bootloader"
BOOTSCRIPT="$WORK/checkpoint_boot.sh"

CPTDIR="$WORK/checkpoints/linux-balar-base-portable"
OUTDIR="$WORK/runs/checkpoint-balar-portable"

rm -rf "$CPTDIR" "$OUTDIR"
mkdir -p "$CPTDIR" "$OUTDIR"

module purge 2>/dev/null || true
module load miniforge/24.11.3-2

source "$WORK/env_balar.sh"

export PYTHONPATH="/usr/lib64/python3.9/site-packages:${PYTHONPATH:-}"
unset PYTHONHOME

PYLIB=$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')

export GPGPUSIM_ROOT="$WORK/gpgpu-sim"
export GPGPUSIM_LIBDIR="$GPGPUSIM_ROOT/lib/gcc-11.5.0/cuda-11080/release"
export LD_LIBRARY_PATH="$PYLIB:$GPGPUSIM_LIBDIR:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

cd "$WORK/sst-elements/src/sst/elements/balar/tests"

timeout 1000s "$GEM5" \
    --outdir="$OUTDIR" \
    "$SCRIPT" \
    --config-file "$CONF" \
    --kernel "$KERNEL" \
    --disk "$ROOTDISK" \
    --disk "$BENCHDISK" \
    --root-partition /dev/vda1 \
    --bootloader "$BOOTLOADER" \
    --bootscript "$BOOTSCRIPT" \
    --checkpoint-mode \
    --checkpoint-directory "$CPTDIR" \
    --checkpoint-only \
    --riscv

RC=$?

echo
echo "CHECKPOINT_RC=$RC"

exit "$RC"
