#!/bin/bash
#SBATCH --account=bsc18
#SBATCH --qos=ngp_debug
#SBATCH --job-name=m37-build
#SBATCH --output=m37-build-%j.out
#SBATCH --error=m37-build-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

set -eu

WORK=/gpfs/scratch/bsc18/$USER/gem5-balar
GEM5="$WORK/gem5"

echo "================================================="
echo " M3.7 GEM5BALAR BUILD"
echo "================================================="
echo "HOST=$(hostname)"
echo "DATE=$(date)"
echo "WORK=$WORK"
echo

module purge 2>/dev/null || true
module load miniforge/24.11.3-2
source activate sconsv4.0

PYLIB=$(python3 -c \
'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')

export GPGPUSIM_ROOT="$WORK/gpgpu-sim"
export GPGPUSIM_LIBDIR="$GPGPUSIM_ROOT/lib/gcc-11.5.0/cuda-11080/release"

export LD_LIBRARY_PATH="$PYLIB:$GPGPUSIM_LIBDIR:${LD_LIBRARY_PATH:-}"

cd "$GEM5"

SCONS=$(which scons)

echo "===== BUILD ENVIRONMENT ====="
echo "python=$(which python3)"
python3 --version
echo "scons=$SCONS"
echo "gcc=$(which gcc)"
echo "g++=$(which g++)"
echo "PYLIB=$PYLIB"
echo "GPGPUSIM_LIBDIR=$GPGPUSIM_LIBDIR"
echo

echo "================================================="
echo " STEP 1/2 - COMPILE GEM5BALAR OBJECT ONLY"
echo "================================================="

/usr/bin/env python3 "$SCONS" \
    --no-compress-debug \
    --ignore-style \
    -j2 \
    build/BSC/dev/balar/gem5_balar.o

echo
echo "===== GEM5BALAR OBJECT OK ====="
ls -lh build/BSC/dev/balar/gem5_balar.o

echo
echo "================================================="
echo " STEP 2/2 - INCREMENTAL GEM5.OPT"
echo "================================================="

/usr/bin/env python3 "$SCONS" \
    --no-compress-debug \
    --ignore-style \
    -j2 \
    build/BSC/gem5.opt

echo
echo "================================================="
echo " BUILD COMPLETE"
echo "================================================="

ls -lh build/BSC/gem5.opt

echo
echo "===== NORMAL SYMBOL TABLE ====="

nm -C build/BSC/gem5.opt | grep -E \
'is_SST_buffer_full|send_(read|write)_request_SST' || true

echo
echo "===== DYNAMIC SYMBOL TABLE ====="

nm -D -C build/BSC/gem5.opt | grep -E \
'is_SST_buffer_full|send_(read|write)_request_SST' || true

echo
echo "DONE $(date)"
