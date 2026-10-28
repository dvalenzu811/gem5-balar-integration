#!/usr/bin/env bash
#SBATCH --account=bsc18
#SBATCH --qos=ngp_debug
#SBATCH --job-name=gem5-port-build
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --mem=16G

set -euo pipefail

# Exact environment used by the previously successful gem5 build.
module purge
module load miniforge/24.11.3-2

export PYTHONNOUSERSITE=1
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export SCONS_BIN=/apps/MINIFORGE/24.11.3-2/bin/scons

REPO=/gpfs/scratch/bsc18/$USER/gem5-balar/gem5-balar-integration
cd "$REPO"

source scripts/setup_env.sh

PYLIB="$(
    python3 -c \
    'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'
)"

export LD_LIBRARY_PATH="$PYLIB:$GPGPUSIM_LIBDIR:${LD_LIBRARY_PATH:-}"

echo "===== NODE ====="
hostname
uname -m

echo
echo "===== BUILD ENV ====="
echo "python=$(command -v python3)"
echo "scons=$SCONS_BIN"
echo "gcc=$CC"
echo "g++=$CXX"
echo "PYLIB=$PYLIB"

echo
echo "===== START GEM5 BUILD ====="

./scripts/build.sh gem5

echo
echo "===== BUILD RESULT ====="

ls -lh \
    "$GEM5_DIR/build/BSC/dev/balar/gem5_balar.o" \
    "$GEM5_DIR/build/BSC/gem5.opt"

echo
echo "GEM5 PORT BUILD: PASS"
