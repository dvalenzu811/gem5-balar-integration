#!/bin/bash
#SBATCH --account=bsc18
#SBATCH --qos=ngp_debug
#SBATCH --job-name=balar-stage-disk
#SBATCH --output=/gpfs/scratch/bsc18/bsc363766/gem5-balar/stage-benchdisk-%j.out
#SBATCH --error=/gpfs/scratch/bsc18/bsc363766/gem5-balar/stage-benchdisk-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00

set -euo pipefail

export BENCHDISK_BASE="/gpfs/scratch/bsc18/bsc363766/gem5-balar/benchmarks-balar.img"
export BENCHDISK_OUT="/gpfs/scratch/bsc18/bsc363766/gem5-balar/generated/benchdisk/benchmarks-balar-portable.img"

cd "/gpfs/scratch/bsc18/bsc363766/gem5-balar/gem5-balar-integration"

./scripts/stage_vecadd_benchdisk.sh
