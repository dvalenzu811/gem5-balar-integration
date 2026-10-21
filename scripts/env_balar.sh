module purge 2>/dev/null || true
module load miniforge/24.11.3-2

export WORK=/gpfs/scratch/bsc18/$USER/gem5-balar
export PYLIB=/gpfs/apps/NGPP/MINIFORGE/24.11.3-2/lib

export GPGPUSIM_ROOT="$WORK/gpgpu-sim"
export GPGPUSIM_LIBDIR="$GPGPUSIM_ROOT/lib/gcc-11.5.0/cuda-11080/release"

export LD_LIBRARY_PATH="$PYLIB:$GPGPUSIM_LIBDIR:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WORK/pydeps:$HOME/gem5-infra/gem5:${PYTHONPATH:-}"

# CUDA tools required by GPGPU-Sim for PTX extraction
export CUDA_INSTALL_PATH=/apps/NVIDIA-HPC-SDK/24.11/Linux_aarch64/24.11/cuda/11.8
export PTXAS_CUDA_INSTALL_PATH="$CUDA_INSTALL_PATH"
export PATH="$CUDA_INSTALL_PATH/bin:$PATH"
