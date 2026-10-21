#ifndef __DEV_BALAR_BALAR_GPU_API_HH__
#define __DEV_BALAR_BALAR_GPU_API_HH__

#include <cstddef>
#include <cstdint>
#include <tuple>

struct dim3
{
    unsigned int x;
    unsigned int y;
    unsigned int z;
};

struct CUstream_st;
using cudaStream_t = CUstream_st *;

enum cudaError : int;
using cudaError_t = cudaError;

enum cudaMemcpyKind : int
{
    cudaMemcpyHostToHost   = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

extern "C"
{

uint64_t cudaMallocSST(void **devPtr, size_t size);

cudaError_t cudaMemcpy(
    void *dst,
    const void *src,
    size_t count,
    cudaMemcpyKind kind
);

void SST_gpgpusim_numcores_equal_check(unsigned numCores);
bool SST_gpgpusim_launch_blocking();
bool SST_gpu_core_cycle();

unsigned __cudaRegisterFatBinarySST(
    char fileName[256]
);

void __cudaRegisterFunctionSST(
    uint64_t fatCubinHandle,
    uint64_t hostFun,
    char deviceFun[512]
);

cudaError_t cudaConfigureCall(
    dim3 gridDim,
    dim3 blockDim,
    size_t sharedMem,
    cudaStream_t stream
);

cudaError_t cudaSetupArgumentSST(
    uint64_t arg,
    uint8_t value[256],
    size_t size,
    size_t offset
);

cudaError_t cudaLaunchSST(
    uint64_t hostFun
);

void SST_receive_mem_reply(
    unsigned core_id,
    void *mem_req
);

}

/*
 * GPGPU-Sim stream_manager.cc declares these callbacks with
 * C++ linkage, not extern "C". Keep exactly the same ABI so
 * these strong definitions override the weak GPGPU-Sim ones.
 */
void SST_callback_memcpy_H2D_done(
    uint64_t dst,
    uint64_t src,
    size_t count,
    cudaStream_t stream
);

void SST_callback_memcpy_D2H_done(
    uint64_t dst,
    uint64_t src,
    size_t count,
    cudaStream_t stream
);

std::tuple<cudaError_t, size_t, unsigned>
SST_cudaGetParamConfig(
    uint64_t hostFun,
    unsigned index
);

#endif
