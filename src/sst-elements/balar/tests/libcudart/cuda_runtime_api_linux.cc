// Copyright 2009-2026 NTESS. Under the terms
// of Contract DE-NA0003525 with NTESS, the U.S.
// Government retains certain rights in this software.
//
// Copyright (c) 2009-2026, NTESS
// All rights reserved.
//
// Portions are copyright of other developers:
// See the file CONTRIBUTORS.TXT in the top level directory
// of the distribution for more information.
//
// This file is part of the SST software package. For license
// information, see the LICENSE file in the top level directory of the
// distribution.

#include "cuda_runtime_api.h"
#include "balar_linux.h"
#include "../../balar_packet.h"
#include "../../balar_consts.h"

using namespace SST::BalarComponent;

// Encode version information here
#ifndef BALAR_CUDA_VERSION
#define BALAR_CUDA_VERSION "11.0"
#endif
static const char *version = BALAR_CUDA_VERSION;

#if !defined(__dv)
#if defined(__cplusplus)
#define __dv(v) = v
#else /* __cplusplus */
#define __dv(v)
#endif /* __cplusplus */
#endif /* !__dv */

extern "C" {
    #include <unistd.h>
    #include <sys/mman.h>
    #include <sys/types.h>
    #include <fcntl.h>
    #include <errno.h>
    #include <stdlib.h>

    inline __attribute__((always_inline)) void __vanadisFence() {
        asm volatile ("fence");
    }

    /*
     * BSC gem5 Linux Full-System mapping.
     *
     * MMIO:
     *      physical 0x10020000 -> Gem5Balar
     *
     * CUDA packet:
     *      normal Linux virtual memory
     *          -> /proc/self/pagemap
     *          -> physical RAM address
     *          -> Gem5Balar dmaRead()
     */

    inline __attribute__((always_inline))
    uint64_t __linuxVirtToPhysRaw(const void *ptr)
    {
        uintptr_t va = (uintptr_t)ptr;

        uint64_t entry = 0;

        off_t offset =
            (off_t)((va / g_page_size) * sizeof(uint64_t));

        ssize_t n = pread(
            g_pagemap_fd,
            &entry,
            sizeof(entry),
            offset
        );

        if (n != (ssize_t)sizeof(entry)) {
            perror("BALAR: pread pagemap");
            abort();
        }

        if ((entry & (1ULL << 63)) == 0) {
            fprintf(
                stderr,
                "BALAR: virtual page for %p is not present\n",
                ptr
            );
            abort();
        }

        uint64_t pfn =
            entry & ((1ULL << 55) - 1);

        if (pfn == 0) {
            fprintf(
                stderr,
                "BALAR: pagemap returned PFN=0 for %p\n",
                ptr
            );
            abort();
        }

        return
            pfn * g_page_size +
            (va & (g_page_size - 1));
    }


    inline __attribute__((always_inline))
    void __linuxInitBalar()
    {
        if (g_balarBaseAddr != NULL &&
            g_scratch_mem != NULL &&
            g_pagemap_fd >= 0) {
            return;
        }

        g_page_size =
            (size_t)sysconf(_SC_PAGESIZE);

        if (g_page_size == 0) {
            fprintf(stderr, "BALAR: invalid page size\n");
            abort();
        }

        /*
         * Map Gem5Balar MMIO.
         */
        g_devmem_fd =
            open("/dev/mem", O_RDWR | O_SYNC);

        if (g_devmem_fd < 0) {
            perror("BALAR: open(/dev/mem)");
            abort();
        }

        void *mmio = mmap(
            NULL,
            g_page_size,
            PROT_READ | PROT_WRITE,
            MAP_SHARED,
            g_devmem_fd,
            BALAR_MMIO_PHYS
        );

        if (mmio == MAP_FAILED) {
            perror("BALAR: mmap MMIO");
            abort();
        }

        g_balarBaseAddr =
            (volatile Addr_t *)mmio;

        /*
         * Open Linux virtual -> physical translation interface.
         */
        g_pagemap_fd =
            open("/proc/self/pagemap", O_RDONLY);

        if (g_pagemap_fd < 0) {
            perror("BALAR: open /proc/self/pagemap");
            abort();
        }

        /*
         * Keep the same virtual layout expected by the Balar
         * runtime, but back it with normal Linux RAM.
         *
         * Physical contiguity is NOT assumed.
         */
        void *shared = NULL;

        if (posix_memalign(
                &shared,
                g_page_size,
                BALAR_SHARED_SIZE) != 0) {
            fprintf(stderr, "BALAR: posix_memalign failed\n");
            abort();
        }

        g_shared_base =
            (uint8_t *)shared;

        g_scratch_mem =
            g_shared_base;

        g_staging_mem =
            g_shared_base + BALAR_STAGING_OFFSET;

        /*
         * Fault in the packet page so pagemap returns a real PFN.
         * Both BalarCudaCallPacket_t and the currently-used return
         * prefix fit in this page.
         */
        memset(
            g_scratch_mem,
            0,
            g_page_size
        );

        g_scratch_phys =
            __linuxVirtToPhysRaw(g_scratch_mem);

        printf(
            "BALAR Linux map: "
            "mmio=%p scratch=%p scratch_phys=0x%lx\n",
            (void *)g_balarBaseAddr,
            (void *)g_scratch_mem,
            (unsigned long)g_scratch_phys
        );
        fflush(stdout);
    }


    inline __attribute__((always_inline))
    BalarCudaCallPacket_t *__balarPacket()
    {
        __linuxInitBalar();

        return (BalarCudaCallPacket_t *)
            g_scratch_mem;
    }


    inline __attribute__((always_inline))
    uint64_t __sharedVirtToPhys(const void *ptr)
    {
        __linuxInitBalar();

        return __linuxVirtToPhysRaw(ptr);
    }


    inline __attribute__((always_inline))
    void *__sharedPhysToVirt(uint64_t phys)
    {
        __linuxInitBalar();

        /*
         * Return packets are DMA-written into the same physical
         * page used by the CUDA call packet.
         */
        uint64_t page_phys =
            g_scratch_phys &
            ~((uint64_t)g_page_size - 1);

        if (phys < page_phys ||
            phys >= page_phys + g_page_size) {

            fprintf(
                stderr,
                "BALAR: physical response 0x%lx "
                "is not in scratch page 0x%lx\n",
                (unsigned long)phys,
                (unsigned long)page_phys
            );
            abort();
        }

        return
            g_scratch_mem +
            (phys - page_phys);
    }


    typedef enum BalarCudaCallType {
        NORMAL_CALL = 1,
        BLOCKING_ISSUE = 1 << 1,
        BLOCKING_COMPLETE = 1 << 2,
    } BalarCudaCallType_t;
}

extern "C"
BalarCudaCallReturnPacket_t *
makeCudaCall(BalarCudaCallPacket_t *call_packet_ptr)
{
    __linuxInitBalar();

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf(
            "Making CUDA API Call ID: %s\n",
            CudaAPIEnumToString(
                call_packet_ptr->cuda_call_id
            )
        );
        fflush(stdout);
    }

    uint64_t packet_phys =
        __sharedVirtToPhys(call_packet_ptr);

    /*
     * Tell Balar where the CUDA packet is in the
     * shared PHYSICAL address space.
     */
    *g_balarBaseAddr = packet_phys;

    __vanadisFence();

    /*
     * MMIO read returns the PHYSICAL address at
     * which Balar/DMA wrote BalarCudaCallReturnPacket_t.
     */
    uint64_t response_phys = 0;

    /*
     * Gem5Balar currently completes the DMA asynchronously.
     * Poll until the MMIO read exposes the address of this
     * packet's completed return.
     *
     * This is temporary until Gem5Balar reproduces the original
     * delayed-MMIO-response semantics of SST Balar.
     */
    do {
        response_phys =
            *g_balarBaseAddr;

        __vanadisFence();

    } while (response_phys != packet_phys);

    return (BalarCudaCallReturnPacket_t *)
        __sharedPhysToVirt(response_phys);
}


extern "C"
BalarCudaCallReturnPacket_t *
readLastCudaStatus()
{
    __linuxInitBalar();

    uint64_t response_phys =
        *g_balarBaseAddr;

    __vanadisFence();

    return (BalarCudaCallReturnPacket_t *)
        __sharedPhysToVirt(response_phys);
}


extern "C" BalarCudaCallReturnPacket_t * makeCudaCallFlags(BalarCudaCallPacket_t * call_packet_ptr, int flags) {
    // Perform retry if the call is blocking
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCall(call_packet_ptr);

    // If the call is blocking on issue, we need to retry
    if (flags & BLOCKING_ISSUE) {
        while (response_packet_ptr->cuda_error == cudaErrorNotReady) {
            response_packet_ptr = makeCudaCall(call_packet_ptr);
        }
    }

    // For blocking complete, we need to wait for the call to finish
    // by polling for the is_cuda_call_done flag
    if (flags & BLOCKING_COMPLETE) {
        while (!(response_packet_ptr->is_cuda_call_done)) {
            response_packet_ptr = readLastCudaStatus();
        }
    }
    return response_packet_ptr;
}

cudaError_t cudaMalloc(void **devPtr, uint64_t size) {
    // Send request to GPU
    BalarCudaCallPacket_t *call_packet_ptr = __balarPacket();
    call_packet_ptr->isSSTmem = true;
    call_packet_ptr->cuda_call_id = CUDA_MALLOC;
    call_packet_ptr->cuda_malloc.size = size;
    call_packet_ptr->cuda_malloc.devPtr = devPtr;

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("Malloc Packet address: %p\n", call_packet_ptr);
        printf("Malloc Packet size: %lu\n", size);
        printf("Malloc Packet devptr: %p\n", devPtr);
        fflush(stdout);
    }

    // Make cuda call
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCall(call_packet_ptr);

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("CUDA API ID: %d with error: %d\nMalloc addr: %lx Dev addr: %lx\n",
                response_packet_ptr->cuda_call_id, response_packet_ptr->cuda_error,
                response_packet_ptr->cudamalloc.malloc_addr, response_packet_ptr->cudamalloc.devptr_addr);
        fflush(stdout);
    }

    *devPtr = (void *)response_packet_ptr->cudamalloc.malloc_addr;

    return response_packet_ptr->cuda_error;
}

cudaError_t cudaMemcpy(
    void *dst,
    const void *src,
    size_t count,
    enum cudaMemcpyKind kind)
{
    __linuxInitBalar();

    if (kind == cudaMemcpyHostToHost) {
        memcpy(dst, src, count);
        return cudaSuccess;
    }

    if (kind != cudaMemcpyHostToDevice &&
        kind != cudaMemcpyDeviceToHost) {

        fprintf(
            stderr,
            "BALAR: unsupported cudaMemcpy kind %d\n",
            (int)kind
        );

        return cudaErrorInvalidMemcpyDirection;
    }

    /*
     * A normal Linux allocation is virtually contiguous but is
     * not guaranteed to be physically contiguous.
     *
     * Therefore issue one Balar CUDA_MEMCPY call per physical
     * page fragment.
     */

    if (kind == cudaMemcpyDeviceToHost) {
        /*
         * Fault destination pages into RAM before consulting
         * /proc/self/pagemap.
         */
        memset(dst, 0, count);
    }

    size_t completed = 0;

    while (completed < count) {

        void *host_ptr = NULL;

        if (kind == cudaMemcpyHostToDevice) {
            host_ptr =
                (void *)((uintptr_t)src + completed);
        } else {
            host_ptr =
                (void *)((uintptr_t)dst + completed);
        }

        uint64_t host_phys =
            __sharedVirtToPhys(host_ptr);

        size_t page_offset =
            (size_t)(
                (uintptr_t)host_ptr &
                (g_page_size - 1)
            );

        size_t page_left =
            g_page_size - page_offset;

        size_t remaining =
            count - completed;

        size_t chunk =
            remaining < page_left ?
            remaining : page_left;

        BalarCudaCallPacket_t *call_packet_ptr =
            __balarPacket();

        memset(
            call_packet_ptr,
            0,
            sizeof(BalarCudaCallPacket_t)
        );

        call_packet_ptr->isSSTmem = true;
        call_packet_ptr->cuda_call_id =
            CUDA_MEMCPY;

        call_packet_ptr->cuda_memcpy.kind =
            kind;

        call_packet_ptr->cuda_memcpy.count =
            chunk;

        if (kind == cudaMemcpyHostToDevice) {

            call_packet_ptr->cuda_memcpy.dst =
                (uint64_t)dst + completed;

            call_packet_ptr->cuda_memcpy.src =
                host_phys;

            call_packet_ptr->cuda_memcpy.payload =
                (uint64_t)src + completed;

        } else {

            call_packet_ptr->cuda_memcpy.dst =
                host_phys;

            call_packet_ptr->cuda_memcpy.src =
                (uint64_t)src + completed;

            call_packet_ptr->cuda_memcpy.payload =
                (uint64_t)dst + completed;
        }

        BalarCudaCallReturnPacket_t *response_packet_ptr =
            makeCudaCallFlags(
                call_packet_ptr,
                BLOCKING_ISSUE |
                BLOCKING_COMPLETE
            );

        if (response_packet_ptr->cuda_error !=
            cudaSuccess) {

            return response_packet_ptr->cuda_error;
        }

        completed += chunk;
    }

    return cudaSuccess;
}


__host__ cudaError_t CUDARTAPI cudaMemcpyToSymbol(
    const void *symbol, const void *src, size_t count, size_t offset,
    enum cudaMemcpyKind kind) {
    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("MemcpyToSymbol dst: %s\n", symbol);
        fflush(stdout);
    }
    BalarCudaCallPacket_t *call_packet_ptr = __balarPacket();
    call_packet_ptr->isSSTmem = true;
    call_packet_ptr->cuda_call_id = CUDA_MEMCPY_TO_SYMBOL;

    // Pass the symbol name by value
    call_packet_ptr->cuda_memcpy_to_symbol.symbol = (uint64_t) symbol;
    call_packet_ptr->cuda_memcpy_to_symbol.src = (uint64_t) src;
    call_packet_ptr->cuda_memcpy_to_symbol.count = (uint64_t) count;
    call_packet_ptr->cuda_memcpy_to_symbol.offset = (uint64_t) offset;
    call_packet_ptr->cuda_memcpy_to_symbol.kind = kind;

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("MemcpyToSymbol Packet address: %p\n", call_packet_ptr);
        printf("MemcpyToSymbol kind: %d\n", (uint8_t) (call_packet_ptr->cuda_memcpy.kind));
        printf("MemcpyToSymbol size: %ld\n", count);
        printf("MemcpyToSymbol src: %p\n", src);
        printf("MemcpyToSymbol symbol: %s\n", symbol);
        fflush(stdout);
    }

    // Make cuda call
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCallFlags(call_packet_ptr, BLOCKING_ISSUE | BLOCKING_COMPLETE);

    // Make the memcpy sync with balar so that CPU will
    // issue another write/read to balar only if the previous memcpy is done
    while (!(response_packet_ptr->is_cuda_call_done)) {
        int wait = 0;
        response_packet_ptr = readLastCudaStatus();
        wait++;
    }

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("CUDA API ID: %d with error: %d\n",
                response_packet_ptr->cuda_call_id, response_packet_ptr->cuda_error);
        fflush(stdout);
    }

    return response_packet_ptr->cuda_error;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbol(
    void *dst, const void *symbol, size_t count, size_t offset,
    enum cudaMemcpyKind kind) {
    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("MemcpyFromSymbol symbol: %s\n", symbol);
        fflush(stdout);
    }
    BalarCudaCallPacket_t *call_packet_ptr = __balarPacket();
    call_packet_ptr->isSSTmem = true;
    call_packet_ptr->cuda_call_id = CUDA_MEMCPY_FROM_SYMBOL;

    // Pass the symbol name by value
    call_packet_ptr->cuda_memcpy_from_symbol.symbol = (uint64_t) symbol;
    call_packet_ptr->cuda_memcpy_from_symbol.dst = (uint64_t) dst;
    call_packet_ptr->cuda_memcpy_from_symbol.count = (uint64_t) count;
    call_packet_ptr->cuda_memcpy_from_symbol.offset = (uint64_t) offset;
    call_packet_ptr->cuda_memcpy_from_symbol.kind = kind;

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("MemcpyFromSymbol Packet address: %p\n", call_packet_ptr);
        printf("MemcpyFromSymbol kind: %d\n", (uint8_t) (call_packet_ptr->cuda_memcpy.kind));
        printf("MemcpyFromSymbol size: %ld\n", count);
        printf("MemcpyFromSymbol dst: %p\n", dst);
        printf("MemcpyFromSymbol symbol: %s\n", symbol);
        fflush(stdout);
    }

    // Make cuda call
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCallFlags(call_packet_ptr, BLOCKING_ISSUE | BLOCKING_COMPLETE);

    // Make the memcpy sync with balar so that CPU will
    // issue another write/read to balar only if the previous memcpy is done
    while (!(response_packet_ptr->is_cuda_call_done)) {
        int wait = 0;
        response_packet_ptr = readLastCudaStatus();
        wait++;
    }

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("CUDA API ID: %d with error: %d\n",
                response_packet_ptr->cuda_call_id, response_packet_ptr->cuda_error);
        fflush(stdout);
    }

    return response_packet_ptr->cuda_error;
}

extern "C" {
// Function declarations
cudaError_t cudaLaunch(uint64_t func);

// Cuda Configure call
cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, uint64_t sharedMem, cudaStream_t stream) {
    BalarCudaCallPacket_t *call_packet_ptr = __balarPacket();
    call_packet_ptr->isSSTmem = true;
    call_packet_ptr->cuda_call_id = CUDA_CONFIG_CALL;

    // Prepare packet
    call_packet_ptr->configure_call.gdx = gridDim.x;
    call_packet_ptr->configure_call.gdy = gridDim.y;
    call_packet_ptr->configure_call.gdz = gridDim.z;
    call_packet_ptr->configure_call.bdx = blockDim.x;
    call_packet_ptr->configure_call.bdy = blockDim.y;
    call_packet_ptr->configure_call.bdz = blockDim.z;
    call_packet_ptr->configure_call.sharedMem = sharedMem;
    call_packet_ptr->configure_call.stream = stream;

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("Start Configure Call:\n");
        printf("Grid Dim: x(%d) y(%d) z(%d)\n", gridDim.x, gridDim.y, gridDim.z);
        printf("Block Dim: x(%d) y(%d) z(%d)\n", blockDim.x, blockDim.y, blockDim.z);
        printf("Stream: %p\n", stream);
        fflush(stdout);
    }

    // Make cuda call
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCall(call_packet_ptr);

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("CUDA API ID: %d with error: %d\n",
                response_packet_ptr->cuda_call_id, response_packet_ptr->cuda_error);
        fflush(stdout);
    }

    return response_packet_ptr->cuda_error;
}

cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, void *stream) {
	return cudaSuccess;
}
unsigned CUDARTAPI __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                               size_t sharedMem,
                                               struct CUstream_st *stream = 0) {
	cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
    return 0;
}

// Cuda Setup argument
cudaError_t cudaSetupArgument(uint64_t arg, uint8_t value[BALAR_CUDA_MAX_ARG_SIZE], uint64_t size, uint64_t offset) {
    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("Start setup argument:\n");
        printf("Size: %d offset: %d\n", size, offset);
        fflush(stdout);
    }

    BalarCudaCallPacket_t *call_packet_ptr = __balarPacket();
    call_packet_ptr->isSSTmem = true;
    call_packet_ptr->cuda_call_id = CUDA_SET_ARG;
    call_packet_ptr->setup_argument.size = size;
    call_packet_ptr->setup_argument.offset = offset;


    if (value == NULL) {
        // The argument is a device pointer
        if (g_debug_level >= LOG_LEVEL_DEBUG) {
            printf("Argument is a device pointer\n");
        }
        call_packet_ptr->setup_argument.arg = arg;
    } else {
        // The argument is a constant value
        if (g_debug_level >= LOG_LEVEL_DEBUG) {
            printf("Argument is a constant\n");
        }
        call_packet_ptr->setup_argument.arg =  (uint64_t) NULL;
        for (int i = 0; i < 8; i++) {
            call_packet_ptr->setup_argument.value[i] = value[i];
        }
    }

    // Make cuda call
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCall(call_packet_ptr);

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("CUDA API ID: %d with error: %d\n",
                response_packet_ptr->cuda_call_id, response_packet_ptr->cuda_error);
        fflush(stdout);
    }

    return response_packet_ptr->cuda_error;
}

BalarCudaCallReturnPacket_t __balarGetParamInfo(uint64_t hostFunc, unsigned index) {
    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("Query param %d of function %d\n", index, hostFunc);
        fflush(stdout);
    }

    BalarCudaCallPacket_t *call_packet_ptr = __balarPacket();
    call_packet_ptr->isSSTmem = true;
    call_packet_ptr->cuda_call_id = CUDA_PARAM_CONFIG;
    call_packet_ptr->cudaparamconfig.hostFun = hostFunc;
    call_packet_ptr->cudaparamconfig.index = index;

    // Make cuda call
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCall(call_packet_ptr);

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("CUDA API ID: %d with error: %d\n",
                response_packet_ptr->cuda_call_id, response_packet_ptr->cuda_error);
        fflush(stdout);
    }

    return *response_packet_ptr;
}

__host__ cudaError_t CUDARTAPI cudaLaunchKernel(const void *hostFun,
                                                dim3 gridDim, dim3 blockDim,
                                                void **args,
                                                size_t sharedMem,
                                                cudaStream_t stream) {
    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("Prepare to launch kernel with id: %d\n", hostFun);
    }
    #if CUDART_VERSION < 10000
        cudaConfigureCall(gridDim, blockDim, sharedMem);
    #endif
    BalarCudaCallReturnPacket_t ret;
    for (unsigned index;;index++) {
        // Get param info
        ret = __balarGetParamInfo((uint64_t) hostFun, index);
        if (ret.cuda_error != cudaSuccess) {
            // Get param info return error value if the index is invalid
            break;
        } else {
            // How to pass argument? All use the value[8]
            // as cudaSetupArgument will make a copy of its content
            // so that GPGPU-Sim will know both the constant and pointer pass to the kernel
            uint8_t value[BALAR_CUDA_MAX_ARG_SIZE];
            if (ret.cudaparamconfig.size > BALAR_CUDA_MAX_ARG_SIZE) {
                printf("CUDA function argument size(%d) exceeds %d bytes limit!\n", ret.cudaparamconfig.size, BALAR_CUDA_MAX_ARG_SIZE);
            }
            memcpy(value, args[index], ret.cudaparamconfig.size);
            cudaSetupArgument((uint64_t) NULL, value, ret.cudaparamconfig.size, ret.cudaparamconfig.alignment);
        }
    }

    return cudaLaunch((uint64_t) hostFun);
}

cudaError_t cudaLaunch(uint64_t func) {
    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("Start kernel launch with id: %d\n", func);
        fflush(stdout);
    }

    BalarCudaCallPacket_t *call_packet_ptr = __balarPacket();
    call_packet_ptr->isSSTmem = true;
    call_packet_ptr->cuda_call_id = CUDA_LAUNCH;
    call_packet_ptr->cuda_launch.func = func;

    // Make cuda call, kernel launch might be blocking issue
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCallFlags(call_packet_ptr, BLOCKING_ISSUE);

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("CUDA API ID: %d with error: %d\n",
                response_packet_ptr->cuda_call_id, response_packet_ptr->cuda_error);
        fflush(stdout);
    }

    return response_packet_ptr->cuda_error;
}

// With LLVM compiler, need to pass the executable path in config script by setting
// BALAR_CUDA_EXE_PATH env var, which will overwrite file_name in Balar packet
unsigned int __cudaRegisterFatBinary(void *fatCubin) {
    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("Registering fat binary\n");
        fflush(stdout);
    }

    BalarCudaCallPacket_t *call_packet_ptr = __balarPacket();
    call_packet_ptr->isSSTmem = true;
    call_packet_ptr->cuda_call_id = CUDA_REG_FAT_BINARY;

    // Make cuda call
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCall(call_packet_ptr);

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("CUDA API ID: %d with error: %d and fatbin handle %d\n",
                response_packet_ptr->cuda_call_id, response_packet_ptr->cuda_error,
                response_packet_ptr->fat_cubin_handle);
        fflush(stdout);
    }

    return response_packet_ptr->fat_cubin_handle;
}

void __cudaRegisterFunction(
    uint64_t fatCubinHandle,
    uint64_t hostFun,
    char deviceFun[BALAR_CUDA_MAX_KERNEL_NAME]
) {
    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("Registering kernel function:\n");
        printf("Fatbin handle: %d, hostFun: %lu, deviceFunc: %s\n",
            fatCubinHandle,
            hostFun,
            deviceFun);
        fflush(stdout);
    }

    BalarCudaCallPacket_t *call_packet_ptr = __balarPacket();
    call_packet_ptr->isSSTmem = true;
    call_packet_ptr->cuda_call_id = CUDA_REG_FUNCTION;
    call_packet_ptr->register_function.fatCubinHandle = fatCubinHandle;
    call_packet_ptr->register_function.hostFun = hostFun;
    memcpy(call_packet_ptr->register_function.deviceFun, deviceFun, 256);

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("Packet's deviceFunc: %s; Addr: %p %p\nScratch mem addr: %p\n",
            call_packet_ptr->register_function.deviceFun,
            &(call_packet_ptr->register_function.deviceFun),
            &deviceFun,
            g_scratch_mem);
        fflush(stdout);
    }

    // Make cuda call
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCall(call_packet_ptr);

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("CUDA API ID: %d with error: %d\n",
                response_packet_ptr->cuda_call_id, response_packet_ptr->cuda_error);
        fflush(stdout);
    }

    return;
}

__host__ cudaError_t CUDARTAPI cudaThreadSynchronize(void) {
    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("Start thread sync\n");
        fflush(stdout);
    }

    BalarCudaCallPacket_t *call_packet_ptr = __balarPacket();
    call_packet_ptr->isSSTmem = true;
    call_packet_ptr->cuda_call_id = CUDA_THREAD_SYNC;

    // Make cuda call
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCallFlags(call_packet_ptr, BLOCKING_ISSUE | BLOCKING_COMPLETE);

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("CUDA API ID: %d with error: %d\n",
                response_packet_ptr->cuda_call_id, response_packet_ptr->cuda_error);
        fflush(stdout);
    }

    return response_packet_ptr->cuda_error;
}

__host__ cudaError_t CUDARTAPI cudaGetLastError(void) {
    BalarCudaCallReturnPacket_t *last = readLastCudaStatus();
    return last->cuda_error;
}

__host__ const char *CUDARTAPI cudaGetErrorString(cudaError_t error) {
  if (error == cudaSuccess) return "no error";
  char buf[1024];
  snprintf(buf, 1024, "<<GPGPU-Sim PTX: there was an error (code = %d)>>",
           error);
  return strdup(buf);
}

__host__ const char*CUDARTAPI cudaGetErrorName(cudaError_t error) {
  if (error == cudaSuccess)
    return "cudaSuccess";
  return "cudaError";
}

__host__ cudaError_t CUDARTAPI cudaMemset(void *mem, int c, size_t count) {
    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("Start cudaMemset, setting 0x%llx to %c for %lld bytes\n", mem, c, count);
        fflush(stdout);
    }

    BalarCudaCallPacket_t *call_packet_ptr = __balarPacket();
    call_packet_ptr->isSSTmem = true;
    call_packet_ptr->cuda_call_id = CUDA_MEMSET;
    call_packet_ptr->cudamemset.mem = mem;
    call_packet_ptr->cudamemset.c = c;
    call_packet_ptr->cudamemset.count = count;

    // Make cuda call
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCall(call_packet_ptr);

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("CUDA API ID: %d with error: %d\n",
                response_packet_ptr->cuda_call_id, response_packet_ptr->cuda_error);
        fflush(stdout);
    }

    return response_packet_ptr->cuda_error;
}

// Weili: There might be an issue with passing deviceName directly.
// As its value is a pointer in Vanadis memory space and this value
// will be used to instantiate std::string.
// Which should cause issue if that Vanadis's address is invalid
// in simulator memory space.
// So we use a fixed size buffer in the packet to pass by value
void __cudaRegisterVar(
    void **fatCubinHandle,
    char *hostVar,           // pointer to...something
    char *deviceAddress,     // name of variable
    const char *deviceName,  // name of variable (same as above)
    int ext, int size, int constant, int global) {
    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("Start registering var\n");
        printf("HostVar 0x%Lx\n", hostVar);
        printf("deviceAddress 0x%Lx\n", deviceAddress);
        printf("deviceName %s\n", deviceName);
        fflush(stdout);
    }

    BalarCudaCallPacket_t *call_packet_ptr = __balarPacket();
    call_packet_ptr->isSSTmem = true;
    call_packet_ptr->cuda_call_id = CUDA_REG_VAR;
    call_packet_ptr->register_var.fatCubinHandle = fatCubinHandle;
    call_packet_ptr->register_var.hostVar = hostVar;
    call_packet_ptr->register_var.deviceAddress = deviceAddress;
    // Copy from char pointer to the array so we don't need to access
    // Vanadis's memory space for this value.
    strncpy(call_packet_ptr->register_var.deviceName, deviceName, strlen(deviceName));
    call_packet_ptr->register_var.deviceName[strlen(deviceName)] = '\0';
    call_packet_ptr->register_var.ext = ext;
    call_packet_ptr->register_var.size = size;
    call_packet_ptr->register_var.constant = constant;
    call_packet_ptr->register_var.global = global;

    // Make cuda call
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCall(call_packet_ptr);

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("CUDA API ID: %d with error: %d\n",
                response_packet_ptr->cuda_call_id, response_packet_ptr->cuda_error);
        fflush(stdout);
    }
}

__host__ cudaError_t CUDARTAPI cudaGetDeviceCount(int *count) {
    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("Start get device count\n");
        fflush(stdout);
    }

    BalarCudaCallPacket_t *call_packet_ptr = __balarPacket();
    call_packet_ptr->isSSTmem = true;
    call_packet_ptr->cuda_call_id = CUDA_GET_DEVICE_COUNT;

    // Make cuda call
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCall(call_packet_ptr);

    // Store count back
    *count = response_packet_ptr->cudagetdevicecount.count;

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("CUDA API ID: %d with error: %d\n",
                response_packet_ptr->cuda_call_id, response_packet_ptr->cuda_error);
        fflush(stdout);
    }

    return response_packet_ptr->cuda_error;
}

__host__ cudaError_t CUDARTAPI cudaSetDevice(int device) {
    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("Start get device count\n");
        fflush(stdout);
    }

    BalarCudaCallPacket_t *call_packet_ptr = __balarPacket();
    call_packet_ptr->isSSTmem = true;
    call_packet_ptr->cuda_call_id = CUDA_SET_DEVICE;
    call_packet_ptr->cudasetdevice.device = device;

    // Make cuda call
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCall(call_packet_ptr);

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("CUDA API ID: %d with error: %d\n",
                response_packet_ptr->cuda_call_id, response_packet_ptr->cuda_error);
        fflush(stdout);
    }

    return response_packet_ptr->cuda_error;
}

__host__ struct cudaChannelFormatDesc CUDARTAPI cudaCreateChannelDesc(
    int x, int y, int z, int w, enum cudaChannelFormatKind f) {
  struct cudaChannelFormatDesc dummy;
  dummy.x = x;
  dummy.y = y;
  dummy.z = z;
  dummy.w = w;
  dummy.f = f;
  return dummy;
}

void __cudaRegisterTexture(
    void **fatCubinHandle, const struct textureReference *hostVar,
    const void **deviceAddress, const char *deviceName, int dim, int norm,
    int ext) {

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("Start register texture\n");
        fflush(stdout);
    }

    BalarCudaCallPacket_t *call_packet_ptr = __balarPacket();
    call_packet_ptr->isSSTmem = true;
    call_packet_ptr->cuda_call_id = CUDA_REG_TEXTURE;
    call_packet_ptr->cudaregtexture.fatCubinHandle = fatCubinHandle;
    call_packet_ptr->cudaregtexture.hostVar_ptr = (uint64_t) hostVar;
    call_packet_ptr->cudaregtexture.texRef = *hostVar;
    call_packet_ptr->cudaregtexture.deviceAddress = deviceAddress;
    strncpy(call_packet_ptr->cudaregtexture.deviceName, deviceName, strlen(deviceName));
    call_packet_ptr->cudaregtexture.deviceName[strlen(deviceName)] = '\0';
    call_packet_ptr->cudaregtexture.dim = dim;
    call_packet_ptr->cudaregtexture.norm = norm;
    call_packet_ptr->cudaregtexture.ext = ext;

    // Make cuda call
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCall(call_packet_ptr);

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("CUDA API ID: %d with error: %d\n",
                response_packet_ptr->cuda_call_id, response_packet_ptr->cuda_error);
        fflush(stdout);
    }
}

__host__ cudaError_t CUDARTAPI cudaBindTexture(
    size_t *offset, const struct textureReference *hostVar, const void *devPtr,
    const struct cudaChannelFormatDesc *desc, size_t size) {
    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("Start bind texture\n");
        fflush(stdout);
    }

    BalarCudaCallPacket_t *call_packet_ptr = __balarPacket();
    call_packet_ptr->isSSTmem = true;
    call_packet_ptr->cuda_call_id = CUDA_BIND_TEXTURE;
    call_packet_ptr->cudabindtexture.offset = offset;
    call_packet_ptr->cudabindtexture.hostVar_ptr = (uint64_t) hostVar;
    call_packet_ptr->cudabindtexture.texRef = *hostVar;
    call_packet_ptr->cudabindtexture.devPtr = devPtr;
    call_packet_ptr->cudabindtexture.desc_struct = *desc;
    call_packet_ptr->cudabindtexture.size = size;

    // Make cuda call
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCall(call_packet_ptr);

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("CUDA API ID: %d with error: %d\n",
                response_packet_ptr->cuda_call_id, response_packet_ptr->cuda_error);
        fflush(stdout);
    }

    return response_packet_ptr->cuda_error;
}

__host__ cudaError_t CUDARTAPI cudaFreeHost(void *ptr) {
  free(ptr);
  return cudaSuccess;
}

// A bit tricky as GPGPUSim needs to track the malloc size as well
__host__ cudaError_t CUDARTAPI cudaMallocHost(void **ptr, size_t size) {
    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("Start cuda malloc host with size %lld\n", size);
        fflush(stdout);
    }

    // Actual malloc done in vanadis, just need to tell GPGPU-Sim about the
    // malloc addr and size info
    *ptr = malloc(size);

    BalarCudaCallPacket_t *call_packet_ptr = __balarPacket();
    call_packet_ptr->isSSTmem = true;
    call_packet_ptr->cuda_call_id = CUDA_MALLOC_HOST;
    call_packet_ptr->cudamallochost.addr = *ptr;
    call_packet_ptr->cudamallochost.size = size;

    // Make cuda call
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCall(call_packet_ptr);

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("CUDA API ID: %d with error: %d\n",
                response_packet_ptr->cuda_call_id, response_packet_ptr->cuda_error);
        fflush(stdout);
    }
    return response_packet_ptr->cuda_error;
}

// Added these CUDA calls that do nothing, just to make clang happy
void CUDARTAPI __cudaRegisterFatBinaryEnd( void **fatCubinHandle ) {
	return;
}

void __cudaUnregisterFatBinary(void **fatCubinHandle) {
	return;
}

__host__ cudaError_t CUDARTAPI cudaFree(void *devPtr) {
	return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI
cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("Start cudaGetDeviceProperties on device\n", device);
        fflush(stdout);
    }

    BalarCudaCallPacket_t *call_packet_ptr = __balarPacket();
    call_packet_ptr->isSSTmem = true;
    call_packet_ptr->cuda_call_id = CUDA_GET_DEVICE_PROPERTIES;
    call_packet_ptr->cudaGetDeviceProperties.device = device;

    // Make cuda call
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCall(call_packet_ptr);

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("CUDA API ID: %d with error: %d\n",
                response_packet_ptr->cuda_call_id, response_packet_ptr->cuda_error);
        fflush(stdout);
    }

    // Write back the device properties
    // use memcpy for struct copy
    memcpy(prop, &(response_packet_ptr->cudaGetDeviceProperties.prop), sizeof(struct cudaDeviceProp));

    return response_packet_ptr->cuda_error;
}

cudaError_t CUDARTAPI cudaSetDeviceFlags(unsigned int flags) {
    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("Start cudaSetDeviceFlags with flags %x\n", flags);
        fflush(stdout);
    }

    BalarCudaCallPacket_t *call_packet_ptr = __balarPacket();
    call_packet_ptr->isSSTmem = true;
    call_packet_ptr->cuda_call_id = CUDA_SET_DEVICE_FLAGS;
    call_packet_ptr->cudaSetDeviceFlags.flags = flags;

    // Make cuda call
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCall(call_packet_ptr);

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("CUDA API ID: %d with error: %d\n",
                response_packet_ptr->cuda_call_id, response_packet_ptr->cuda_error);
        fflush(stdout);
    }

    return response_packet_ptr->cuda_error;
}

__host__ cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *stream) {
    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("Start cudaStreamCreate\n");
        fflush(stdout);
    }

    BalarCudaCallPacket_t *call_packet_ptr = __balarPacket();
    call_packet_ptr->isSSTmem = true;
    call_packet_ptr->cuda_call_id = CUDA_STREAM_CREATE;

    // Make cuda call
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCall(call_packet_ptr);

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("CUDA API ID: %d with error: %d and stream %p\n",
                response_packet_ptr->cuda_call_id, response_packet_ptr->cuda_error,
                response_packet_ptr->cudaStreamCreate.stream);
        fflush(stdout);
    }

    // Write back the stream
    *stream = (cudaStream_t) response_packet_ptr->cudaStreamCreate.stream;

    return response_packet_ptr->cuda_error;
}

__host__ cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event) {
    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("Start cudaEventCreate\n");
        fflush(stdout);
    }

    BalarCudaCallPacket_t *call_packet_ptr = __balarPacket();
    call_packet_ptr->isSSTmem = true;
    call_packet_ptr->cuda_call_id = CUDA_EVENT_CREATE;

    // Make cuda call
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCall(call_packet_ptr);

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("CUDA API ID: %d with error: %d and event %p\n",
                response_packet_ptr->cuda_call_id, response_packet_ptr->cuda_error,
                response_packet_ptr->cudaEventCreate.event);
        fflush(stdout);
    }

    // Write back the event
    *event = (cudaEvent_t) response_packet_ptr->cudaEventCreate.event;

    return response_packet_ptr->cuda_error;
}

cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) {
    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("Start cudaEventCreateWithFlags with %x\n", flags);
        fflush(stdout);
    }

    BalarCudaCallPacket_t *call_packet_ptr = __balarPacket();
    call_packet_ptr->isSSTmem = true;
    call_packet_ptr->cuda_call_id = CUDA_EVENT_CREATE_WITH_FLAGS;
    call_packet_ptr->cudaEventCreateWithFlags.flags = flags;

    // Make cuda call
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCall(call_packet_ptr);

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("CUDA API ID: %d with error: %d and event %p\n",
                response_packet_ptr->cuda_call_id, response_packet_ptr->cuda_error,
                response_packet_ptr->cudaEventCreateWithFlags.event);
        fflush(stdout);
    }

    // Write back the event
    *event = (cudaEvent_t) response_packet_ptr->cudaEventCreateWithFlags.event;

    return response_packet_ptr->cuda_error;
}

__host__ cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event,
                                               cudaStream_t stream) {
    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("Start cudaEventRecord with event %p and stream %p\n", event, stream);
        fflush(stdout);
    }

    BalarCudaCallPacket_t *call_packet_ptr = __balarPacket();
    call_packet_ptr->isSSTmem = true;
    call_packet_ptr->cuda_call_id = CUDA_EVENT_RECORD;
    call_packet_ptr->cudaEventRecord.event = event;
    call_packet_ptr->cudaEventRecord.stream = stream;

    // Make cuda call
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCallFlags(call_packet_ptr, BLOCKING_ISSUE);

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("CUDA API ID: %d with error: %d\n",
                response_packet_ptr->cuda_call_id, response_packet_ptr->cuda_error);
        fflush(stdout);
    }

    return response_packet_ptr->cuda_error;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src,
                                               size_t count,
                                               enum cudaMemcpyKind kind,
                                               cudaStream_t stream) {
    // Send request to GPU
    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("cudaMemcpyAsync dst: %llx\n", dst);
        fflush(stdout);
    }
    BalarCudaCallPacket_t *call_packet_ptr = __balarPacket();
    call_packet_ptr->isSSTmem = true;
    call_packet_ptr->cuda_call_id = CUDA_MEMCPY_ASYNC;

    call_packet_ptr->cudaMemcpyAsync.kind = kind;
    call_packet_ptr->cudaMemcpyAsync.dst = (uint64_t) dst;
    call_packet_ptr->cudaMemcpyAsync.src = (uint64_t) src;
    call_packet_ptr->cudaMemcpyAsync.count = count;
    call_packet_ptr->cudaMemcpyAsync.payload = (uint64_t) src;
    call_packet_ptr->cudaMemcpyAsync.stream = stream;

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("cudaMemcpyAsync Packet address: %p\n", call_packet_ptr);
        printf("cudaMemcpyAsync kind: %d\n", (uint8_t) (call_packet_ptr->cuda_memcpy.kind));
        printf("cudaMemcpyAsync size: %ld\n", count);
        printf("cudaMemcpyAsync src: %p\n", src);
        printf("cudaMemcpyAsync dst: %p\n", dst);
        printf("cudaMemcpyAsync stream: %p\n", stream);
        fflush(stdout);
    }

    // Make cuda call
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCallFlags(call_packet_ptr, BLOCKING_ISSUE);

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("CUDA API ID: %d with error: %d\n",
                response_packet_ptr->cuda_call_id, response_packet_ptr->cuda_error);
        fflush(stdout);
    }

    return response_packet_ptr->cuda_error;
}

__host__ cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event) {
    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("Start cudaEventSynchronize with event %p\n", event);
        fflush(stdout);
    }

    BalarCudaCallPacket_t *call_packet_ptr = __balarPacket();
    call_packet_ptr->isSSTmem = true;
    call_packet_ptr->cuda_call_id = CUDA_EVENT_SYNCHRONIZE;
    call_packet_ptr->cudaEventSynchronize.event = event;

    // Make cuda call
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCallFlags(call_packet_ptr, BLOCKING_COMPLETE);

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("CUDA API ID: %d with error: %d\n",
                response_packet_ptr->cuda_call_id, response_packet_ptr->cuda_error);
        fflush(stdout);
    }

    return response_packet_ptr->cuda_error;
}

__host__ cudaError_t CUDARTAPI cudaEventElapsedTime(float *ms,
                                                    cudaEvent_t start,
                                                    cudaEvent_t end) {
    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("Start cudaEventElapsedTime with start %p and end %p\n", start, end);
        fflush(stdout);
    }

    BalarCudaCallPacket_t *call_packet_ptr = __balarPacket();
    call_packet_ptr->isSSTmem = true;
    call_packet_ptr->cuda_call_id = CUDA_EVENT_ELAPSED_TIME;
    call_packet_ptr->cudaEventElapsedTime.start = start;
    call_packet_ptr->cudaEventElapsedTime.end = end;

    // Make cuda call
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCall(call_packet_ptr);

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("CUDA API ID: %d with error: %d and time %f ms\n",
                response_packet_ptr->cuda_call_id, response_packet_ptr->cuda_error,
                response_packet_ptr->cudaEventElapsedTime.ms);
        fflush(stdout);
    }

    // Write back the time
    *ms = response_packet_ptr->cudaEventElapsedTime.ms;

    return response_packet_ptr->cuda_error;
}

__host__ cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("Start cudaStreamSynchronize with stream %p\n", stream);
        fflush(stdout);
    }

    BalarCudaCallPacket_t *call_packet_ptr = __balarPacket();
    call_packet_ptr->isSSTmem = true;
    call_packet_ptr->cuda_call_id = CUDA_STREAM_SYNC;
    call_packet_ptr->cudaStreamSynchronize.stream = stream;

    // Make cuda call
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCallFlags(call_packet_ptr, BLOCKING_COMPLETE);

    // Wait for cudaEventSynchronize to complete by polling
    while (!(response_packet_ptr->is_cuda_call_done)) {
        int wait = 0;
        response_packet_ptr = readLastCudaStatus();
        wait++;
    }

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("CUDA API ID: %d with error: %d\n",
                response_packet_ptr->cuda_call_id, response_packet_ptr->cuda_error);
        fflush(stdout);
    }

    return response_packet_ptr->cuda_error;
}

__host__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream) {
    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("Start cudaStreamDestroy with stream %p\n", stream);
        fflush(stdout);
    }

    // First we need to synchronize the stream
    cudaStreamSynchronize(stream);

    BalarCudaCallPacket_t *call_packet_ptr = __balarPacket();
    call_packet_ptr->isSSTmem = true;
    call_packet_ptr->cuda_call_id = CUDA_STREAM_DESTROY;
    call_packet_ptr->cudaStreamDestroy.stream = stream;

    // Make cuda call
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCall(call_packet_ptr);

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("CUDA API ID: %d with error: %d\n",
                response_packet_ptr->cuda_call_id, response_packet_ptr->cuda_error);
        fflush(stdout);
    }

    return response_packet_ptr->cuda_error;
}

__host__ cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) {
    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("Start cudaEventDestroy with event %p\n", event);
        fflush(stdout);
    }

    BalarCudaCallPacket_t *call_packet_ptr = __balarPacket();
    call_packet_ptr->isSSTmem = true;
    call_packet_ptr->cuda_call_id = CUDA_EVENT_DESTROY;
    call_packet_ptr->cudaEventDestroy.event = event;

    // Make cuda call
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCall(call_packet_ptr);

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("CUDA API ID: %d with error: %d\n",
                response_packet_ptr->cuda_call_id, response_packet_ptr->cuda_error);
        fflush(stdout);
    }

    return response_packet_ptr->cuda_error;
}

__host__ cudaError_t CUDARTAPI cudaDeviceGetAttribute(int *value,
                                                      enum cudaDeviceAttr attr,
                                                      int device) {
    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("Start cudaDeviceGetAttribute with device %d\n", device);
        fflush(stdout);
    }

    BalarCudaCallPacket_t *call_packet_ptr = __balarPacket();
    call_packet_ptr->isSSTmem = true;
    call_packet_ptr->cuda_call_id = CUDA_DEVICE_GET_ATTRIBUTE;
    call_packet_ptr->cudaDeviceGetAttribute.attr = attr;
    call_packet_ptr->cudaDeviceGetAttribute.device = device;

    // Make cuda call
    BalarCudaCallReturnPacket_t *response_packet_ptr = makeCudaCall(call_packet_ptr);

    if (g_debug_level >= LOG_LEVEL_DEBUG) {
        printf("CUDA API ID: %d with error: %d\n",
                response_packet_ptr->cuda_call_id, response_packet_ptr->cuda_error);
        fflush(stdout);
    }

    // Write back the value
    *value = response_packet_ptr->cudaDeviceGetAttribute.value;

    return response_packet_ptr->cuda_error;
}

// Required by the cuda-samples, but haven't been implemented yet
// in GPGPU-Sim
__host__ cudaError_t cudaHostRegister(void *ptr, size_t size,
                                      unsigned int flags) {
    printf("WARNING: cudaHostRegister has not been implemented yet.");
    return cudaSuccess;
}

__host__ cudaError_t cudaHostUnregister(void *ptr) {
    printf("WARNING: cudaHostUnregister has not been implemented yet.");
    return cudaSuccess;
}



}