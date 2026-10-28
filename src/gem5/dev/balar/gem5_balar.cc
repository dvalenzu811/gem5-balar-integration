#include "dev/balar/gem5_balar.hh"
#include "dev/balar/balar_gpu_api.hh"

#include <cstring>
#include <cstdio>
#include <algorithm>
#include <memory>

#include "base/logging.hh"
#include "mem/packet.hh"
#include "sim/system.hh"

/*
 * GPGPU-Sim declares the memcpy callbacks in the global C++ namespace.
 * These strong definitions therefore MUST also live in the global
 * namespace so their mangled symbols match exactly.
 */
namespace
{

gem5::Gem5Balar *gBalarCallbackTarget = nullptr;

} // anonymous namespace


void
SST_callback_memcpy_H2D_done(
    uint64_t dst,
    uint64_t src,
    size_t count,
    cudaStream_t stream)
{
    fprintf(stderr,
        "[GEM5-BALAR-CB] H2D DONE dst=0x%llx src=0x%llx count=%zu stream=%p\n",
        (unsigned long long)dst,
        (unsigned long long)src,
        count,
        (void *)stream);
    fflush(stderr);

    (void)stream;

    panic_if(
        gBalarCallbackTarget == nullptr,
        "BALAR H2D callback without Gem5Balar instance"
    );

    gBalarCallbackTarget->memcpyH2DDone(dst, src, count);
}


void
SST_callback_memcpy_D2H_done(
    uint64_t dst,
    uint64_t src,
    size_t count,
    cudaStream_t stream)
{   
    fprintf(stderr,
        "[GEM5-BALAR-CB] D2H DONE dst=0x%llx src=0x%llx count=%zu stream=%p\n",
        (unsigned long long)dst,
        (unsigned long long)src,
        count,
        (void *)stream);
    fflush(stderr);

    (void)stream;

    panic_if(
        gBalarCallbackTarget == nullptr,
        "BALAR D2H callback without Gem5Balar instance"
    );

    gBalarCallbackTarget->memcpyD2HDone(dst, src, count);
}

void
SST_callback_kernel_done(cudaStream_t stream)
{
    fprintf(
        stderr,
        "[GEM5-BALAR-CB] KERNEL DONE stream=%p\n",
        (void *)stream
    );
    fflush(stderr);

    panic_if(
        gBalarCallbackTarget == nullptr,
        "BALAR kernel callback without Gem5Balar instance"
    );

    gBalarCallbackTarget->kernelDone(
        static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(stream)
        )
    );
}

/*
 * Strong replacements for the weak SST hooks provided by GPGPU-Sim.
 *
 * IMPORTANT:
 * These functions have C++ linkage in GPGPU-Sim, so DO NOT wrap
 * them in extern "C".
 */

bool
is_SST_buffer_full(unsigned core_id)
{
    panic_if(
        gBalarCallbackTarget == nullptr,
        "is_SST_buffer_full without Gem5Balar instance"
    );

    return gBalarCallbackTarget->gpuMemBufferFull(core_id);
}


void
send_read_request_SST(
    unsigned core_id,
    uint64_t address,
    size_t size,
    void *mem_req)
{
    panic_if(
        gBalarCallbackTarget == nullptr,
        "send_read_request_SST without Gem5Balar instance"
    );

    gBalarCallbackTarget->sendGpuMemRead(
        core_id,
        address,
        size,
        mem_req
    );
}


void
send_write_request_SST(
    unsigned core_id,
    uint64_t address,
    size_t size,
    void *mem_req)
{
    panic_if(
        gBalarCallbackTarget == nullptr,
        "send_write_request_SST without Gem5Balar instance"
    );

    gBalarCallbackTarget->sendGpuMemWrite(
        core_id,
        address,
        size,
        mem_req
    );
}

namespace gem5
{

Gem5Balar::GpuMemPort::GpuMemPort(
    const std::string &name,
    Gem5Balar &owner)
    : RequestPort(name),
      owner(owner)
{
}


bool
Gem5Balar::GpuMemPort::recvTimingResp(PacketPtr pkt)
{
    return owner.recvGpuMemTimingResp(pkt);
}


void
Gem5Balar::GpuMemPort::recvReqRetry()
{
    owner.recvGpuMemReqRetry();
}

Gem5Balar::Gem5Balar(const Params &p)
    : DmaDevice(p),
      pioAddr(p.pio_addr),
      pioSize(p.pio_size),
      pioDelay(p.pio_latency),
      gpuMemPort(name() + ".gpu_mem", *this),
      dmaReadDoneEvent([this] { dmaReadDone(); }, name()),
      dmaWriteDoneEvent([this] { dmaWriteDone(); }, name()),
      memcpyDataReadDoneEvent(
          [this] { memcpyDataReadDone(); }, name()),
      memcpyDataWriteDoneEvent(
          [this] { memcpyDataWriteDone(); }, name()),
      gpuTickEvent([this] { gpuTick(); }, name())
{
    panic_if(
        gBalarCallbackTarget != nullptr,
        "%s: only one Gem5Balar callback target is supported",
        name()
    );

    gBalarCallbackTarget = this;

    /*
     * Keep an explicit reference to both strong callbacks in the
     * gem5 executable. GPGPU-Sim provides weak definitions that
     * these symbols must override.
     */
    inform(
        "%s: [BALAR-CB] callbacks H2D=%p D2H=%p KERNEL=%p",
        name(),
        reinterpret_cast<void *>(&SST_callback_memcpy_H2D_done),
        reinterpret_cast<void *>(&SST_callback_memcpy_D2H_done),
        reinterpret_cast<void *>(&SST_callback_kernel_done)
    );

    gpuCoreCount = p.gpu_cores;
    gpuClockPeriod = p.gpu_clock_period;
    cudaExecutable = p.cuda_executable;
        maxPendingCacheTrans = p.max_cache_trans;

    gpuMemOutstandingPerCluster.assign(
        gpuCoreCount,
        0
    );

    gpuMemRequestorId =
        sys->getRequestorId(this);

    inform(
        "%s: [BALAR-GPU-MEM] clusters=%u max_pending_per_cluster=%u "
        "requestor_id=%u",
        name(),
        gpuCoreCount,
        maxPendingCacheTrans,
        (unsigned)gpuMemRequestorId
    );

    if (!cudaExecutable.empty()) {
        inform("%s: [BALAR-INIT] CUDA executable=%s",
               name(),
               cudaExecutable.c_str());
    }

    inform("%s: [BALAR-GPU-CLOCK] configured period=%llu ticks",
           name(),
           (unsigned long long)gpuClockPeriod);

    inform("%s: [BALAR-INIT] initializing GPGPU-Sim with %u GPU clusters",
           name(), gpuCoreCount);

    SST_gpgpusim_numcores_equal_check(gpuCoreCount);

    launchBlocking = SST_gpgpusim_launch_blocking();

    inform("%s: [BALAR-INIT] GPGPU-Sim ready, launch_blocking=%u",
           name(), (unsigned)launchBlocking);

    inform("%s: Gem5Balar created. MMIO [%#llx - %#llx]",
           name(),
           (unsigned long long)pioAddr,
           (unsigned long long)(pioAddr + pioSize - 1));
}

AddrRangeList
Gem5Balar::getAddrRanges() const
{
    AddrRangeList ranges;
    ranges.push_back(RangeSize(pioAddr, pioSize));
    return ranges;
}

Port &
Gem5Balar::getPort(
    const std::string &if_name,
    PortID idx)
{
    if (if_name == "gpu_mem")
        return gpuMemPort;

    return DmaDevice::getPort(if_name, idx);
}

Tick
Gem5Balar::read(PacketPtr pkt)
{
    panic_if(pkt->getAddr() < pioAddr ||
             pkt->getAddr() >= pioAddr + pioSize,
             "%s: read outside MMIO range", name());

    panic_if(pkt->getSize() != sizeof(uint64_t),
             "%s: only 64-bit MMIO reads supported for now", name());

    /*
     * Balar return protocol:
     *
     *   0              -> CUDA response not ready yet
     *   dmaSourceAddr  -> physical address containing
     *                     BalarCudaCallReturnPacket_t
     *
     * The original Balar protocol reuses the same scratch
     * physical address for request and response.
     */
    const uint64_t responsePhys =
        responseReady ? dmaSourceAddr : 0;

    pkt->setUintX(responsePhys, ByteOrder::little);

    if (responseReady) {
        inform(
            "%s: [BALAR-MMIO] READ response_phys=%#llx ready=1",
            name(),
            (unsigned long long)responsePhys
        );
    }

    pkt->makeResponse();
    return pioDelay;
}

Tick
Gem5Balar::write(PacketPtr pkt)
{
    panic_if(pkt->getAddr() < pioAddr ||
             pkt->getAddr() >= pioAddr + pioSize,
             "%s: write outside MMIO range", name());

    panic_if(pkt->getSize() != sizeof(uint64_t),
             "%s: only 64-bit MMIO writes supported for now", name());

    const Addr offset = pkt->getAddr() - pioAddr;
    const uint64_t value = pkt->getUintX(ByteOrder::little);

    panic_if(offset != 0,
             "%s: only MMIO offset 0 is implemented for now", name());

    panic_if(dmaBusy || dmaPending(),
             "%s: overlapping Balar DMA request", name());

    /*
     * M3.5: start the gem5-driven equivalent of Balar's SST clock
     * on first real interaction with the accelerator.
     */
    startGpuClock();

    dmaSourceAddr = value;
    dmaBusy = true;
    responseReady = false;

    std::memset(&cudaPacket, 0, sizeof(cudaPacket));
    std::memset(&cudaReturnPacket, 0, sizeof(cudaReturnPacket));

    inform("%s: MMIO WRITE offset=0 packet_addr=%#llx",
           name(),
           (unsigned long long)dmaSourceAddr);

    inform("%s: starting DMA READ addr=%#llx size=%llu",
           name(),
           (unsigned long long)dmaSourceAddr,
           (unsigned long long)sizeof(cudaPacket));

    dmaRead(
        dmaSourceAddr,
        sizeof(cudaPacket),
        &dmaReadDoneEvent,
        reinterpret_cast<uint8_t *>(&cudaPacket)
    );

    pkt->makeResponse();
    return pioDelay;
}

void
Gem5Balar::startGpuClock()
{
    if (gpuClockRunning)
        return;

    panic_if(gpuClockPeriod == 0,
             "%s: GPU clock period cannot be zero",
             name());

    gpuClockRunning = true;
    gpuTickCount = 0;

    inform("%s: [BALAR-GPU-CLOCK] START tick=%llu period=%llu",
           name(),
           (unsigned long long)curTick(),
           (unsigned long long)gpuClockPeriod);

    schedule(
        gpuTickEvent,
        curTick() + gpuClockPeriod
    );
}

void
Gem5Balar::gpuTick()
{
    if (!gpuClockRunning)
        return;

    ++gpuTickCount;

    /*
     * Exact replacement of:
     *
     *     BalarMMIO::clockTic()
     *         -> SST_gpu_core_cycle()
     *
     * SST_gpu_core_cycle() wraps SST_Cycle().
     */
    const bool done = SST_gpu_core_cycle();

    /*
     * Don't print every 1 ns tick forever.
     *
     * First ticks validate EventQueue timing; afterwards
     * only occasional progress or the stop condition.
     */
    if (gpuTickCount <= 8 ||
        (gpuTickCount % 1000000ULL) == 0 ||
        done) {
        inform("%s: [BALAR-GPU-CLOCK] cycle=%llu tick=%llu done=%u",
               name(),
               (unsigned long long)gpuTickCount,
               (unsigned long long)curTick(),
               (unsigned)done);
    }

    /*
     * Original SST clock handler returns the bool produced
     * by SST_gpu_core_cycle().
     *
     * SST_Cycle() normally returns false and only returns
     * true for its terminal/break-limit condition.
     */
    if (done) {

        uint64_t totalOutstanding = 0;

        for (const auto outstanding :
             gpuMemOutstandingPerCluster) {
            totalOutstanding += outstanding;
        }

        inform(
            "%s: [BALAR-GPU-MEM] SUMMARY "
            "issued=%llu completed=%llu "
            "blocked=%llu retries=%llu "
            "outstanding=%llu queued=%llu waiting_retry=%u",
            name(),
            (unsigned long long)gpuMemFetchIssued,
            (unsigned long long)gpuMemFetchCompleted,
            (unsigned long long)gpuMemBlockedCount,
            (unsigned long long)gpuMemRetryCount,
            (unsigned long long)totalOutstanding,
            (unsigned long long)gpuMemSendQueue.size(),
            (unsigned)gpuMemWaitingForRetry
        );

        gpuClockRunning = false;

        inform("%s: [BALAR-GPU-CLOCK] STOP cycle=%llu tick=%llu",
               name(),
               (unsigned long long)gpuTickCount,
               (unsigned long long)curTick());

        return;
    }

    schedule(
        gpuTickEvent,
        curTick() + gpuClockPeriod
    );
}

void
Gem5Balar::dispatchCudaCall()
{
    constexpr uint32_t CUDA_REG_FAT_BINARY = 1;
    constexpr uint32_t CUDA_REG_FUNCTION   = 2;
    constexpr uint32_t CUDA_MEMCPY         = 3;
    constexpr uint32_t CUDA_CONFIG_CALL    = 4;
    constexpr uint32_t CUDA_SET_ARG        = 5;
    constexpr uint32_t CUDA_LAUNCH         = 6;
    constexpr uint32_t CUDA_MALLOC         = 9;
    constexpr uint32_t CUDA_PARAM_CONFIG   = 12;

    inform("%s: [BALAR-DECODE] dispatch CUDA API id=%u",
           name(),
           (unsigned)cudaPacket.cudaCallId);

    std::memset(
        &cudaReturnPacket,
        0,
        sizeof(cudaReturnPacket)
    );

    cudaReturnPacket.cudaCallId =
        cudaPacket.cudaCallId;

    cudaReturnPacket.cudaError = 0;
    cudaReturnPacket.isDone = 1;

    switch (cudaPacket.cudaCallId) {

      case CUDA_REG_FAT_BINARY:
      {
        auto &p = cudaPacket.payload.registerFatbin;

        p.fileName[255] = '\0';

        const char *fileName =
            cudaExecutable.empty()
                ? p.fileName
                : cudaExecutable.c_str();

        inform(
            "%s: [BALAR-API] CUDA_REG_FAT_BINARY file=%s",
            name(),
            fileName
        );

        unsigned handle =
            __cudaRegisterFatBinarySST(
                const_cast<char *>(fileName)
            );

        cudaReturnPacket.payload.fatCubinHandle =
            static_cast<uint64_t>(handle);

        inform(
            "%s: [BALAR-GPU] fatbin handle=%#llx",
            name(),
            (unsigned long long)
                cudaReturnPacket.payload.fatCubinHandle
        );

        break;
      }


      case CUDA_REG_FUNCTION:
      {
        auto &p =
            cudaPacket.payload.registerFunction;

        p.deviceFun[511] = '\0';

        inform(
            "%s: [BALAR-API] CUDA_REG_FUNCTION "
            "handle=%#llx hostFun=%#llx deviceFun=%s",
            name(),
            (unsigned long long)p.fatCubinHandle,
            (unsigned long long)p.hostFun,
            p.deviceFun
        );

        __cudaRegisterFunctionSST(
            p.fatCubinHandle,
            p.hostFun,
            p.deviceFun
        );

        break;
      }



      case CUDA_MEMCPY:
      {
        const auto &p =
            cudaPacket.payload.cudaMemcpy;

        panic_if(
            !cudaPacket.isSstMem,
            "%s: CUDA_MEMCPY currently requires isSSTmem=1",
            name()
        );

        panic_if(
            p.kind != static_cast<uint32_t>(cudaMemcpyHostToDevice) &&
            p.kind != static_cast<uint32_t>(cudaMemcpyDeviceToHost),
            "%s: unsupported CUDA_MEMCPY kind=%u",
            name(),
            p.kind
        );

        inform(
            "%s: [BALAR-API] CUDA_MEMCPY "
            "dst=%#llx src=%#llx count=%llu kind=%u",
            name(),
            (unsigned long long)p.dst,
            (unsigned long long)p.src,
            (unsigned long long)p.count,
            p.kind
        );

        /*
         * The guest Linux runtime already guarantees that one packet
         * never crosses a physical 4 KiB page. Here we expand that
         * fragment to complete 64-byte CHI cache lines.
         *
         * H2D:
         *   guest RAM --dmaRead--> buffer --cudaMemcpy--> GPU
         *
         * D2H:
         *   first read the complete destination lines so bytes outside
         *   the requested sub-range survive the later full-line write.
         */
        if (p.count == 0)
            break;

        const bool h2d =
            p.kind ==
            static_cast<uint32_t>(cudaMemcpyHostToDevice);

        const Addr guestPhys =
            h2d ?
            static_cast<Addr>(p.src) :
            static_cast<Addr>(p.dst);

        memcpyGpuAddr =
            h2d ? p.dst : p.src;

        memcpyCount =
            static_cast<size_t>(p.count);

        constexpr Addr DmaLineBytes = 64;
        constexpr Addr DmaLineMask = DmaLineBytes - 1;

        memcpyGuestAlignedAddr =
            guestPhys & ~DmaLineMask;

        memcpyDataOffset =
            static_cast<size_t>(
                guestPhys - memcpyGuestAlignedAddr
            );

        memcpyAlignedBytes =
            (
                memcpyDataOffset +
                memcpyCount +
                DmaLineMask
            ) & ~static_cast<size_t>(DmaLineMask);

        memcpyBuffer.assign(
            memcpyAlignedBytes,
            0
        );

        cudaReturnPacket.isDone = 0;

        memcpyPhase =
            h2d ?
            MemcpyPhase::H2DRead :
            MemcpyPhase::D2HRead;

        inform(
            "%s: [BALAR-MEMCPY] DMA READ "
            "guest=%#llx aligned=%#llx bytes=%llu "
            "offset=%llu phase=%s",
            name(),
            (unsigned long long)guestPhys,
            (unsigned long long)memcpyGuestAlignedAddr,
            (unsigned long long)memcpyAlignedBytes,
            (unsigned long long)memcpyDataOffset,
            h2d ? "H2D" : "D2H-RMW"
        );

        dmaRead(
            memcpyGuestAlignedAddr,
            memcpyAlignedBytes,
            &memcpyDataReadDoneEvent,
            memcpyBuffer.data()
        );

        /*
         * The response is deliberately deferred until the complete
         * synchronous cudaMemcpy has finished.
         */
        return;
      }


      case CUDA_CONFIG_CALL:
      {
        const auto &p =
            cudaPacket.payload.configureCall;

        dim3 grid {
            p.gdx,
            p.gdy,
            p.gdz
        };

        dim3 block {
            p.bdx,
            p.bdy,
            p.bdz
        };

        inform(
            "%s: [BALAR-API] CUDA_CONFIG_CALL "
            "grid=(%u,%u,%u) block=(%u,%u,%u) "
            "shared=%llu stream=%#llx",
            name(),
            grid.x, grid.y, grid.z,
            block.x, block.y, block.z,
            (unsigned long long)p.sharedMem,
            (unsigned long long)p.stream
        );

        cudaReturnPacket.cudaError =
            static_cast<uint32_t>(
                cudaConfigureCall(
                    grid,
                    block,
                    static_cast<size_t>(p.sharedMem),
                    reinterpret_cast<cudaStream_t>(
                        static_cast<uintptr_t>(p.stream)
                    )
                )
            );

        cudaLaunchStreamStack.push_back(p.stream);

        break;
      }


      case CUDA_SET_ARG:
      {
        auto &p =
            cudaPacket.payload.setupArgument;

        panic_if(
            p.size > sizeof(p.value),
            "%s: CUDA_SET_ARG size=%llu exceeds 256",
            name(),
            (unsigned long long)p.size
        );

        inform(
            "%s: [BALAR-API] CUDA_SET_ARG "
            "arg=%#llx size=%llu offset=%llu",
            name(),
            (unsigned long long)p.arg,
            (unsigned long long)p.size,
            (unsigned long long)p.offset
        );

        cudaReturnPacket.cudaError =
            static_cast<uint32_t>(
                cudaSetupArgumentSST(
                    p.arg,
                    p.value,
                    static_cast<size_t>(p.size),
                    static_cast<size_t>(p.offset)
                )
            );

        break;
      }


      case CUDA_LAUNCH:
      {
        const uint64_t func =
            cudaPacket.payload.cudaLaunch.func;

        inform(
            "%s: [BALAR-API] CUDA_LAUNCH func=%#llx",
            name(),
            (unsigned long long)func
        );

        panic_if(
            cudaLaunchStreamStack.empty(),
            "%s: CUDA_LAUNCH without preceding CUDA_CONFIG_CALL",
            name()
        );

        const uint64_t stream =
            cudaLaunchStreamStack.back();

        cudaReturnPacket.cudaError =
            static_cast<uint32_t>(
                cudaLaunchSST(func)
            );

        if (cudaReturnPacket.cudaError == 0) {
            ++pendingKernelsPerStream[stream];

            inform(
                "%s: [BALAR-KERNEL] LAUNCH stream=%#llx pending=%u",
                name(),
                (unsigned long long)stream,
                pendingKernelsPerStream[stream]
            );
        }

        cudaLaunchStreamStack.pop_back();

        break;
      }


      case CUDA_MALLOC:
      {
        const uint64_t devPtr =
            cudaPacket.payload.cudaMalloc.devPtr;

        const uint64_t size =
            cudaPacket.payload.cudaMalloc.size;

        inform(
            "%s: [BALAR-API] CUDA_MALLOC "
            "devPtr=%#llx size=%llu",
            name(),
            (unsigned long long)devPtr,
            (unsigned long long)size
        );

        void **guestDevPtr =
            reinterpret_cast<void **>(
                static_cast<uintptr_t>(devPtr)
            );

        lastGpuMallocAddr =
            cudaMallocSST(
                guestDevPtr,
                static_cast<size_t>(size)
            );

        inform(
            "%s: [BALAR-GPU] "
            "cudaMallocSST returned %#llx",
            name(),
            (unsigned long long)lastGpuMallocAddr
        );

        cudaReturnPacket.payload.cudaMalloc.mallocAddr =
            lastGpuMallocAddr;

        cudaReturnPacket.payload.cudaMalloc.devPtrAddr =
            devPtr;

        break;
      }


      case CUDA_PARAM_CONFIG:
      {
        const auto &p =
            cudaPacket.payload.paramConfig;

        inform(
            "%s: [BALAR-API] CUDA_PARAM_CONFIG "
            "hostFun=%#llx index=%u",
            name(),
            (unsigned long long)p.hostFun,
            p.index
        );

        auto result =
            SST_cudaGetParamConfig(
                p.hostFun,
                p.index
            );

        cudaReturnPacket.cudaError =
            static_cast<uint32_t>(
                std::get<0>(result)
            );

        cudaReturnPacket.payload.paramConfig.size =
            static_cast<uint64_t>(
                std::get<1>(result)
            );

        cudaReturnPacket.payload.paramConfig.alignment =
            std::get<2>(result);

        inform(
            "%s: [BALAR-GPU] PARAM_CONFIG "
            "error=%u size=%llu alignment=%u",
            name(),
            cudaReturnPacket.cudaError,
            (unsigned long long)
                cudaReturnPacket.payload.paramConfig.size,
            cudaReturnPacket.payload.paramConfig.alignment
        );

        break;
      }


      default:
        warn(
            "%s: CUDA API id=%u not implemented yet",
            name(),
            (unsigned)cudaPacket.cudaCallId
        );

        dmaBusy = false;
        responseReady = false;
        return;
    }

    inform(
        "%s: [BALAR-RETURN] id=%u error=%u done=%u",
        name(),
        cudaReturnPacket.cudaCallId,
        cudaReturnPacket.cudaError,
        (unsigned)cudaReturnPacket.isDone
    );

    startCudaReturnWrite();
}

void
Gem5Balar::memcpyDataReadDone()
{
    panic_if(
        memcpyPhase != MemcpyPhase::H2DRead &&
        memcpyPhase != MemcpyPhase::D2HRead,
        "%s: unexpected memcpy data DMA-read completion phase=%u",
        name(),
        static_cast<unsigned>(memcpyPhase)
    );

    uint8_t *data =
        memcpyBuffer.data() + memcpyDataOffset;

    cudaError_t err;

    if (memcpyPhase == MemcpyPhase::H2DRead) {

        memcpyPhase = MemcpyPhase::H2DGpu;

        inform(
            "%s: [BALAR-MEMCPY] H2D host buffer -> GPU %#llx "
            "count=%llu",
            name(),
            (unsigned long long)memcpyGpuAddr,
            (unsigned long long)memcpyCount
        );

        err = cudaMemcpy(
            reinterpret_cast<void *>(
                static_cast<uintptr_t>(memcpyGpuAddr)
            ),
            data,
            memcpyCount,
            cudaMemcpyHostToDevice
        );

    } else {

        /*
         * D2H read above preserved the surrounding bytes of the
         * complete CHI lines. GPGPU-Sim now overwrites only the
         * requested sub-range.
         */
        memcpyPhase = MemcpyPhase::D2HGpu;

        inform(
            "%s: [BALAR-MEMCPY] D2H GPU %#llx -> host buffer "
            "count=%llu",
            name(),
            (unsigned long long)memcpyGpuAddr,
            (unsigned long long)memcpyCount
        );

        err = cudaMemcpy(
            data,
            reinterpret_cast<const void *>(
                static_cast<uintptr_t>(memcpyGpuAddr)
            ),
            memcpyCount,
            cudaMemcpyDeviceToHost
        );
    }

    cudaReturnPacket.cudaError =
        static_cast<uint32_t>(err);

    if (cudaReturnPacket.cudaError != 0) {
        inform(
            "%s: [BALAR-MEMCPY] GPGPU-Sim cudaMemcpy error=%u",
            name(),
            cudaReturnPacket.cudaError
        );

        finishMemcpy();
    }
}


void
Gem5Balar::memcpyH2DDone(
    uint64_t dst,
    uint64_t src,
    size_t count)
{
    panic_if(
        memcpyPhase != MemcpyPhase::H2DGpu,
        "%s: unexpected H2D callback phase=%u",
        name(),
        static_cast<unsigned>(memcpyPhase)
    );

    panic_if(
        dst != memcpyGpuAddr ||
        count != memcpyCount,
        "%s: H2D callback mismatch "
        "dst=%#llx/%#llx count=%llu/%llu",
        name(),
        (unsigned long long)dst,
        (unsigned long long)memcpyGpuAddr,
        (unsigned long long)count,
        (unsigned long long)memcpyCount
    );

    inform(
        "%s: [BALAR-MEMCPY] H2D GPU callback complete "
        "dst=%#llx count=%llu host_src=%#llx",
        name(),
        (unsigned long long)dst,
        (unsigned long long)count,
        (unsigned long long)src
    );

    finishMemcpy();
}


void
Gem5Balar::memcpyD2HDone(
    uint64_t dst,
    uint64_t src,
    size_t count)
{
    panic_if(
        memcpyPhase != MemcpyPhase::D2HGpu,
        "%s: unexpected D2H callback phase=%u",
        name(),
        static_cast<unsigned>(memcpyPhase)
    );

    panic_if(
        src != memcpyGpuAddr ||
        count != memcpyCount,
        "%s: D2H callback mismatch "
        "src=%#llx/%#llx count=%llu/%llu",
        name(),
        (unsigned long long)src,
        (unsigned long long)memcpyGpuAddr,
        (unsigned long long)count,
        (unsigned long long)memcpyCount
    );

    inform(
        "%s: [BALAR-MEMCPY] D2H GPU callback complete "
        "src=%#llx count=%llu host_dst=%#llx",
        name(),
        (unsigned long long)src,
        (unsigned long long)count,
        (unsigned long long)dst
    );

    memcpyPhase = MemcpyPhase::D2HWrite;

    inform(
        "%s: [BALAR-MEMCPY] D2H DMA WRITE "
        "aligned=%#llx bytes=%llu",
        name(),
        (unsigned long long)memcpyGuestAlignedAddr,
        (unsigned long long)memcpyAlignedBytes
    );

    dmaWrite(
        memcpyGuestAlignedAddr,
        memcpyAlignedBytes,
        &memcpyDataWriteDoneEvent,
        memcpyBuffer.data()
    );
}

void
Gem5Balar::kernelDone(uint64_t stream)
{
    auto it = pendingKernelsPerStream.find(stream);

    panic_if(
        it == pendingKernelsPerStream.end() ||
        it->second == 0,
        "%s: kernelDone for stream=%#llx with no pending kernel",
        name(),
        (unsigned long long)stream
    );

    --it->second;

    inform(
        "%s: [BALAR-KERNEL] DONE stream=%#llx pending=%u",
        name(),
        (unsigned long long)stream,
        it->second
    );

    if (it->second == 0) {
        pendingKernelsPerStream.erase(it);
    }
}

void
Gem5Balar::memcpyDataWriteDone()
{
    panic_if(
        memcpyPhase != MemcpyPhase::D2HWrite,
        "%s: unexpected memcpy data DMA-write completion phase=%u",
        name(),
        static_cast<unsigned>(memcpyPhase)
    );

    inform(
        "%s: [BALAR-MEMCPY] D2H guest-memory DMA complete",
        name()
    );

    finishMemcpy();
}


void
Gem5Balar::finishMemcpy()
{
    memcpyPhase = MemcpyPhase::Idle;

    cudaReturnPacket.isDone = 1;

    inform(
        "%s: [BALAR-MEMCPY] COMPLETE error=%u",
        name(),
        cudaReturnPacket.cudaError
    );

    /*
     * Keep capacity for the following page fragments but release the
     * logical contents. The following CUDA_MEMCPY packet will reuse it.
     */
    memcpyBuffer.clear();

    startCudaReturnWrite();
}


void
Gem5Balar::startCudaReturnWrite()
{
    /*
     * Original Balar behavior:
     * overwrite the CUDA request in the same guest scratch
     * physical address with the CUDA return packet.
     */
    inform("%s: [BALAR-DMA] starting RETURN WRITE addr=%#llx size=%llu",
           name(),
           (unsigned long long)dmaSourceAddr,
           (unsigned long long)sizeof(cudaReturnPacket));

    dmaWrite(
        dmaSourceAddr,
        sizeof(cudaReturnPacket),
        &dmaWriteDoneEvent,
        reinterpret_cast<uint8_t *>(&cudaReturnPacket)
    );
}

void
Gem5Balar::dmaWriteDone()
{
    /*
     * Only now is the response visible to the guest.
     */
    responseReady = true;
    dmaBusy = false;

    inform("%s: [BALAR-DMA] RETURN WRITE COMPLETE addr=%#llx",
           name(),
           (unsigned long long)dmaSourceAddr);

    inform("%s: [BALAR-MMIO] response ready response_phys=%#llx",
           name(),
           (unsigned long long)dmaSourceAddr);
}

void
Gem5Balar::dmaReadDone()
{
    inform("%s: DMA READ COMPLETE addr=%#llx",
           name(),
           (unsigned long long)dmaSourceAddr);

    inform("%s: CUDA PACKET id=%u isSSTmem=%u devPtr=%#llx size=%#llx",
           name(),
           (unsigned)cudaPacket.cudaCallId,
           (unsigned)cudaPacket.isSstMem,
           (unsigned long long)cudaPacket.payload.cudaMalloc.devPtr,
           (unsigned long long)cudaPacket.payload.cudaMalloc.size);

    /*
     * dispatchCudaCall() starts the return DMA write.
     * dmaBusy remains asserted until dmaWriteDone().
     */
    dispatchCudaCall();
}

bool
Gem5Balar::gpuMemBufferFull(
    unsigned clusterId) const
{
    panic_if(
        clusterId >= gpuMemOutstandingPerCluster.size(),
        "%s: invalid GPU cluster id %u (clusters=%llu)",
        name(),
        clusterId,
        (unsigned long long)gpuMemOutstandingPerCluster.size()
    );

    uint64_t totalOutstanding = 0;

    for (const auto outstanding : gpuMemOutstandingPerCluster) {
        totalOutstanding += outstanding;
    }

    /*
     * Original SST Balar:
     *   one gpu_to_cache link per GPU cluster.
     *
     * Current M3.7 Gem5Balar:
     *   all GPU clusters share one gpu_mem RequestPort / CNM_RNI_DMA.
     *
     * Preserve the original per-cluster limit, but also bound the
     * aggregate pressure injected into the shared gem5 timing port.
     */
    return
        gpuMemOutstandingPerCluster[clusterId]
            >= maxPendingCacheTrans ||
        totalOutstanding
            >= maxPendingCacheTrans;
}

void
Gem5Balar::sendGpuMemRead(
    unsigned clusterId,
    uint64_t address,
    size_t size,
    void *memFetch)
{
    enqueueGpuMemRequest(
        clusterId,
        static_cast<Addr>(address),
        size,
        memFetch,
        false
    );
}


void
Gem5Balar::sendGpuMemWrite(
    unsigned clusterId,
    uint64_t address,
    size_t size,
    void *memFetch)
{
    /*
     * M3.7 validation policy:
     *
     * GPGPU-Sim already owns the functional GPU-memory contents.
     * For now we send a read transaction through gem5/CHI also for
     * stores so that the external system models timing without writing
     * dummy data into Linux-visible physical RAM.
     *
     * After M3.7 validates the round trip, this becomes a WriteReq
     * against a GPU-reserved/shadow timing range.
     */
    enqueueGpuMemRequest(
        clusterId,
        static_cast<Addr>(address),
        size,
        memFetch,
        true
    );
}


void
Gem5Balar::enqueueGpuMemRequest(
    unsigned clusterId,
    Addr address,
    size_t size,
    void *memFetch,
    bool originalWrite)
{
    panic_if(
        clusterId >= gpuCoreCount,
        "%s: mem_fetch from invalid cluster=%u clusters=%u",
        name(),
        clusterId,
        gpuCoreCount
    );

    panic_if(
        memFetch == nullptr,
        "%s: null mem_fetch from cluster=%u",
        name(),
        clusterId
    );

    panic_if(
        size == 0,
        "%s: zero-size mem_fetch cluster=%u addr=%#llx",
        name(),
        clusterId,
        (unsigned long long)address
    );

    panic_if(
        gpuMemBufferFull(clusterId),
        "%s: GPGPU-Sim issued mem_fetch while cluster %u is full "
        "(outstanding=%u max=%u)",
        name(),
        clusterId,
        gpuMemOutstandingPerCluster[clusterId],
        maxPendingCacheTrans
    );

    auto *transaction =
        new GpuMemTransaction;

    transaction->id = ++gpuMemTransactionSeq;
    transaction->issueTick = static_cast<uint64_t>(curTick());
    transaction->issueGpuCycle = gpuTickCount;

    transaction->clusterId = clusterId;
    transaction->memFetch = memFetch;
    transaction->address = address;
    transaction->size = size;
    transaction->originalWrite = originalWrite;

    /*
     * Count the whole mem_fetch as outstanding as soon as GPGPU-Sim
     * hands ownership of its timing request to Gem5Balar.
     */
    ++gpuMemOutstandingPerCluster[clusterId];
    ++gpuMemFetchIssued;

    /*
     * Ruby/CHI works with 64-byte cache lines in the current machine.
     * Split a single mem_fetch if it crosses cache-line boundaries.
     */
   const size_t lineSize = cacheBlockSize();

    /*
    * M3.7 timing-only GPU aperture.
    *
    * GPGPU-Sim addresses are GPU addresses, not gem5 system physical
    * addresses. Map them into a valid 1 GiB window of guest RAM only
    * for CHI/Ruby timing transactions.
    *
    * Guest RAM:
    *   [0x80000000, 0x180000000)
    *
    * GPU timing aperture:
    *   [0x100000000, 0x140000000)
    */
    constexpr Addr gpuTimingBase = 0x100000000ULL;
    constexpr Addr gpuTimingSize = 0x40000000ULL;

    const Addr mappedStartAddr =
        gpuTimingBase + (address & (gpuTimingSize - 1));

    size_t completed = 0;

    while (completed < size) {

        const Addr gpuPartAddr =
            address + completed;

        const Addr partAddr =
            gpuTimingBase +
            (gpuPartAddr & (gpuTimingSize - 1));

        const size_t lineOffset =
            partAddr % lineSize;

        const size_t bytesThisPacket =
            std::min(
                size - completed,
                lineSize - lineOffset
            );

        Request::Flags flags;
        flags.set(Request::READ_ONCE);

        RequestPtr req =
            std::make_shared<Request>(
                partAddr,
                bytesThisPacket,
                flags,
                gpuMemRequestorId
            );

        /*
         * M3.7:
         * Use ReadReq for both original loads and stores.
         *
         * This gives us a real CHI/Ruby timing transaction but avoids
         * modifying guest RAM with dummy store data.
         */
        PacketPtr pkt =
            new Packet(
                req,
                MemCmd::ReadReq
            );

        pkt->allocate();

        pkt->senderState =
            new GpuMemSenderState(transaction);

        ++transaction->remainingParts;

        gpuMemSendQueue.push_back(pkt);

        completed += bytesThisPacket;
    }

    if (gpuMemFetchIssued <= 64 ||
        (gpuMemFetchIssued % 10000ULL) == 0) {

        inform(
            "%s: [BALAR-GPU-MEM] FETCH ISSUE #%llu "
            "tx=%llu tick=%llu gpu_cycle=%llu "
            "cluster=%u op=%s gpu_addr=%#llx mapped=%#llx "
            "size=%llu parts=%u outstanding=%u",
            name(),
            (unsigned long long)gpuMemFetchIssued,
            (unsigned long long)transaction->id,
            (unsigned long long)transaction->issueTick,
            (unsigned long long)transaction->issueGpuCycle,
            clusterId,
            originalWrite ? "WRITE->TIMING_READ" : "READ",
            (unsigned long long)address,
            (unsigned long long)mappedStartAddr,
            (unsigned long long)size,
            transaction->remainingParts,
            gpuMemOutstandingPerCluster[clusterId]
        );
    }

    trySendGpuMemRequests();
}


void
Gem5Balar::trySendGpuMemRequests()
{
    if (gpuMemWaitingForRetry)
        return;

    while (!gpuMemSendQueue.empty()) {

        PacketPtr pkt =
            gpuMemSendQueue.front();

        if (!gpuMemPort.sendPacket(pkt)) {

            gpuMemWaitingForRetry = true;

            ++gpuMemBlockedCount;

            auto *state =
                static_cast<GpuMemSenderState *>(pkt->senderState);

            panic_if(
                state == nullptr ||
                state->transaction == nullptr,
                "%s: blocked gpu packet without transaction",
                name()
            );

            if (gpuMemBlockedCount <= 64 ||
                (gpuMemBlockedCount % 10000ULL) == 0) {

                inform(
                    "%s: [BALAR-GPU-MEM] PORT BLOCKED #%llu "
                    "tx=%llu tick=%llu gpu_cycle=%llu queue=%llu",
                    name(),
                    (unsigned long long)gpuMemBlockedCount,
                    (unsigned long long)state->transaction->id,
                    (unsigned long long)curTick(),
                    (unsigned long long)gpuTickCount,
                    (unsigned long long)gpuMemSendQueue.size()
                );
            }

            return;
        }

        gpuMemSendQueue.pop_front();

        ++gpuMemPacketsIssued;
    }
}


void
Gem5Balar::recvGpuMemReqRetry()
{
    panic_if(
        !gpuMemWaitingForRetry,
        "%s: unexpected gpu_mem recvReqRetry()",
        name()
    );

    gpuMemWaitingForRetry = false;

    ++gpuMemRetryCount;

    if (gpuMemRetryCount <= 64 ||
        (gpuMemRetryCount % 10000ULL) == 0) {

        inform(
            "%s: [BALAR-GPU-MEM] PORT RETRY #%llu "
            "tick=%llu gpu_cycle=%llu queue=%llu",
            name(),
            (unsigned long long)gpuMemRetryCount,
            (unsigned long long)curTick(),
            (unsigned long long)gpuTickCount,
            (unsigned long long)gpuMemSendQueue.size()
        );
    }

    trySendGpuMemRequests();
}


bool
Gem5Balar::recvGpuMemTimingResp(PacketPtr pkt)
{
    panic_if(
        pkt == nullptr,
        "%s: null gpu_mem response",
        name()
    );

    panic_if(
        !pkt->isResponse(),
        "%s: gpu_mem received packet that is not a response",
        name()
    );

    auto *senderState =
        static_cast<GpuMemSenderState *>(
            pkt->senderState
        );

    panic_if(
        senderState == nullptr ||
        senderState->transaction == nullptr,
        "%s: gpu_mem response without transaction state",
        name()
    );

    GpuMemTransaction *transaction =
        senderState->transaction;

    const unsigned clusterId =
        transaction->clusterId;

    delete senderState;
    pkt->senderState = nullptr;

    delete pkt;

    panic_if(
        transaction->remainingParts == 0,
        "%s: gpu_mem transaction already completed",
        name()
    );

    --transaction->remainingParts;

    /*
     * Do not wake GPGPU-Sim until every gem5 packet belonging to this
     * single mem_fetch has returned.
     */
    if (transaction->remainingParts == 0) {

        panic_if(
            gpuMemOutstandingPerCluster[clusterId] == 0,
            "%s: cluster %u outstanding underflow",
            name(),
            clusterId
        );
        const uint64_t completionTick =
        static_cast<uint64_t>(curTick());

        const uint64_t completionGpuCycle =
            gpuTickCount;

        const uint64_t latencyTicks =
            completionTick - transaction->issueTick;

        const uint64_t elapsedGpuCycles =
            completionGpuCycle - transaction->issueGpuCycle;

        --gpuMemOutstandingPerCluster[clusterId];

        ++gpuMemFetchCompleted;

        if (gpuMemFetchCompleted <= 64 ||
            (gpuMemFetchCompleted % 10000ULL) == 0) {

            inform(
                "%s: [BALAR-GPU-MEM] FETCH COMPLETE #%llu "
                "tx=%llu cluster=%u op=%s "
                "issue_tick=%llu complete_tick=%llu latency_ticks=%llu "
                "issue_cycle=%llu complete_cycle=%llu elapsed_gpu_cycles=%llu "
                "addr=%#llx size=%llu outstanding=%u",
                name(),
                (unsigned long long)gpuMemFetchCompleted,
                (unsigned long long)transaction->id,
                clusterId,
                transaction->originalWrite ?
                    "WRITE->TIMING_READ" : "READ",
                (unsigned long long)transaction->issueTick,
                (unsigned long long)completionTick,
                (unsigned long long)latencyTicks,
                (unsigned long long)transaction->issueGpuCycle,
                (unsigned long long)completionGpuCycle,
                (unsigned long long)elapsedGpuCycles,
                (unsigned long long)transaction->address,
                (unsigned long long)transaction->size,
                gpuMemOutstandingPerCluster[clusterId]
            );
        }

        /*
         * This is the original GPGPU-Sim/SST return path.
         * GPGPU-Sim owns memFetch; gem5 must never delete it.
         */
        void *memFetch =
            transaction->memFetch;

        delete transaction;

        SST_receive_mem_reply(
            clusterId,
            memFetch
        );
    }

    /*
     * A returned response may free resources downstream.
     */
    trySendGpuMemRequests();

    return true;
}

} // namespace gem5
