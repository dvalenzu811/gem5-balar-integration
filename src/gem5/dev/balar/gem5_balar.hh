#ifndef __DEV_BALAR_GEM5_BALAR_HH__
#define __DEV_BALAR_GEM5_BALAR_HH__

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <deque>
#include <memory>

#include "dev/dma_device.hh"
#include "params/Gem5Balar.hh"
#include "sim/eventq.hh"
#include "mem/port.hh"
#include "mem/request.hh"

namespace gem5
{

class Gem5Balar : public DmaDevice
{
  public:
    using Params = Gem5BalarParams;
    Gem5Balar(const Params &p);

    AddrRangeList getAddrRanges() const override;
    Tick read(PacketPtr pkt) override;
    Tick write(PacketPtr pkt) override;

    Port &getPort(
        const std::string &if_name,
        PortID idx = InvalidPortID
    ) override;

    /*
    * GPGPU-Sim/SST memory hooks.
    *
    * These are called by the strong global wrappers implemented in
    * gem5_balar.cc. The mem_fetch pointer is opaque to gem5.
    */
    bool gpuMemBufferFull(unsigned clusterId) const;

    void sendGpuMemRead(
        unsigned clusterId,
        uint64_t address,
        size_t size,
        void *memFetch
    );

    void sendGpuMemWrite(
        unsigned clusterId,
        uint64_t address,
        size_t size,
        void *memFetch
    );

    // Strong GPGPU-Sim memcpy callback targets.
    void memcpyH2DDone(uint64_t dst, uint64_t src, size_t count);
    void memcpyD2HDone(uint64_t dst, uint64_t src, size_t count);

  private:

    static constexpr size_t BalarRequestBytes = 576;

    struct BalarCudaCallWire
    {
        uint32_t cudaCallId;       // +0
        uint8_t isSstMem;          // +4
        uint8_t abiPadding[3];     // +5..7

        union
        {
            struct
            {
                uint64_t devPtr;   // +8
                uint64_t size;     // +16
            } cudaMalloc;

            struct
            {
                char fileName[256];    // +8
            } registerFatbin;

            struct
            {
                uint64_t fatCubinHandle;   // +8
                uint64_t hostFun;           // +16
                char deviceFun[512];        // +24
            } registerFunction;

            struct
            {
                uint64_t dst;           // +8
                uint64_t src;           // +16
                uint64_t count;         // +24
                uint64_t payload;       // +32
                uint64_t dstBuf;        // +40
                uint64_t srcBuf;        // +48
                uint32_t kind;          // +56
                uint32_t padding;       // +60
            } cudaMemcpy;

            struct
            {
                uint64_t sharedMem;     // +8
                uint64_t stream;        // +16

                uint32_t gdx;           // +24
                uint32_t gdy;           // +28
                uint32_t gdz;           // +32

                uint32_t bdx;           // +36
                uint32_t bdy;           // +40
                uint32_t bdz;           // +44
            } configureCall;

            struct
            {
                uint64_t arg;           // +8
                uint64_t size;          // +16
                uint64_t offset;        // +24
                uint8_t value[256];     // +32
            } setupArgument;

            struct
            {
                uint64_t func;          // +8
            } cudaLaunch;

            struct
            {
                uint64_t hostFun;       // +8
                uint32_t index;         // +16
                uint32_t padding;
            } paramConfig;

            /*
             * Real Balar packet = 536 B.
             * We DMA 576 B so every CHI transfer covers complete 64 B lines.
             */
            uint8_t raw[BalarRequestBytes - 8];

        } payload;
    };

    static_assert(sizeof(BalarCudaCallWire) == 576);

    static_assert(offsetof(BalarCudaCallWire, cudaCallId) == 0);
    static_assert(offsetof(BalarCudaCallWire, isSstMem) == 4);

    static_assert(
        offsetof(BalarCudaCallWire, payload.cudaMalloc.devPtr) == 8);
    static_assert(
        offsetof(BalarCudaCallWire, payload.cudaMalloc.size) == 16);

    static_assert(
        offsetof(BalarCudaCallWire, payload.registerFatbin.fileName) == 8);

    static_assert(
        offsetof(BalarCudaCallWire,
                 payload.registerFunction.fatCubinHandle) == 8);
    static_assert(
        offsetof(BalarCudaCallWire,
                 payload.registerFunction.hostFun) == 16);
    static_assert(
        offsetof(BalarCudaCallWire,
                 payload.registerFunction.deviceFun) == 24);

    static_assert(
        offsetof(BalarCudaCallWire,
                 payload.cudaMemcpy.dst) == 8);
    static_assert(
        offsetof(BalarCudaCallWire,
                 payload.cudaMemcpy.src) == 16);
    static_assert(
        offsetof(BalarCudaCallWire,
                 payload.cudaMemcpy.count) == 24);
    static_assert(
        offsetof(BalarCudaCallWire,
                 payload.cudaMemcpy.payload) == 32);
    static_assert(
        offsetof(BalarCudaCallWire,
                 payload.cudaMemcpy.dstBuf) == 40);
    static_assert(
        offsetof(BalarCudaCallWire,
                 payload.cudaMemcpy.srcBuf) == 48);
    static_assert(
        offsetof(BalarCudaCallWire,
                 payload.cudaMemcpy.kind) == 56);

    static_assert(
        offsetof(BalarCudaCallWire,
                 payload.configureCall.sharedMem) == 8);
    static_assert(
        offsetof(BalarCudaCallWire,
                 payload.configureCall.stream) == 16);
    static_assert(
        offsetof(BalarCudaCallWire,
                 payload.configureCall.gdx) == 24);
    static_assert(
        offsetof(BalarCudaCallWire,
                 payload.configureCall.bdx) == 36);

    static_assert(
        offsetof(BalarCudaCallWire,
                 payload.setupArgument.arg) == 8);
    static_assert(
        offsetof(BalarCudaCallWire,
                 payload.setupArgument.size) == 16);
    static_assert(
        offsetof(BalarCudaCallWire,
                 payload.setupArgument.offset) == 24);
    static_assert(
        offsetof(BalarCudaCallWire,
                 payload.setupArgument.value) == 32);

    static_assert(
        offsetof(BalarCudaCallWire,
                 payload.cudaLaunch.func) == 8);

    static_assert(
        offsetof(BalarCudaCallWire,
                 payload.paramConfig.hostFun) == 8);
    static_assert(
        offsetof(BalarCudaCallWire,
                 payload.paramConfig.index) == 16);


    struct BalarCudaReturnWire
    {
        uint32_t cudaCallId;       // +0
        uint32_t cudaError;        // +4
        uint8_t isDone;            // +8
        uint8_t abiPadding[7];     // union starts +16

        union
        {
            uint64_t fatCubinHandle;

            struct
            {
                uint64_t mallocAddr;
                uint64_t devPtrAddr;
            } cudaMalloc;

            struct
            {
                uint64_t size;
                uint32_t alignment;
                uint32_t padding;
            } paramConfig;

            uint8_t raw[48];

        } payload;
    };

    static_assert(sizeof(BalarCudaReturnWire) == 64);

    static_assert(offsetof(BalarCudaReturnWire, cudaCallId) == 0);
    static_assert(offsetof(BalarCudaReturnWire, cudaError) == 4);
    static_assert(offsetof(BalarCudaReturnWire, isDone) == 8);

    static_assert(
        offsetof(BalarCudaReturnWire,
                 payload.fatCubinHandle) == 16);

    static_assert(
        offsetof(BalarCudaReturnWire,
                 payload.cudaMalloc.mallocAddr) == 16);
    static_assert(
        offsetof(BalarCudaReturnWire,
                 payload.cudaMalloc.devPtrAddr) == 24);

    static_assert(
        offsetof(BalarCudaReturnWire,
                 payload.paramConfig.size) == 16);
    static_assert(
        offsetof(BalarCudaReturnWire,
                 payload.paramConfig.alignment) == 24);

        /*
     * M3.7: GPGPU-Sim mem_fetch -> gem5 timing-memory bridge.
     *
     * One GpuMemPort is shared by all GPGPU-Sim clusters for the
     * first validation milestone.
     */
    class GpuMemPort : public RequestPort
    {
      public:
        GpuMemPort(
            const std::string &name,
            Gem5Balar &owner
        );

        bool sendPacket(PacketPtr pkt)
        {
            return sendTimingReq(pkt);
        }

      protected:
        bool recvTimingResp(PacketPtr pkt) override;
        void recvReqRetry() override;

      private:
        Gem5Balar &owner;
    };


    struct GpuMemTransaction
    {
        unsigned clusterId = 0;
        void *memFetch = nullptr;

        Addr address = 0;
        size_t size = 0;

        bool originalWrite = false;

        /*
         * One mem_fetch may have to be split at gem5 cache-line
         * boundaries. Reply to GPGPU-Sim only after every part returns.
         */
        unsigned remainingParts = 0;
    };


    struct GpuMemSenderState : public Packet::SenderState
    {
        explicit GpuMemSenderState(GpuMemTransaction *t)
            : transaction(t)
        {}

        GpuMemTransaction *transaction;
    };


    void enqueueGpuMemRequest(
        unsigned clusterId,
        Addr address,
        size_t size,
        void *memFetch,
        bool originalWrite
    );

    void trySendGpuMemRequests();

    bool recvGpuMemTimingResp(PacketPtr pkt);
    void recvGpuMemReqRetry();

    void dmaReadDone();

    void dmaWriteDone();
    void dispatchCudaCall();
    void startCudaReturnWrite();

    // M3.6c: synchronous Balar cudaMemcpy state machine.
    void memcpyDataReadDone();
    void memcpyDataWriteDone();
    void finishMemcpy();

    // M3.5: gem5 EventQueue replacement for the SST Balar clock.
    void startGpuClock();
    void gpuTick();

    Addr pioAddr;
    Addr pioSize;
    Tick pioDelay;
    
        GpuMemPort gpuMemPort;

    RequestorID gpuMemRequestorId = 0;

    unsigned maxPendingCacheTrans = 512;

    std::vector<unsigned> gpuMemOutstandingPerCluster;

    std::deque<PacketPtr> gpuMemSendQueue;

    bool gpuMemWaitingForRetry = false;

    uint64_t gpuMemFetchIssued = 0;
    uint64_t gpuMemFetchCompleted = 0;
    uint64_t gpuMemPacketsIssued = 0;
    
    Addr dmaSourceAddr = 0;
    bool dmaBusy = false;
    bool responseReady = false;

    BalarCudaCallWire cudaPacket{};
    BalarCudaReturnWire cudaReturnPacket{};

    // Balar <-> GPGPU-Sim state.
    unsigned gpuCoreCount = 0;
    bool launchBlocking = false;
    uint64_t lastGpuMallocAddr = 0;

    std::string cudaExecutable;

    enum class MemcpyPhase : uint8_t
    {
        Idle,
        H2DRead,
        H2DGpu,
        D2HRead,
        D2HGpu,
        D2HWrite
    };

    MemcpyPhase memcpyPhase = MemcpyPhase::Idle;

    // Guest physical range. DMA accesses are expanded to complete
    // 64-byte CHI cache lines.
    Addr memcpyGuestAlignedAddr = 0;
    size_t memcpyAlignedBytes = 0;
    size_t memcpyDataOffset = 0;

    uint64_t memcpyGpuAddr = 0;
    size_t memcpyCount = 0;

    std::vector<uint8_t> memcpyBuffer;

    // M3.5 GPU clock state.
    Tick gpuClockPeriod = 0;
    bool gpuClockRunning = false;
    uint64_t gpuTickCount = 0;

    EventFunctionWrapper dmaReadDoneEvent;
    EventFunctionWrapper dmaWriteDoneEvent;
    EventFunctionWrapper memcpyDataReadDoneEvent;
    EventFunctionWrapper memcpyDataWriteDoneEvent;
    EventFunctionWrapper gpuTickEvent;
};

} // namespace gem5

#endif // __DEV_BALAR_GEM5_BALAR_HH__
