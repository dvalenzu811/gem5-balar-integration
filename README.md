# gem5-Balar Integration

`Gem5Balar` is a gem5 device that bridges CUDA/GPGPU-Sim activity into a gem5 full-system simulation.

The project was developed to replace the original SST-side Balar integration with a gem5-native component while keeping the existing GPGPU-Sim/SST callback ABI whenever possible.

The repository serves two purposes:

1. **Reusable Gem5Balar component** — integrate Balar and GPGPU-Sim into another gem5 system.
2. **Reference integration** — reproduce the configuration used during development with gem5, a RISC-V CNM board, CHI, GPGPU-Sim and a CUDA `vecadd` application.

---

# 1. Architecture

At a high level:

```text
                    Linux guest
                        │
                  CUDA application
                        │
                 CUDA runtime calls
                        │
                        ▼
                 ┌─────────────┐
                 │ Gem5Balar   │
                 └─────────────┘
                  │     │     │
                  │     │     │
                PIO    DMA  gpu_mem
                  │     │     │
                  │     │     ▼
                  │     │   gem5 memory system
                  │     │      │
                  │     │      ▼
                  │     │     CHI
                  │     │
                  │     ▼
                  │ CPU memory
                  │
                  ▼
            CPU control path

                 Gem5Balar
                      │
                      ▼
                GPGPU-Sim
                      │
                      ▼
                  GPU model
```

`Gem5Balar` therefore handles two different paths:

```text
Control path
============

Guest CUDA call
      │
      ▼
     MMIO
      │
      ▼
 Gem5Balar
      │
      ├── DMA request/return structures
      │
      ▼
 GPGPU-Sim


GPU memory timing path
======================

GPGPU-Sim mem_fetch
      │
      ▼
send_read_request_SST()
send_write_request_SST()
      │
      ▼
 Gem5Balar
      │
      ▼
 gpu_mem RequestPort
      │
      ▼
 gem5 memory hierarchy
      │
      ▼
 timing response
      │
      ▼
 Gem5Balar
      │
      ▼
SST_receive_mem_reply()
      │
      ▼
 GPGPU-Sim
```

---

# 2. Gem5Balar as a reusable gem5 component

The main component is located under:

```text
src/gem5/dev/balar/
├── Gem5Balar.py
├── gem5_balar.hh
├── gem5_balar.cc
└── balar_gpu_api.hh
```

`Gem5Balar` derives from gem5's `DmaDevice`.

The component exposes three relevant interfaces.

## PIO

The PIO interface is used by the guest to submit CUDA/Balar commands.

Example configuration:

```text
pio_addr = configured MMIO base
pio_size = 0x1000
pio_latency = 10 ns
```

The CPU-side software writes CUDA call information through this region and Gem5Balar dispatches the corresponding operation.

---

## DMA

The standard `DmaDevice` DMA interface is used for CPU-side memory transfers associated with the Balar control protocol.

Examples include:

```text
CUDA call packet retrieval
ReturnPacket writes
H2D/D2H control-side transfers
```

This path should remain independent from GPU `mem_fetch` timing traffic.

---

## gpu_mem

`gpu_mem` is a separate gem5 `RequestPort` introduced for GPU memory timing traffic.

```text
GPGPU-Sim
    │
    ▼
 mem_fetch
    │
    ▼
Gem5Balar.gpu_mem
    │
    ▼
gem5 memory hierarchy
```

The intention is to prevent GPU memory traffic from sharing the control DMA path.

A gem5 system integrating Gem5Balar therefore needs to connect both:

```text
balar.dma
balar.gpu_mem
```

to suitable memory-side requesters.

---

# 3. GPGPU-Sim interface

Gem5Balar reuses the callback interface originally used by the SST integration.

This minimizes invasive changes inside GPGPU-Sim.

Relevant hooks include:

```cpp
bool is_SST_buffer_full(unsigned cluster_id);

void send_read_request_SST(
    unsigned cluster_id,
    uint64_t address,
    size_t size,
    void *mem_fetch);

void send_write_request_SST(
    unsigned cluster_id,
    uint64_t address,
    size_t size,
    void *mem_fetch);
```

The response path uses:

```cpp
SST_receive_mem_reply(
    cluster_id,
    mem_fetch);
```

The names still contain `SST` for compatibility with the existing GPGPU-Sim integration API.

In the current implementation these functions are provided by Gem5Balar rather than by SST.

Conceptually:

```text
Original architecture

GPGPU-Sim
    │
    ▼
SST callback API
    │
    ▼
SST Balar
    │
    ▼
SST memory system


Current architecture

GPGPU-Sim
    │
    ▼
same callback API
    │
    ▼
Gem5Balar
    │
    ▼
gem5 memory system
```

---

# 4. Integrating Gem5Balar into another gem5 system

A new gem5 integration needs to provide at least:

```text
1. Gem5Balar SimObject
2. CPU-visible PIO connection
3. control DMA connection
4. gpu_mem timing connection
5. GPGPU-Sim library
6. GPU clock/event driving
7. compatible CUDA/Balar guest runtime
```

The exact memory topology is not fixed by Gem5Balar.

For example, an implementation could use:

```text
one shared gpu_mem requester
```

or:

```text
multiple GPU requesters
```

or potentially:

```text
one requester per GPU cluster
```

depending on the target gem5 memory architecture.

The current reference integration uses a single shared `gpu_mem` RequestPort.

---

# 5. Reference integration used in this project

The development configuration uses:

```text
gem5 Full-System
        +
RISC-V Linux guest
        +
RISC-V CNM board
        +
Ruby / CHI memory hierarchy
        +
Gem5Balar
        +
GPGPU-Sim
```

The relevant memory topology is currently:

```text
Gem5Balar.dma
      │
      ▼
CNM_RNI_DMA #0
      │
      ▼
RubySequencer #0
      │
      ▼
CHI


Gem5Balar.gpu_mem
      │
      ▼
CNM_RNI_DMA #1
      │
      ▼
RubySequencer #1
      │
      ▼
CHI
```

This separates:

```text
control DMA traffic
```

from:

```text
GPU mem_fetch timing traffic
```

---

# 6. GPU execution model

GPGPU-Sim remains responsible for GPU execution.

Gem5Balar periodically advances the GPU simulator using the gem5 event queue.

Conceptually:

```text
gem5 EventQueue
      │
      ▼
GPU clock event
      │
      ▼
SST_gpu_core_cycle()
      │
      ▼
GPGPU-Sim
```

The current reference configuration uses:

```text
gpu_clock_period = 1 ns
```

This value is part of the current experimental setup and should not be considered a universal GPU clock configuration.

---

# 7. GPU memory model

The current integration separates:

```text
functional GPU memory
```

from:

```text
gem5 timing traffic
```

GPGPU-Sim remains the functional owner of GPU memory.

Gem5Balar generates gem5 transactions so that GPU memory requests experience the timing behavior of the gem5 memory hierarchy.

Therefore, at the current milestone:

```text
GPGPU-Sim
    └── functional GPU memory semantics

gem5
    └── memory-system timing
```

This is an important design choice and may evolve in future versions.

---

# 8. mem_fetch flow

A GPU request currently follows this path:

```text
GPGPU-Sim cluster
       │
       ▼
    mem_fetch
       │
       ▼
send_read_request_SST()
or
send_write_request_SST()
       │
       ▼
Gem5Balar::sendGpuMemRead/Write()
       │
       ▼
enqueueGpuMemRequest()
       │
       ▼
split request if necessary
       │
       ▼
gpu_mem RequestPort
       │
       ▼
CNM RNI
       │
       ▼
CHI
       │
       ▼
timing response
       │
       ▼
GpuMemPort::recvTimingResp()
       │
       ▼
Gem5Balar::recvGpuMemTimingResp()
       │
       ▼
SST_receive_mem_reply()
       │
       ▼
GPGPU-Sim
```

Requests crossing cache-line boundaries are split into multiple gem5 packets.

Gem5Balar tracks all parts belonging to the same GPGPU-Sim `mem_fetch`.

GPGPU-Sim is notified only after every part of the original request has completed.

---

# 9. Address mapping

GPGPU-Sim GPU addresses are not necessarily valid guest physical addresses.

Directly injecting raw GPU addresses into the current gem5/Ruby memory hierarchy caused invalid address routing.

The current implementation therefore maps GPU timing requests into a dedicated guest-physical aperture.

Reference configuration:

```text
base = 0x100000000
size = 0x40000000
```

Mapping:

```text
mapped_address =
    gpu_timing_base +
    (gpu_address & (gpu_timing_size - 1))
```

For example:

```text
GPU address:
0x1f8941e0

Timing address:
0x11f8941e0
```

The original GPU address is still retained by the transaction metadata.

This mapping is currently a **timing-model mechanism**, not a final GPU virtual/physical memory architecture.

---

# 10. Partial cache-line requests

GPU `mem_fetch` requests may be smaller than a gem5 cache line.

For the current CHI implementation, partial ordinary load requests reached a controller path requiring a complete cache-line data block.

The current integration therefore marks these timing requests as:

```cpp
Request::READ_ONCE
```

which selects the `LoadReadOnce` path in the specific gem5 CHI implementation used by this project.

This decision is specific to the current controller behavior and should not be interpreted as a general CHI requirement.

---

# 11. Write timing

At the current development milestone, GPU writes are represented in gem5 as timing reads:

```text
GPGPU WRITE
      │
      ▼
TIMING READ in gem5
```

GPGPU-Sim continues to provide the functional memory semantics.

This avoids modifying guest physical memory while the GPU memory ownership model is still being developed.

A future implementation may introduce:

```text
reserved GPU physical memory
translated GPU memory
shadow GPU memory
or a dedicated GPU memory controller
```

to support true gem5-side write semantics.

---

# 12. Backpressure

The original SST Balar architecture provided a memory interface for each GPU cluster.

Conceptually:

```text
cluster 0 -> gpu_to_cache_links[0]
cluster 1 -> gpu_to_cache_links[1]
...
cluster N -> gpu_to_cache_links[N]
```

The current Gem5Balar reference implementation instead multiplexes GPU clusters into a single shared:

```text
gpu_mem RequestPort
```

This creates different backpressure requirements.

Per-cluster outstanding request tracking is preserved, while aggregate pressure on the shared gem5 port is currently under evaluation.

This is an active M3.7 design area.

Possible future designs include:

```text
single shared requester
multiple requesters shared by groups of clusters
one requester per GPU cluster
GPU-specific Ruby requester/coalescer
```

---

# 13. Repository structure

```text
gem5-balar-integration/
├── README.md
│
├── src/
│   └── gem5/
│       └── dev/
│           └── balar/
│               ├── Gem5Balar.py
│               ├── gem5_balar.hh
│               ├── gem5_balar.cc
│               └── balar_gpu_api.hh
│
├── patches/
│   ├── gem5-infra/
│   ├── gpgpu-sim/
│   └── sst-elements/
│
├── configs/
│   └── gpgpusim.config
│
├── scripts/
│   ├── env_balar.sh
│   ├── m37_build.slurm
│   └── m37_vecadd.slurm
│
├── tests/
│   └── vecadd/
│
├── docs/
│   ├── architecture/
│   ├── adr/
│   └── milestones/
│
└── evidence/
    └── versions.txt
```

The repository does **not** contain complete copies of gem5, GPGPU-Sim or gem5-infra.

Instead, it stores:

```text
Gem5Balar sources
configuration
integration patches
scripts
documentation
validation evidence
```

---

# 14. Reference environment

The current development setup uses:

```text
gem5
GPGPU-Sim
gem5-infra
RISC-V Full-System Linux
Ruby / CHI
Slurm
```

The versions and commits used for validated experiments should be recorded in:

```text
evidence/versions.txt
```

This file should be treated as the authoritative reference for reproducing a specific experiment.

---

# 15. Building the reference integration

The reference environment expects:

```text
WORK=/gpfs/scratch/bsc18/$USER/gem5-balar

$WORK/gem5
$WORK/gpgpu-sim
$WORK/gem5-balar-integration
```

Gem5Balar is compiled as part of gem5.

The provided reference build script is:

```bash
sbatch scripts/m37_build.slurm
```

During development, incremental builds are preferred after modifying Gem5Balar.

Typical targets are:

```bash
scons --no-compress-debug -j2 \
    build/BSC/dev/balar/gem5_balar.o

scons --no-compress-debug -j2 \
    build/BSC/gem5.opt
```

A full clean rebuild should not normally be necessary for Gem5Balar source changes.

---

# 16. Running the reference vecadd experiment

The reference integration uses a CUDA vector-add application running inside the RISC-V guest.

The experiment can be launched with:

```bash
sbatch scripts/m37_vecadd.slurm
```

The application exercises:

```text
CUDA registration
H2D transfer
kernel launch
GPU execution
GPU mem_fetch generation
D2H transfer
```

The test is intended as an end-to-end integration test rather than a GPU performance benchmark.

---

# 17. Validation markers

Useful Gem5Balar runtime markers include:

```text
[BALAR-API]
[BALAR-DMA]
[BALAR-GPU-CLOCK]
[BALAR-GPU-MEM]
```

For the GPU memory path, the intended round trip is:

```text
FETCH ISSUE
      │
      ▼
PACKET ACCEPT
      │
      ▼
gem5 / CHI
      │
      ▼
FETCH COMPLETE
      │
      ▼
GPGPU REPLY
```

These markers make it possible to identify whether a request:

```text
was produced by GPGPU-Sim
was accepted by gem5
returned through the memory hierarchy
was delivered back to GPGPU-Sim
```

---

# 18. Current status

## M3.6 — CUDA control path

Validated.

The following path has been demonstrated with a real CUDA `vecadd` workload:

```text
guest CUDA application
        │
        ▼
Gem5Balar
        │
        ▼
GPGPU-Sim
        │
        ▼
CUDA kernel registration / launch
```

The control-side DMA and callback infrastructure is operational.

---

## M3.7 — GPU memory timing path

In progress.

Implemented:

```text
GPGPU-Sim mem_fetch hooks
dedicated Gem5Balar gpu_mem RequestPort
separate CHI requester
cache-line splitting
transaction aggregation
timing responses
gem5 RequestPort retry handling
GPU timing address aperture
partial-request READ_ONCE handling
```

Currently under validation:

```text
shared-port backpressure
complete mem_fetch response round trip
sustained memory traffic behavior
fairness between GPU clusters
```

M3.7 should not be considered complete until the full response path:

```text
GPGPU-Sim
    -> Gem5Balar
    -> gem5 / CHI
    -> Gem5Balar
    -> GPGPU-Sim
```

has been demonstrated reliably.

---

# 19. Known limitations

The current implementation contains several experimental design choices:

- one shared `gpu_mem` RequestPort for all GPU clusters;
- timing-address remapping into a guest-physical aperture;
- GPU writes represented as timing reads;
- GPGPU-Sim remains the functional owner of GPU memory;
- GPU clock period is currently configured explicitly by Gem5Balar;
- the reference CNM integration currently requires board-side modifications;
- backpressure behavior for the shared requester is still being validated.

These should be treated as explicit design points rather than hidden implementation details.

---

# 20. Future work

Possible next steps include:

```text
complete M3.7 round-trip validation
evaluate global versus per-cluster backpressure
measure RequestPort/RNI contention
introduce fair scheduling between GPU clusters
evaluate multiple gpu_mem requesters
replace temporary board modifications with a formal integration API
define a proper GPU physical-address model
implement real timing writes
validate additional CUDA workloads
collect timing and contention statistics
```

---

# 21. Project goal

The long-term objective is to provide an integration where:

```text
gem5
    models CPU, system and memory hierarchy timing

GPGPU-Sim / Accel-Sim
    models GPU execution

Gem5Balar
    bridges both simulators
```

while preserving enough compatibility with the previous SST/Balar interface to reuse the existing CUDA/GPGPU-Sim software stack with minimal changes.
