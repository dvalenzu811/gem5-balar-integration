from m5.objects.Device import DmaDevice
from m5.params import *


class Gem5Balar(DmaDevice):
    type = "Gem5Balar"
    cxx_header = "dev/balar/gem5_balar.hh"
    cxx_class = "gem5::Gem5Balar"

    pio_addr = Param.Addr(
        "Base address of the Balar MMIO region"
    )

    pio_size = Param.Addr(
        0x1000,
        "Size of the Balar MMIO region"
    )

    pio_latency = Param.Latency(
        "10ns",
        "Balar MMIO access latency"
    )

    gpu_cores = Param.Unsigned(
        80,
        "Number of simulated GPU clusters"
    )

    max_cache_trans = Param.Unsigned(
        512,
        "Maximum outstanding mem_fetch transactions per GPU cluster"
    )

    gpu_mem = RequestPort(
        "GPGPU-Sim mem_fetch timing port"
    )

    cuda_executable = Param.String(
        "",
        "CUDA executable used by GPGPU-Sim for fatbin registration"
    )

    # Original SST Balar defaults its clock parameter to 1 GHz.
    # Express the equivalent period explicitly for gem5 EventQueue.
    gpu_clock_period = Param.Latency(
        "1ns",
        "Period between calls to SST_gpu_core_cycle()"
    )
