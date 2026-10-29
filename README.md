# gem5-Balar Integration

Portable integration of **BALAR + gem5 + GPGPU-Sim** used as the functional baseline for the gem5-Balar project.

The repository does not contain full copies of gem5, GPGPU-Sim, SST Elements or gem5-infra. Instead, it stores the source files, patches, scripts and configuration required to reproduce the integration on top of known external revisions.


## Repository structure

```text
configs/        Local configuration templates
evidence/       Validated versions and build information
patches/        Patches for external repositories
scripts/        Setup, build and execution scripts
src/            Canonical source files maintained by this repository
```

Historical M3.7 files are kept under:

```text
patches/legacy/m37/
scripts/legacy/
```

Reference or diagnostic patches that are not automatically installed are stored under:

```text
patches/reference/
```

## External dependencies

The integration expects external source trees for:

- gem5
- GPGPU-Sim / Accel-Sim
- SST Elements
- gem5-infra

The revisions used by the validated baseline are documented in:

```text
evidence/versions.txt
```
## Requirements and validated versions

This repository is **not a standalone installer**.

The scripts in this repository do not download or install gem5, GPGPU-Sim, SST Elements, gem5-infra, compilers or system dependencies.

They reproduce the BALAR integration on top of **already existing external source trees**.

The exact external revisions used by the validated baseline are recorded in:

```text
evidence/versions.txt
```

Those revisions should be considered part of the reproducible configuration.

### Validated build environment

The current portable baseline has been validated with the following environment:

| Component | Validated version / configuration |
|---|---|
| gem5 | DEVELOP-FOR-25.1 baseline |
| Python | 3.12.9 |
| SCons | 4.0.1 |
| GCC | 11.5.0 |
| G++ | 11.5.0 |
| BSC Miniforge module | 24.11.3-2 |
| GPGPU-Sim CUDA configuration | CUDA 11.8 (`cuda-11080`) |
| gem5 build | `build/BSC/gem5.opt` |
| Host architecture used at BSC | AArch64 compute nodes |

On BSC, the validated gem5 build environment is initialized with:

```bash
module purge
module load miniforge/24.11.3-2

export PYTHONNOUSERSITE=1
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export SCONS_BIN=/apps/MINIFORGE/24.11.3-2/bin/scons
```

This environment is reproduced by:

```text
scripts/build_gem5_compute.slurm
```

The successful portable build generated:

```text
build/BSC/dev/balar/gem5_balar.o
build/BSC/gem5.opt
```

and the resulting binary exported:

```text
is_SST_buffer_full(...)
send_read_request_SST(...)
send_write_request_SST(...)
```

### Required external repositories

Before using this integration repository, the following repositories must already exist:

```text
gem5
GPGPU-Sim / Accel-Sim
SST Elements
gem5-infra
```

Their paths are configured through:

```text
configs/local.env
```

For example:

```bash
GEM5_DIR=/path/to/gem5
GPGPUSIM_ROOT=/path/to/gpgpu-sim
SST_ELEMENTS_DIR=/path/to/sst-elements
GEM5_INFRA_DIR=/path/to/gem5-infra
```

The exact source revisions expected by the integration are stored in:

```text
evidence/versions.txt
```

`scripts/apply.sh` checks these revisions before installing the integration.

A revision mismatch is therefore treated separately from a build failure.

## What the scripts actually do

### `scripts/setup_env.sh`

Does **not** install dependencies.

It reads the local configuration and exports the paths and environment variables required by the existing installations.

### `scripts/apply.sh`

Does **not** clone external repositories.

It assumes the external repositories already exist.

It then:

1. checks their revisions;
2. copies canonical BALAR files from `src/`;
3. applies the patches stored under `patches/`;
4. detects patches that are already applied;
5. installs workload/runtime files into the corresponding external trees.

Conceptually:

```text
existing external repositories
        +
gem5-balar-integration/src
        +
gem5-balar-integration/patches
        |
        v
configured BALAR development trees
```

### `scripts/build.sh`

Does **not** install compilers or build dependencies.

It dispatches to:

```text
scripts/build_gpgpu.sh
scripts/build_gem5.sh
scripts/build_vecadd.sh
```

and builds the corresponding component using the already configured development environment.

## Reproduction model

A fresh setup therefore consists of two different stages.

### 1. External environment preparation

This part is currently **not automated by this repository**.

The user must provide:

```text
compatible gem5 source tree
compatible GPGPU-Sim source tree
compatible SST Elements source tree
compatible gem5-infra source tree
required Python/SCons/compiler environment
RISC-V toolchain required by the guest workload
required images/checkpoints
```

The versions and revisions used by the reference environment are documented under `evidence/`.

### 2. BALAR integration

Once those dependencies exist:

```bash
cp configs/local.env.example configs/local.env
# Edit the paths.

source scripts/setup_env.sh

./scripts/apply.sh
```

Then the individual components can be built.

On BSC:

```bash
./scripts/build.sh gpgpu
sbatch scripts/build_gem5_compute.slurm
```

The vecadd binaries are built with the appropriate RISC-V cross-build environment:

```bash
./scripts/build.sh vecadd
```

After the benchmark disk and checkpoint are available:

```bash
./scripts/stage_vecadd_benchdisk.sh
./scripts/generate_vecadd.sh
./scripts/run_vecadd.sh
```

Therefore, this repository should be understood as a **portable integration layer**, not as a complete dependency installer.

## Local configuration

Create the machine-specific configuration:

```bash
cp configs/local.env.example configs/local.env
```

Edit `configs/local.env` and configure the paths to the external repositories.

For example:

```text
GEM5_DIR=...
GPGPUSIM_ROOT=...
SST_ELEMENTS_DIR=...
GEM5_INFRA_DIR=...
```

`configs/local.env` is intentionally ignored by Git.

## Environment

Load the integration environment with:

```bash
source scripts/setup_env.sh
```

## Applying the integration

Install the BALAR modifications into the external source trees with:

```bash
./scripts/apply.sh
```

The script:

1. checks the expected repository revisions;
2. copies the canonical files from `src/`;
3. applies the required patches;
4. detects patches that are already applied;
5. installs the BALAR runtime and workload files.

## Building

The main build interface is:

```bash
./scripts/build.sh <target>
```

Available targets are:

```text
gpgpu
gem5
vecadd
```

### GPGPU-Sim

```bash
./scripts/build.sh gpgpu
```

### gem5

```bash
./scripts/build.sh gem5
```

On BSC, gem5 is built on a compute node using:

```bash
sbatch scripts/build_gem5_compute.slurm
```

The validated build produces:

```text
build/BSC/dev/balar/gem5_balar.o
build/BSC/gem5.opt
```

and exports the BALAR GPU-memory hooks:

```text
is_SST_buffer_full(...)
send_read_request_SST(...)
send_write_request_SST(...)
```

### vecadd

```bash
./scripts/build.sh vecadd
```

The build creates separate host and guest binaries:

```text
build/vecadd/host/vecadd-ubuntu
build/vecadd/guest/vecadd-ubuntu
```

Their roles are documented in:

```text
evidence/vecadd-binary-roles.txt
```

## Running vecadd

The basic workflow is:

```bash
./scripts/stage_vecadd_benchdisk.sh
./scripts/generate_vecadd.sh
./scripts/run_vecadd.sh
```

A portable checkpoint can be created using:

```text
scripts/create_balar_portable_checkpoint.slurm
```

Generated disk images, checkpoints, simulation outputs and large binaries are intentionally not stored in Git.

## Current architecture

The current GPU-memory path is approximately:

```text
CUDA application
      |
      v
BALAR Linux runtime
      |
      v
Gem5Balar
      |
      v
GPGPU-Sim
      |
      | mem_fetch
      v
BALAR compatibility API
      |
      v
Gem5Balar GPU memory port
      |
      | sendTimingReq()
      v
gem5 timing memory system
```

The current implementation includes:

- CUDA/BALAR control calls
- GPGPU-Sim callbacks
- `mem_fetch` forwarding
- gem5 `RequestPort` integration
- `sendTimingReq()`
- `recvTimingResp()`
- `recvReqRetry()`
- outstanding transaction tracking
- backpressure handling

A real `vecadd` execution has exercised the complete control path and GPU-memory timing path.

## Architectural status

This implementation is an **experimental functional baseline**.

It was developed to prove that GPGPU-Sim memory traffic could enter the gem5 timing memory system and experience real timing and backpressure.

Several aspects are expected to change in future architecture work, including:

- the current shared GPU memory interface;
- read/write semantics;
- address translation;
- outstanding-request limits;
- DMA versus ordinary GPU-memory traffic;
- parts of the SST compatibility layer.

In particular, the current experimental path may represent GPU writes as:

```text
WRITE -> TIMING_READ
```

This is useful for timing-path validation but is not intended to represent the final memory semantics.

The current implementation should therefore remain available as a reproducible reference while future architecture work is developed separately.

## Reproducing the baseline

Typical workflow:

```bash
git clone <repository>
cd gem5-balar-integration

cp configs/local.env.example configs/local.env
# Edit configs/local.env

source scripts/setup_env.sh

./scripts/apply.sh

./scripts/build.sh gpgpu
./scripts/build.sh gem5
./scripts/build.sh vecadd

./scripts/stage_vecadd_benchdisk.sh
./scripts/generate_vecadd.sh
./scripts/run_vecadd.sh
```

On BSC, use the supplied Slurm scripts for operations requiring compute nodes.

## Validated portable baseline

The portable integration was established in:

```text
2a9f670 build: make BALAR gem5 integration portable
```

Future architectural development should be performed separately so this baseline remains available for comparison and regression testing.
