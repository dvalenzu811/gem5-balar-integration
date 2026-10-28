#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" &&
    pwd
)"

target="${1:-help}"

case "$target" in
    gpgpu)
        exec "${SCRIPT_DIR}/build_gpgpu.sh"
        ;;

    vecadd)
        exec "${SCRIPT_DIR}/build_vecadd.sh"
        ;;

    gem5)
        exec "${SCRIPT_DIR}/build_gem5.sh"
        ;;

    help|-h|--help)
        echo "Gem5Balar build targets:"
        echo
        echo "  $0 gpgpu   Build modified GPGPU-Sim"
        echo "  $0 vecadd  Build host and Linux-guest vecadd ELFs"
        echo "  $0 gem5    Build gem5 + Gem5Balar"
        echo
        echo "Typical environments:"
        echo "  gpgpu/gem5 : BSC compute node"
        echo "  vecadd      : RISC-V cross-build machine"
        ;;

    *)
        echo "ERROR: unknown build target: $target"
        echo "Run:"
        echo "  $0 help"
        exit 2
        ;;
esac
