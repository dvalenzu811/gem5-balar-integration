#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" &&
    pwd
)"

ROOT="$(
    cd "${SCRIPT_DIR}/.." &&
    pwd
)"

GUEST_ELF="${ROOT}/build/vecadd/guest/vecadd-ubuntu"

if [[ ! -f "$GUEST_ELF" ]]; then
    echo "ERROR: guest vecadd ELF not found:"
    echo "  $GUEST_ELF"
    echo
    echo "Build it first:"
    echo "  ./scripts/build.sh vecadd"
    exit 1
fi

: "${BENCHDISK_BASE:?Set BENCHDISK_BASE to the input benchmark disk image}"
: "${BENCHDISK_OUT:=${ROOT}/build/benchdisk/benchmarks-balar.img}"

for tool in debugfs e2fsck sha256sum; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: required tool not found: $tool"
        exit 1
    fi
done

if [[ ! -f "$BENCHDISK_BASE" ]]; then
    echo "ERROR: benchmark disk base image not found:"
    echo "  $BENCHDISK_BASE"
    exit 1
fi

mkdir -p "$(dirname "$BENCHDISK_OUT")"

echo "===== COPYING BASE BENCHMARK DISK ====="

cp --reflink=auto --sparse=always \
    "$BENCHDISK_BASE" \
    "$BENCHDISK_OUT"

echo
echo "input:  $BENCHDISK_BASE"
echo "output: $BENCHDISK_OUT"

echo
echo "===== EXISTING GUEST FILE ====="

debugfs -R \
    "stat /bin/vecadd-ubuntu" \
    "$BENCHDISK_OUT" \
    2>/dev/null || true

# Replace the existing file if present.
debugfs -w -R \
    "rm /bin/vecadd-ubuntu" \
    "$BENCHDISK_OUT" \
    2>/dev/null || true

echo
echo "===== WRITING NEW GUEST ELF ====="

debugfs -w -R \
    "write $GUEST_ELF /bin/vecadd-ubuntu" \
    "$BENCHDISK_OUT"

debugfs -w -R \
    "set_inode_field /bin/vecadd-ubuntu mode 0100755" \
    "$BENCHDISK_OUT"

echo
echo "===== IMAGE FILE STAT ====="

debugfs -R \
    "stat /bin/vecadd-ubuntu" \
    "$BENCHDISK_OUT"

VERIFY="$(mktemp)"

cleanup()
{
    rm -f "$VERIFY"
}

trap cleanup EXIT

debugfs -R \
    "dump /bin/vecadd-ubuntu $VERIFY" \
    "$BENCHDISK_OUT"

echo
echo "===== SHA256 VERIFICATION ====="

echo "source:"
sha256sum "$GUEST_ELF"

echo
echo "image:"
sha256sum "$VERIFY"

SOURCE_SHA="$(
    sha256sum "$GUEST_ELF" |
    awk '{print $1}'
)"

IMAGE_SHA="$(
    sha256sum "$VERIFY" |
    awk '{print $1}'
)"

if [[ "$SOURCE_SHA" != "$IMAGE_SHA" ]]; then
    echo
    echo "ERROR: benchmark disk copy does not match guest ELF."
    exit 1
fi

echo
echo "Guest ELF staging SHA256: OK"

echo
echo "===== FILESYSTEM CHECK ====="

set +e
e2fsck -fn "$BENCHDISK_OUT"
RC=$?
set -e

# e2fsck status 0 = clean, 1 = filesystem errors corrected.
# -n is read-only, so for our validation we require clean.
if [[ "$RC" -ne 0 ]]; then
    echo
    echo "ERROR: e2fsck returned $RC"
    exit "$RC"
fi

echo
echo "Benchmark disk staging complete."
echo
echo "Host path:"
echo "  $BENCHDISK_OUT"
echo
echo "Guest executable:"
echo "  /data/bin/vecadd-ubuntu"
