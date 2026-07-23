#!/usr/bin/env bash
# Build + flash + RTT monitor convenience script.
# Usage:
#   ./scripts/flash.sh           # build + flash + monitor RTT
#   ./scripts/flash.sh build     # build only
#   ./scripts/flash.sh flash     # flash only (assumes already built)
#   ./scripts/flash.sh rtt       # monitor RTT only
#   ./scripts/flash.sh size      # print binary size
#   ./scripts/flash.sh artifacts # write ELF + bin + hex to artifacts/

set -euo pipefail

CHIP="${CHIP:-STM32H743ZI}"
PROFILE="${PROFILE:-release}"
TARGET="thumbv7em-none-eabihf"
APP="stm32h7"
ELF="target/${TARGET}/${PROFILE}/${APP}"
ARTIFACTS="artifacts"

cmd="${1:-all}"

build() {
    echo ">>> cargo build --${PROFILE} --target ${TARGET}"
    cargo build --"${PROFILE}" --target "${TARGET}"
}

flash() {
    echo ">>> probe-rs download --chip ${CHIP}"
    probe-rs download --chip "${CHIP}" "${ELF}"
    echo ">>> probe-rs reset --chip ${CHIP}"
    probe-rs reset --chip "${CHIP}"
}

artifacts() {
    mkdir -p "${ARTIFACTS}"
    rust-objcopy -O binary "${ELF}" "${ARTIFACTS}/app.bin"
    rust-objcopy -O ihex  "${ELF}" "${ARTIFACTS}/app.hex"
    cp "${ELF}" "${ARTIFACTS}/app.elf"
    echo ">>> artifacts written to ${ARTIFACTS}/"
    ls -lh "${ARTIFACTS}/"
}

rtt() {
    echo ">>> probe-rs rtt --chip ${CHIP}"
    probe-rs rtt --chip "${CHIP}"
}

size() {
    echo ">>> Binary size:"
    size "${ELF}"
    echo ">>> Section sizes:"
    size -A "${ELF}" 2>/dev/null | head -30 || true
}

case "$cmd" in
    build)     build ;;
    flash)     flash ;;
    rtt)       rtt ;;
    size)      size ;;
    artifacts) artifacts ;;
    all)       build && flash && echo "(run './scripts/flash.sh rtt' in another shell for logs)" ;;
    *)         echo "Unknown command: $cmd"; echo "Usage: $0 {build|flash|rtt|size|artifacts|all}"; exit 1 ;;
esac
