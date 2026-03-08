#!/usr/bin/env bash

set -e

TARGET=thumbv7em-none-eabihf
PROFILE=release
APP_NAME=$(basename $(pwd))

ELF=target/$TARGET/$PROFILE/$APP_NAME
ARTIFACTS=artifacts

echo "=============================="
echo " Building firmware"
echo "=============================="

cargo build --release

echo "=============================="
echo " Preparing artifacts"
echo "=============================="

mkdir -p $ARTIFACTS

echo "Generating BIN..."
rust-objcopy -O binary $ELF $ARTIFACTS/app.bin

echo "Generating HEX..."
rust-objcopy -O ihex $ELF $ARTIFACTS/app.hex

echo "Copying ELF..."
cp $ELF $ARTIFACTS/app.elf

echo "=============================="
echo " Firmware artifacts generated"
echo "=============================="

ls -lh $ARTIFACTS