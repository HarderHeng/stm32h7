# PLAN: STM32H7 Rust Firmware Implementation

**Companion to**: `2026-07-22-stm32h7-rust-firmware-design.md`
**Strategy**: Bottom-up, host-test-first. Each phase produces something runnable/verifiable before moving on.

---

## Phase Overview

| # | Phase | Goal | Verification | Estimated Effort |
|---|---|---|---|---|
| 0 | Workspace scaffold | `cargo build --release` succeeds for empty target | 2 hours |
| 1 | Quantization tooling | Generate INT8 weights from upstream `policy.onnx` | 3 hours |
| 2 | `mlp` crate | Host unit tests pass with <1% error vs Python FP32 | 6 hours |
| 3 | `ankle` crate | Host unit tests for IK/FK/Jacobian | 5 hours |
| 4 | `canproto` crate | Bit-exact MIT frame codec vs dm_motor_driver.cpp | 3 hours |
| 5 | `hal` traits + mocks | Mock traits enable host integration tests | 2 hours |
| 6 | Embassy scaffold | `cargo build` succeeds with 4 tasks + ISRs | 4 hours |
| 7 | IMU driver | USART4 + HiPNUC state machine | 4 hours |
| 8 | CAN driver | bxCAN wrapper, DM motor protocol | 6 hours |
| 9 | Observation + action | Frame stack, post-process pipeline | 3 hours |
| 10 | Main integration | Full pipeline running in QEMU | 4 hours |
| 11 | Hardware smoke | Flash to real H743, motor response test | manual |

**Total engineering effort**: ~42 hours (~1 week for a Rust-experienced embedded engineer)

---

## Phase 0: Workspace Scaffold

**Goal**: A Cargo workspace that cross-compiles to `thumbv7em-none-eabihf` and emits an empty ELF file.

**Files to create**:
- `Cargo.toml` (workspace root)
- `rust-toolchain.toml`
- `.cargo/config.toml`
- `crates/bin/Cargo.toml`
- `crates/bin/memory.x`
- `crates/bin/src/main.rs` (minimal `#[entry] fn main()`)
- `.gitignore`

**Verification**:
```bash
cd ~/test/stm32h7
cargo build --release
# Expected: target/thumbv7em-none-eabihf/release/atom01-fw exists
# Expected: file size < 1 KB (just startup code)
```

**Notes**:
- Use `cortex-m-rt` for the `#[entry]` and vector table
- Use `panic-probe` or `panic-semihosting` for startup panic handling
- `memory.x` must match H743 memory layout exactly

---

## Phase 1: Quantization Tooling

**Goal**: A Python script that reads `policy.onnx` and produces INT8 weight `.bin` files + a `manifest.json`.

**Files to create**:
- `tools/quantize_onnx.py`
- `tools/test_quantization.py` (validates INT8 model produces same output as FP32 within 1%)
- `tools/requirements.txt` (onnx, onnxruntime, numpy)

**Script workflow**:
1. Load `policy.onnx` via `onnx.load()`
2. Extract weights per layer (4 weight tensors + 4 bias tensors)
3. Compute per-tensor scale: `scale = max(|tensor|) / 127`
4. Quantize: `q = round(tensor / scale).clip(-128, 127)`
5. Write to `weights/{w,b}{1..4}_int8.bin`
6. Compute reference output using FP32 ONNX Runtime
7. Compute reference output using INT8 (manual dequantization)
8. Assert |INT8_out - FP32_out| / |FP32_out| < 0.01 for 10 random inputs
9. Write `manifest.json` with shapes, scales, validation results

**Output `weights/manifest.json`**:
```json
{
  "input_shape": [1, 780],
  "output_shape": [1, 23],
  "layers": [
    {"name": "w1", "shape": [512, 780], "scale": 0.0123, "size_bytes": 399360},
    ...
  ],
  "validation": {
    "max_relative_error": 0.0042,
    "samples_tested": 10,
    "passed": true
  }
}
```

**Verification**:
```bash
python tools/quantize_onnx.py \
    --input /path/to/policy.onnx \
    --output weights/

ls weights/*.bin  # should see 8 .bin files + manifest.json
cat weights/manifest.json | jq .validation.passed  # should be "true"
```

---

## Phase 2: `mlp` Crate

**Goal**: A pure-Rust MLP that performs INT8 forward pass with <1% error vs Python reference.

**Files to create**:
- `crates/mlp/Cargo.toml`
- `crates/mlp/src/lib.rs`
- `crates/mlp/tests/reference_test.rs` (host-only)

**Internal layout**:
```rust
// crates/mlp/src/lib.rs

pub mod weights {
    pub const W1: &[u8] = include_bytes!("../../../weights/w1_int8.bin");
    pub const W2: &[u8] = include_bytes!("../../../weights/w2_int8.bin");
    pub const W3: &[u8] = include_bytes!("../../../weights/w3_int8.bin");
    pub const W4: &[u8] = include_bytes!("../../../weights/w4_int8.bin");
    pub const B1: &[u8] = include_bytes!("../../../weights/b1_int8.bin");
    pub const B2: &[u8] = include_bytes!("../../../weights/b2_int8.bin");
    pub const B3: &[u8] = include_bytes!("../../../weights/b3_int8.bin");
    pub const B4: &[u8] = include_bytes!("../../../weights/b4_int8.bin");
}

pub const SCALES: [(f32, f32); 4] = [
    (0.0123, 0.001),  // (w_scale, b_scale) for layer 1
    ...
];

pub struct Mlp;

impl Mlp {
    pub fn forward(stacked_obs: &[f32; 780], out: &mut [f32; 23]) {
        // 4 × (INT8 GEMM + ELU + bias)
    }
}

fn elu(x: f32) -> f32 {
    if x >= 0.0 { x } else { x.exp() - 1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn forward_matches_python_within_1_percent() {
        let obs = load_test_obs();  // from fixtures
        let mut out = [0.0; 23];
        Mlp::forward(&obs, &mut out);
        let expected = load_expected_out();
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() / b.abs().max(1e-6) < 0.01,
                "output mismatch: got {}, expected {}", a, b);
        }
    }
}
```

**Verification**:
```bash
cd crates/mlp
cargo test
# Expected: 1 passed; 0 failed
```

**Note**: This crate must be `no_std + alloc`-free. Uses `include_bytes!` to embed weights at compile time.

---

## Phase 3: `ankle` Crate

**Goal**: Rust port of `decouple_atom01.cpp` with full unit test coverage.

**Files to create**:
- `crates/ankle/Cargo.toml`
- `crates/ankle/src/lib.rs`
- `crates/ankle/tests/fk_convergence_test.rs`
- `crates/ankle/tests/jacobian_test.rs`
- `crates/ankle/tests/closed_chain_test.rs`

**Verification scenarios**:
1. **FK at zero pose**: motor angles (0, 0) → joint angles (0, 0)
2. **FK roundtrip**: pick joint angles → IK → motor angles → FK → same joint angles (within 1e-3)
3. **Jacobian at typical pose**: numerical gradient matches analytical Jacobian (within 1e-2)
4. **Left vs right**: l_spacing sign flips correctly

```rust
#[test]
fn fk_zero_pose_returns_zero_joint() {
    let decoupler = AnkleDecoupler::new(Side::Left);
    let result = decoupler.forward_kinematics(0.0, 0.0);
    assert!(result.pitch.abs() < 1e-3);
    assert!(result.roll.abs() < 1e-3);
    assert!(result.converged);
}

#[test]
fn roundtrip_preserves_joint_angles() {
    let decoupler = AnkleDecoupler::new(Side::Left);
    let q_target = [0.3, 0.1];  // pitch, roll
    let motor_angles = decoupler.inverse_kinematics(q_target[0], q_target[1]);
    let q_back = decoupler.forward_kinematics(motor_angles[0], motor_angles[1]);
    assert!((q_back.pitch - q_target[0]).abs() < 1e-3);
    assert!((q_back.roll - q_target[1]).abs() < 1e-3);
}
```

**Verification**:
```bash
cd crates/ankle
cargo test
# Expected: 5+ passed
```

---

## Phase 4: `canproto` Crate

**Goal**: Pure-Rust MIT frame encoder + decoder, bit-exact match with C++ reference.

**Files to create**:
- `crates/canproto/Cargo.toml`
- `crates/canproto/src/lib.rs`
- `crates/canproto/tests/mapping_test.rs` (range_map)
- `crates/canproto/tests/mit_encode_test.rs`
- `crates/canproto/tests/mit_decode_test.rs`

**Test cases** (sourced from `dm_motor_driver.cpp:365-409`):
1. `range_map(0.0, -12.5, 12.5, 0, 0xFFFF) == 0x8000`
2. `range_map(12.5, -12.5, 12.5, 0, 0xFFFF) == 0xFFFF`
3. `range_map(-12.5, -12.5, 12.5, 0, 0xFFFF) == 0`
4. Encode (0.0, 0.0, 100.0, 3.3, 0.0) for DM4340P motor 1 → expected byte sequence verified against `inference_node.cpp` reference
5. Decode feedback frame → expected MitFeedback struct

**Verification**:
```bash
cd crates/canproto
cargo test
```

---

## Phase 5: `hal` Traits + Mocks

**Goal**: HAL traits that both real drivers and host-side tests can implement.

**Files to create**:
- `crates/hal/Cargo.toml`
- `crates/hal/src/lib.rs` (traits)
- `crates/hal/src/mock.rs` (mock implementations for tests)

**Trait surface**:
```rust
pub trait CanBus {
    type Error;
    fn transmit(&mut self, frame: &CanFrame) -> Result<(), Self::Error>;
    fn receive(&mut self) -> Option<CanFrame>;
}

pub trait SerialPort {
    fn read_byte(&mut self) -> Option<u8>;
    fn write_bytes(&mut self, data: &[u8]) -> Result<(), SerialError>;
}
```

**Verification**: `cargo test -p hal` compiles and runs (smoke tests only).

---

## Phase 6: Embassy Scaffold

**Goal**: A minimal Embassy async app with 2 tasks + 2 ISRs that compiles to STM32H7. No real work yet.

**Files to create**:
- `crates/bin/Cargo.toml` (update with embassy-executor deps)
- `crates/bin/src/main.rs`

**Key structure**:
```rust
#[embassy_executor::main (or #[embassy_executor::task] per async function)]
mod app {
    #[shared]
    struct Shared { obs: [f32; 78], action: [f32; 23] }

    #[local]
    struct Local { mlp: MlpInference, ankle: AnkleDecoupler }

    #[init]
    fn init(cx: init::Context) -> (Shared, Local) { ... }

    #[task(shared = [obs, action], priority = 2, period = 20ms)]
    fn inference_task(cx: inference_task::Context) { ... }

    #[task(shared = [action], priority = 3, period = 4ms)]
    fn control_task(cx: control_task::Context) { ... }

    #[task(binds = USART4, priority = 4)]
    fn usart4(cx: usart4::Context) { ... }

    #[task(binds = CAN1_RX0, priority = 5)]
    fn can1_rx0(cx: can1_rx0::Context) { ... }
}
```

**Verification**:
```bash
cd crates/bin
cargo build --release
# Expected: ELF builds, < 5 KB
```

**Note**: The first compile may take 5-10 minutes for embassy-stm32 + embassy-executor dependency tree.

---

## Phase 7: IMU Driver

**Goal**: USART4 + HiPNUC frame parser state machine.

**Files to create**:
- `crates/drivers/Cargo.toml`
- `crates/drivers/src/lib.rs`
- `crates/drivers/src/imu.rs`

**State machine** (Hi91 protocol from `hipnuc_imu_driver.cpp`):
```
IDLE → recv 0x5A → recv 0xA5 → recv length → recv payload → recv CRC → parse → IDLE
```

**Test strategy**:
- Unit test: feed captured bytes → assert parsed quat/ang_vel
- Integration test: USART loopback (TX → RX on same board)

---

## Phase 8: CAN Driver

**Goal**: bxCAN wrapper + DM motor protocol.

### Phase 8.0: Bus Architecture Decision (must resolve first)

Three options, listed in order of preference:

| Option | Bus count | Pros | Cons |
|---|---|---|---|
| **A. Dual-bus shared** | 2 bxCAN, 23 motors time-multiplexed | Native hardware, simplest firmware | ~2× bus load; need motor_id arbitration |
| **B. Quad-bus via 2× SPI-MCP2515** | 2 bxCAN + 2 SPI-CAN bridges | Matches upstream topology | 2 extra SPI peripherals, +$5 BOM |
| **C. Dual-bus + software multiplexer** | 2 bxCAN, manual CAN ID ranges per bus | Cheapest | Reduces effective bandwidth by half |

**Recommended**: Option A (dual-bus shared) for v1.0. Acceptable because 250 Hz × 23 motors / 1 Mbps ≈ 2.3% bus utilization. Re-evaluate to Option B if motor jitter becomes an issue.

**If you choose Option B or C instead, document the choice here and adjust file list below.**

### Files to create (assuming Option A)
- `crates/drivers/src/can.rs` (bxCAN1 + bxCAN2 wrapper)
- `crates/drivers/src/dm_motor.rs` (MIT encode wrapper using `canproto`)

**Key structure**:
- `CanBusPair` struct holds CAN1 + CAN2 handles
- Motor address → bus routing table built from `motor_id` in robot.yaml
- Transmission dispatches to correct bus by motor address

**Test strategy**:
- Loopback test: send MIT frame on CAN1 → receive on CAN1 (loopback mode)
- Bit-exact test against `dm_motor_driver.cpp` reference

---

## Phase 9: Observation + Action Post-Processing

**Goal**: Frame stack + USD→URDF reorder + clip/scale/offset.

**Files to create**:
- `crates/bin/src/observation.rs`
- `crates/bin/src/action.rs`

**Frame stack**: ring buffer of 10 obs[78] arrays, writes most-recent, reads chronological.

---

## Phase 10: Main Integration + QEMU Test

**Goal**: Full pipeline runs in QEMU emulation.

**Files to update**:
- `crates/bin/src/main.rs` (full Embassy app)

**QEMU test** (verified QEMU ≥8.0):
```bash
# STM32H743 is supported in QEMU as `-M stm32h7xx-soc` since QEMU 8.0
# Actual hardware-specific peripherals (USART4, CAN1, etc.) are NOT emulated
# — QEMU test only verifies: init code, Embassy async task scheduling, panic behavior
qemu-system-arm \
    -M stm32h7xx-soc \
    -cpu cortex-m7 \
    -kernel target/thumbv7em-none-eabihf/release/atom01-fw \
    -nographic \
    -semihosting \
    -monitor none \
    -serial null

# For RTT log capture, use a JLink gdb server or `probe-rs run --chip STM32H743VITx`
# QEMU does NOT natively support SEGGER RTT — use semihosting instead for QEMU tests
```

**Limitation**: QEMU does not emulate bxCAN, USART4, or external SPI Flash. The Phase 10 test only validates:
- Firmware boots without panic
- Embassy executor starts all spawned tasks
- Init sequence completes (clock config, weight load)
- No hardware-dependent path is exercised

Full hardware validation is in Phase 11.

---

## Phase 11: Hardware Smoke Test (Manual)

**Goal**: Verify the firmware boots and controls motors on real STM32H743.

**Procedure** (manual, documented in README):
1. Connect ST-Link V2 to PC
2. Power on H743 board
3. Flash: `cargo run --release`
4. Observe RTT log: `probe-rs rtt`
5. Verify init logs appear (clocks, peripherals, weight load)
6. Verify control_task runs every 4 ms (RTT log timestamps)
7. Disconnect motors from robot, test MIT command encoding with single motor in torque mode
8. Verify joint position feedback received

---

## Execution Order

The phases have minimal dependencies between them. Parallelizable:

```
Sequential dependencies:
  Phase 0 → Phase 6 → Phase 10 → Phase 11

Phase 1 must precede Phase 2
Phase 5 must precede Phase 8 (for mock-based tests)
Phases 2, 3, 4 are independent of each other
Phase 7 depends on Phase 6
Phase 8 depends on Phase 6, Phase 4, Phase 5
Phase 9 depends on Phase 2

Optimal sequence:
  0 → 1 → (2, 3, 4 in parallel) → 5 → 6 → (7, 8, 9 in parallel) → 10 → 11
```

---

## Verification Checklist

After all phases complete:

```bash
cd ~/test/stm32h7

# 1. Lint
cargo clippy --workspace --all-targets -- -D warnings

# 2. Format
cargo fmt --all -- --check

# 3. Host tests (algorithm crates)
cargo test --workspace
# Expected: mlp (1), ankle (5), canproto (4), hal (1) all pass

# 4. Target build
cargo build --release --target thumbv7em-none-eabihf
# Expected: target/thumbv7em-none-eabihf/release/atom01-fw exists, < 30 KB

# 5. QEMU smoke
qemu-system-arm -M stm32h743 -cpu cortex-m7 \
    -kernel target/thumbv7em-none-eabihf/release/atom01-fw \
    -nographic -semihosting-config enable=on,target=native &
sleep 5 && kill %1
# Expected: "init complete" log line

# 6. Memory check (linker map)
cargo build --release --target thumbv7em-none-eabihf
# Inspect target/.../release/atom01-fw.map
# Expected: .text < 800 KB, .data + .bss < 100 KB
```

---

## Notes on Testing Philosophy

1. **Algorithm crates are pure Rust, no `no_std` needed for tests** — they test on host x86_64 in milliseconds
2. **Real drivers are `no_std`** — tested via integration tests that use mock HAL
3. **Embassy bin** — only smoke-tested on target; full validation is on hardware
4. **Numerical accuracy** is verified against Python reference fixtures checked into `tests/fixtures/`
5. **Reference URLs** to upstream C++ code are in every file's module-level doc comment for traceability
