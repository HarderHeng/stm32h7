# SPEC: STM32H7 Rust Firmware for ROBOTO_ORIGIN Control Pipeline

**Date**: 2026-07-22
**Status**: Draft (pending review)
**Author**: Sisyphus (brainstorming output)
**Target hardware**: STM32H743VIT6 with 2 MB internal Flash, 1 MB internal RAM, 8 MB external SPI Flash (W25Q64), optional SD card slot

---

## 1. Purpose & Scope

### 1.1 Goal

Port the complete robot control pipeline of [ROBOTO_ORIGIN](https://github.com/Roboparty/roboto_origin) — originally implemented in C++ on a Linux SBC (RDK X5, Orange Pi 5 Plus) using ROS2 Humble + ONNX Runtime — to a **standalone embedded firmware** running on STM32H743, written in Rust using the Embassy async embedded framework.

The firmware must execute the **full inference + actuation pipeline** at production-grade latency without any Linux, ROS2, or external ML runtime dependency.

### 1.2 In-Scope

- IMU acquisition (HiPNUC HI13 via UART @ 921600 baud, 500 Hz)
- Observation building (78-dim, 10-frame history = 780-dim network input)
- On-device neural network inference (MLP [780→512→256→128→23], INT8 quantized, 50 Hz)
- Action post-processing (clip → scale → default-angle offset → USD→URDF reorder)
- Ankle closed-chain decoupling (forward kinematics, 4-bar IK, Jacobian transpose for torque mapping)
- Per-joint MIT-frame CAN command generation (23 motors, 4 CAN buses, 250 Hz)
- Real-time scheduling (Embassy executor with cooperative async tasks, NVIC-priority ISRs)
- Motor feedback acquisition and joint state tracking

### 1.3 Out-of-Scope

- RL training (handled by atom01_train, not ported)
- Sim2Sim verification (assumed already complete upstream)
- Multi-policy switching UI (single base policy only; can be extended later)
- Vision pipeline (no cameras, no visual encoders)
- Wireless / SSH / WiFi AP (firmware is headless; debug log via RTT only)
- Battery management (BMS daemon runs on a separate SBC; this firmware assumes 48 V bus is stable)

### 1.4 Success Criteria

| # | Criterion | Metric | How Verified |
|---|---|---|---|
| S1 | Firmware builds for `thumbv7em-none-eabihf` | `cargo build --release` succeeds | CI / manual |
| S2 | All algorithm crates pass host-side unit tests | `cargo test` 100% pass | CI |
| S3 | MLP INT8 forward matches Python FP32 reference within 1% relative error | Test in Phase 2 | Unit test |
| S4 | Ankle forward kinematics converges in <10 iterations from motor angles | Test in Phase 3 | Unit test |
| S5 | MIT frame encoding bit-exact to dm_motor_driver.cpp reference | Test in Phase 4 | Unit test |
| S6 | Inference task runs every 20 ms ± 0.5 ms on real hardware | logic analyzer + Embassy Instant::now() timestamps | Integration |
| S7 | Control task runs every 4 ms ± 0.1 ms on real hardware | logic analyzer + Embassy Instant::now() timestamps | Integration |
| S8 | All 23 joints respond to control commands on real hardware | Smoke test with motors disabled | Manual |

---

## 2. Hardware Assumptions

### 2.1 Target MCU

| Property | Value | Source |
|---|---|---|
| MCU | STM32H743VIT6 | User-stated |
| Core | Cortex-M7 @ 480 MHz | Datasheet |
| FPU | Single + Double precision | Datasheet |
| DSP extensions | Yes (M7 DSP) | ARM ARM |
| Internal Flash | 2 MB | User-stated |
| Internal RAM | 1 MB (DTCM 64 KB + AXI SRAM 256 KB + SRAM D2 128 KB + SRAM D3 64 KB) | Datasheet |
| External SPI Flash | 8 MB (W25Q64JV, single-bit SPI @ ≤50 MHz) | User-stated |
| SD card slot | SDHC-capable, SPI mode (or SDMMC if hardware supports) | User-stated |

### 2.2 Peripherals Used

| Peripheral | Pin (assumed) | Use |
|---|---|---|
| USART4 | TX/RX (assumed PA0/PA1) | HiPNUC IMU, 921600 baud |
| CAN1 | PB8/PB9 (assumed) | Left leg (motors 1-6) |
| CAN2 | PB12/PB13 (assumed) | Right leg + waist (motors 7-13) |
| CAN3 | PD0/PD1 (assumed) | Left arm (motors 14-18) |
| CAN4 | Not on H743 — likely routed via SPI-CAN bridge or shared bus | Right arm (motors 19-23) |

**Note**: STM32H743 has only **2 bxCAN controllers** natively. CAN3/CAN4 require external MCP2515/FD2515 over SPI. The 4-bus topology from the reference design must be mapped to available hardware. **Open question for Phase 7**: confirm pin assignments and whether 4 buses are needed or if a dual-bus fallback is acceptable.

### 2.3 Power & Reset

- 3.3 V regulated supply
- Boot from internal Flash by default (BOOT0 = 0)
- Optional external SPI Flash used as data storage only (not XIP)

---

## 3. Functional Requirements

### 3.1 Observation Building (FR-1)

The observation vector must match the training-time definition in `base_env.py:146-187`:

| Field | Dim | Source | Scale |
|---|---|---|---|
| `ang_vel` | 3 | IMU angular velocity (body frame) | 1.0 |
| `projected_gravity` | 3 | IMU quaternion inverted applied to (0,0,-1) | 1.0 |
| `cmd_vel` | 3 | External command (gamepad / `cmd_vel` topic) | 1.0 |
| `dof_pos - default` | 23 | Motor encoders, default-offset | 1.0 |
| `dof_vel` | 23 | Motor encoders, finite-difference | 1.0 |
| `last_action` | 23 | Previous MLP output | 1.0 |

Total single-frame: **78 floats**. Stacked over 10 frames: **780 floats**.

Clipping: each element clamped to ±100 before stacking (matches training).

### 3.2 Network Inference (FR-2)

- Architecture: MLP `[780 → 512 → 256 → 128 → 23]`
- Activation: ELU on hidden layers, identity on output
- Quantization: INT8 weights, FP32 activations (post-dequantization)
- Frequency: 50 Hz (every 20 ms)
- Latency budget: < 5 ms (vs. 20 ms budget — 4× safety margin)

### 3.3 Action Post-Processing (FR-3)

Applied in this exact order (matches `inference_node.cpp:303-309`):

```
for i in 0..23:
    a[i] = clip(raw[i], ±100)
    out[usd2urdf[i]] = a[i] * action_scale + default_angle[usd2urdf[i]]
out = usd2urdf_reordered(action)
```

- `action_scale = 0.25` — from upstream `inference.yaml:19`
- `clip_actions = 100.0` — from upstream `inference.yaml:20`
- `usd2urdf = [0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22]` — from upstream `inference.yaml:21`
- `default_angle = [0, 0, -0.1, 0.3, -0.2, 0, ...]` (23-vector) — from upstream `inference.yaml:26-30`

**Source-of-truth file**: all four constants are imported verbatim from `modules/atom01_deploy/src/inference/config/inference.yaml` at code-review time. A comment in `crates/bin/src/config.rs` will cite the upstream file path and commit hash.

### 3.4 Ankle Decoupling (FR-4)

Closed-chain parallel mechanism (4-bar linkage) for both left and right ankles:

- **Inputs**: 2 motor angles (long-link, short-link), joint velocity feedback
- **Outputs**: 2 motor torques (after Jacobian transpose mapping from joint torque)
- **Algorithm**:
  1. Forward kinematics: motor angles → joint angles (pitch, roll) via 4-bar constraint solver (Newton iteration, ≤100 iter, tol 1e-3)
  2. Joint-space PD: `τ_joint = kp·(q* − q) + kd·(0 − v)`
  3. Inverse Jacobian transpose: `τ_motor = J^T · τ_joint`
- **Constants** (from `decouple_atom01.cpp:7-41`):
  - `l_rod = {180, 110}` mm
  - `l_bar = 20` mm
  - `l_spacing = ±42.35` mm (sign flips for left vs right)

### 3.5 Motor Command Dispatch (FR-5)

Per-joint MIT command, format identical to `dm_motor_driver.cpp:358-409`:

| Field | Bits | Range | Mapping |
|---|---|---|---|
| `can_id` | 11 | motor_id 1-23 | direct |
| `pos` | 16 | ±12.5 rad | unsigned mapping |
| `vel` | 12 | ±20 rad/s (DM4340P) / ±25 rad/s (DM10010L) | unsigned mapping |
| `kp` | 12 | [0, 500] | unsigned mapping |
| `kd` | 12 | [0, 5] | unsigned mapping |
| `τ` | 12 | ±28 Nm (DM4340P) / ±200 Nm (DM10010L) | unsigned mapping |

For non-ankle joints: `vel = 0`, `τ = 0`, `pos = signed_target`, `kp/kd = robot.yaml values`.
For ankle motors: `pos = vel = kp = kd = 0`, `τ = computed torque` (see FR-4).

### 3.6 Real-Time Loop (FR-6)

Implemented with **Embassy async executor** (`embassy-executor` with `arch-cortex-m` feature). Two RTOS-style tasks cooperatively scheduled, plus four hardware interrupts.

| Async Task | Period | Priority | Stack | Work |
|---|---|---|---|---|
| `inference_task` | 20 ms | P2 | 1 KB | Build obs → stack → MLP → post-process |
| `control_task` | 4 ms | P3 | 1 KB | Read feedback → decouple → 23× MIT frame → CAN |

| ISR | Priority | Work |
|---|---|---|
| `CAN1_RX0` | P5 (NVIC) | Parse feedback → update `joint_states` |
| `CAN2_RX0` | P5 (NVIC) | Parse feedback → update `joint_states` |
| `USART4` | P4 (NVIC) | Parse HiPNUC byte → update `imu_state` |
| `DMA1 str4` (CAN TX complete) | P5 (NVIC) | Wake `control_task`, signal TX complete |

**Priority semantics**: lower number = higher priority. Tasks communicate via `embassy-sync` channels and `CriticalSectionSafeCell` shared state.

**Why Embassy over RTIC** (decision recorded): the existing scaffold at `~/test/stm32h7` was pre-built for Embassy (embassy-stm32 + embassy-executor + defmt + probe-rs). Embassy provides:
- `async/await` ergonomics for sequential logic (e.g., I2C, SPI transactions)
- Cooperative scheduling between async tasks
- NVIC-priority binding for hardware interrupts
- First-class embassy-stm32 HAL integration

**Trade-off accepted**: Embassy uses software scheduling (not compile-time priority checking like RTIC). Priority inversions must be caught by review. Mitigated by strict stack + state-machine design in tasks.

Worst-case latency from control_task trigger to last CAN TX complete: < 500 µs (well within 4 ms).

---

## 4. Non-Functional Requirements

### 4.1 Performance

| Metric | Target | Measurement |
|---|---|---|
| MLP inference latency | < 5 ms | Logic analyzer on GPIO toggle |
| Control loop latency | < 1 ms | Same |
| End-to-end obs→CAN latency | < 8 ms | Same |
| Jitter on inference_task | < ±0.5 ms | Embassy Instant::now() software timestamps |
| Jitter on control_task | < ±0.1 ms | Same |

### 4.2 Memory

| Resource | Available | Used (estimated) |
|---|---|---|
| Internal Flash | 2 MB | ~800 KB (code + constants) |
| Internal RAM | 1 MB | ~600 KB (incl. INT8 weights loaded at boot) |
| External SPI Flash | 8 MB | ~600 KB (weights) |
| Stack (per task) | 8 KB default | 1 KB configured |

INT8 weight footprint: ~0.55 MB (loaded to RAM at boot from SPI Flash in ~120 ms).

### 4.3 Reliability

- All `unwrap()` calls in firmware code must be justified (ISR-safe paths) or replaced with `defmt::panic!` (logged before halt)
- No heap allocation anywhere (all data structures `heapless::Vec` / stack arrays)
- Watchdog timer (IWDG) enabled in hardware init, fed by idle task
- CAN bus-off recovery: automatic retry per bxCAN spec
- IMU parser error recovery: drop frame, resync on next 0x5A header

### 4.4 Determinism

- All tasks have bounded execution time (verified by inspection + test)
- No `print` in time-critical paths (`defmt` is non-blocking but ISRs must stay short)
- Memory access patterns cache-friendly (sequential weight reads, no random pointer chasing)
- Embassy uses cooperative scheduling; priority inversions caught via code review and lockdep-style asserts

### 4.5 Power

Not a primary concern (firmware assumes wall power or stable battery via upstream BMS). CPU runs at 480 MHz full speed; unused peripherals gated off.

### 4.6 Maintainability

- Every algorithm crate must be `no_std + alloc`-free and host-testable
- Each module has a one-line top-of-file purpose comment
- Each public function has a doc comment with pre/post-conditions
- Magic numbers (kp, kd, motor_sign, etc.) live in a single `config.rs` per crate, sourced from YAML in upstream project
- Reference Python equivalent or C source URL in comments for traceability

---

## 5. Architecture

### 5.1 Crate Layout

```
robot-fw/                         # Cargo workspace root
├── Cargo.toml                   # workspace = [...]
├── .cargo/config.toml           # target = thumbv7em-none-eabihf
├── rust-toolchain.toml          # stable + rustfmt + clippy
├── crates/
│   ├── mlp/                     # Pure-Rust MLP forward (host-testable)
│   │   ├── Cargo.toml
│   │   └── src/lib.rs           # ~150 LOC: weights, INT8 GEMM, ELU
│   ├── ankle/                   # Pure-Rust ankle decoupling (host-testable)
│   │   ├── Cargo.toml
│   │   └── src/lib.rs           # ~250 LOC: IK, FK, Jacobian
│   ├── canproto/                # Pure-Rust MIT frame codec (host-testable)
│   │   ├── Cargo.toml
│   │   └── src/lib.rs           # ~120 LOC: encode/decode, range_map
│   ├── hal/                     # Hardware abstraction traits + mocks
│   │   ├── Cargo.toml
│   │   └── src/lib.rs           # CanBus, SerialPort traits + Mock impls
│   ├── drivers/                 # Real hardware drivers (no_std)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── can.rs           # bxCAN wrapper, 4 buses
│   │       └── imu.rs           # USART4 + HiPNUC state machine
│   └── bin/                     # Firmware entry point
│       ├── Cargo.toml
│       ├── memory.x             # Linker script for H743 memory map
│       └── src/main.rs          # Embassy app, init, tasks, ISRs
├── tools/
│   └── quantize_onnx.py         # FP32 ONNX → INT8 .bin + scales.json
├── weights/
│   ├── w1_int8.bin
│   ├── w2_int8.bin
│   ├── w3_int8.bin
│   ├── w4_int8.bin
│   ├── b1_int8.bin, b2_int8.bin, b3_int8.bin, b4_int8.bin
│   └── manifest.json
├── tests/                       # Cross-crate integration tests (host)
│   └── pipeline_test.rs
└── docs/superpowers/{specs,plans}/
```

### 5.2 Dependency Graph

```
bin ──► drivers ──► hal
  │        │
  │        └──► canproto ──┐
  │                       ├──► (no deps, all std + heapless)
  ├──────► mlp ───────────┘
  ├──────► ankle
  └──────► canproto
```

`mlp`, `ankle`, `canproto`, `hal` are pure Rust with no MCU dependency → host-testable.
`drivers` and `bin` require `cortex-m` and STM32 HAL → target-only.

### 5.3 Data Flow (per 4 ms control tick)

```
   ┌────────────────────────────────────────────────────────────────────┐
   │                          Embassy control_task (P3)                    │
   │                                                                    │
   │  1. lock joint_states (from CAN RX ISRs)                           │
   │  2. lock imu_state (from UART RX ISR)                             │
   │  3. lock action (from inference_task)                             │
   │  4. For each non-ankle motor (19):                                 │
   │       motor_cmd.pos = motor_sign[i] * action[motor2urdf[i]]       │
   │       motor_cmd.vel = 0                                            │
   │       motor_cmd.kp  = config.kp[i]                                │
   │       motor_cmd.kd  = config.kd[i]                                │
   │       motor_cmd.tau = 0                                            │
   │  5. For each ankle motor (4):                                       │
   │       a. fk_ankle(current motor angles, side)                      │
   │       b. compute joint torque: tau_j = kp·(q*−q) + kd·(0−v)       │
   │       c. compute motor torque: tau_m = J^T · tau_j                │
   │       d. motor_cmd.tau = motor_sign[i] * tau_m                    │
   │  6. For each of 23 motors:                                          │
   │       frame = canproto::encode_mit(motor_id, motor_cmd)           │
   │       bus.transmit(frame)                                          │
   │  7. release all locks                                              │
   └────────────────────────────────────────────────────────────────────┘

   ┌────────────────────────────────────────────────────────────────────┐
   │                          Embassy inference_task (P2)                   │
   │                                                                    │
   │  1. Build obs[78] from imu_state + joint_states + cmd_vel          │
   │  2. Clip to ±100                                                   │
   │  3. Push to ring buffer (10 frames)                               │
   │  4. Flatten to stacked[780]                                       │
   │  5. mlp.forward(stacked, &mut raw_action[23])                     │
   │  6. Post-process: clip, scale, add default, USD→URDF reorder      │
   │  7. Store result → action                                         │
   └────────────────────────────────────────────────────────────────────┘
```

---

## 6. Module API Contracts

### 6.1 `mlp` crate

Weights are embedded at compile time via `include_bytes!` (chosen for simplicity; no abstraction layer needed).

```rust
pub const INPUT_DIM: usize = 780;     // 78 obs × 10 frames
pub const OUTPUT_DIM: usize = 23;
pub const HIDDEN_DIMS: [usize; 3] = [512, 256, 128];

/// INT8 weights, row-major [out_dim × in_dim], loaded from weights/*.bin at compile time
pub mod weights {
    pub const W1_BYTES: &[u8] = include_bytes!("../../../weights/w1_int8.bin");
    pub const W2_BYTES: &[u8] = include_bytes!("../../../weights/w2_int8.bin");
    pub const W3_BYTES: &[u8] = include_bytes!("../../../weights/w3_int8.bin");
    pub const W4_BYTES: &[u8] = include_bytes!("../../../weights/w4_int8.bin");
    pub const B1_BYTES: &[u8] = include_bytes!("../../../weights/b1_int8.bin");
    pub const B2_BYTES: &[u8] = include_bytes!("../../../weights/b2_int8.bin");
    pub const B3_BYTES: &[u8] = include_bytes!("../../../weights/b3_int8.bin");
    pub const B4_BYTES: &[u8] = include_bytes!("../../../weights/b4_int8.bin");
}

/// Per-layer quantization scales (from weights/manifest.json)
/// Format: (w_scale, b_scale, out_scale)
pub const SCALES: [(f32, f32, f32); 4] = [
    (/* layer 1 scales from manifest */),
    (/* layer 2 */),
    (/* layer 3 */),
    (/* layer 4 */),
];

/// Hidden-layer buffer dimensions (compile-time constants, no allocation needed)
pub const fn hidden_buf(n: usize) -> usize { n }

pub struct Mlp;

impl Mlp {
    /// Forward pass: stacked_obs[780] → raw_action[23]
    /// Pre: stacked_obs contains 10 valid frames, 78 floats each
    /// Post: raw_action filled with pre-scale, pre-offset values (to be post-processed by caller)
    ///
    /// Stack usage: max(512, 256, 128) * 4 bytes = 2 KB
    pub fn forward(stacked_obs: &[f32; 780], raw_action: &mut [f32; 23]);
}
```

### 6.2 `ankle` crate

```rust
pub type Side = bool; // false = right, true = left

#[derive(Debug, Clone, Copy)]
pub struct AnkleParams {
    pub l_rod_long: f32,    // 180.0 mm
    pub l_rod_short: f32,   // 110.0 mm
    pub l_bar: f32,         // 20.0 mm
    pub l_spacing: f32,     // 42.35 mm (sign per side)
}

impl Default for AnkleParams {
    fn default() -> Self;  // matches decouple_atom01.cpp values
}

pub struct AnkleDecoupler {
    params: AnkleParams,
}

impl AnkleDecoupler {
    pub fn new(side: Side) -> Self;

    /// Inverse kinematics: joint angles (pitch, roll) → motor angles
    /// Returns (motor_long, motor_short) for the 2 actuators in the parallel mechanism
    /// Algorithm: closed-form solution via cosine rule (matches decouple_atom01.cpp:50-111)
    pub fn inverse_kinematics(
        &self,
        q_pitch: f32,   // target pitch (rad)
        q_roll: f32,    // target roll (rad)
    ) -> [f32; 2];    // [motor_long, motor_short]

    /// Forward kinematics: motor angles → joint (pitch, roll)
    /// Newton iteration (≤100 iter, tol 1e-3) on the 4-bar constraint
    /// Returns FkResult with joint angles, Jacobian, convergence info
    pub fn forward_kinematics(
        &self,
        motor_long: f32,   // long-link motor angle (rad)
        motor_short: f32,  // short-link motor angle (rad)
    ) -> FkResult;

    /// Compute Jacobian (2x2) at given joint pitch
    pub fn jacobian(&self, q_pitch: f32) -> [[f32; 2]; 2];

    /// Map joint torques to motor torques via Jacobian transpose
    pub fn joint_torque_to_motor(
        &self,
        tau_joint: [f32; 2],  // [pitch, roll]
        q_pitch: f32,
    ) -> [f32; 2];            // [motor_long, motor_short]

    /// Full step: PD in joint space + J^T mapping to motor torque
    pub fn compute_motor_torque(
        &self,
        q_target: [f32; 2],   // [pitch_target, roll_target]
        q_actual: [f32; 2],   // current joint angles from FK
        v_actual: [f32; 2],   // current joint velocities
        kp: [f32; 2],
        kd: [f32; 2],
    ) -> [f32; 2];
}

pub struct FkResult {
    pub pitch: f32,
    pub roll: f32,
    pub jacobian: [[f32; 2]; 2],
    pub iterations: u8,
    pub converged: bool,
}
```

### 6.3 `canproto` crate

```rust
pub const MIT_FRAME_DLC: u8 = 8;

#[derive(Debug, Clone, Copy)]
pub struct CanId(pub u16);

#[derive(Debug, Clone, Copy)]
pub struct CanFrame {
    pub id: CanId,
    pub dlc: u8,
    pub data: [u8; 8],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MotorModel { Dm4340P, Dm10010L }

impl MotorModel {
    pub fn pos_max(self) -> f32;       // 12.5
    pub fn spd_max(self) -> f32;       // 20.0 / 25.0
    pub fn tau_max(self) -> f32;       // 28.0 / 200.0
}

#[derive(Debug, Clone, Copy)]
pub struct MitCommand {
    pub pos: f32,    // rad
    pub vel: f32,    // rad/s (signed, range-mapped)
    pub kp: f32,     // [0, 500]
    pub kd: f32,     // [0, 5]
    pub tau: f32,    // Nm (signed)
}

/// Encode a MIT command into a 8-byte CAN frame for the given motor
pub fn encode_mit(motor_id: u8, model: MotorModel, cmd: &MitCommand) -> CanFrame;

/// Decode a CAN frame into MIT feedback (motor pos, vel, torque, temp, error)
pub fn decode_mit_feedback(frame: &CanFrame) -> Result<MitFeedback, DecodeError>;

#[derive(Debug, Clone, Copy)]
pub struct MitFeedback {
    pub motor_id: u8,
    pub position: f32,    // rad
    pub velocity: f32,    // rad/s
    pub torque: f32,      // Nm
    pub error_code: u8,
}

pub fn range_map(x: f32, in_min: f32, in_max: f32, out_min: u32, out_max: u32) -> u32;
```

### 6.4 `hal` crate

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

pub trait SpiFlash {
    fn read(&mut self, addr: u32, buf: &mut [u8]) -> Result<(), SpiError>;
    fn init(&mut self) -> Result<(), SpiError>;
}

#[cfg(any(test, feature = "host"))]
pub mod mock {
    pub struct MockCanBus { /* ... */ }
    pub struct MockSerial { /* ... */ }
}
```

---

## 7. Testing Strategy

### 7.1 Test Pyramid

| Level | Crates | Method | Speed |
|---|---|---|---|
| Unit (algorithm) | `mlp`, `ankle`, `canproto` | `cargo test` on host x86_64 | <1 s total |
| Unit (mock HAL) | `drivers` | `cargo test` on host x86_64 | <1 s |
| Cross-crate | `mlp` + `ankle` + `canproto` integration | `cargo test` on host x86_64 | <1 s |
| Target build | all | `cargo build --release --target thumbv7em-none-eabihf` | <60 s |
| QEMU runtime | `bin` | `qemu-system-arm -M stm32h743` | manual |
| Hardware smoke | `bin` | `probe-rs run` on real H743 | manual |

### 7.2 Reference Test Vectors

Test fixtures are checked-in JSON files (host-only, never compiled into firmware):

- `tests/fixtures/mlp_random_obs.json`: 100 random obs vectors + expected FP32 reference outputs (pre-generated by Python)
- `tests/fixtures/ankle_ik_convergence.json`: motor angles + expected joint angles
- `tests/fixtures/can_mit_encoding.json`: command floats + expected bytes (matches dm_motor_driver.cpp reference values)

### 7.3 CI Commands

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace  # all algorithm tests
cargo build --release --target thumbv7em-none-eabihf
```

---

## 8. Build & Deployment

### 8.1 Toolchain

```toml
# rust-toolchain.toml
[toolchain]
channel = "stable"
components = ["rustfmt", "clippy"]
targets = ["thumbv7em-none-eabihf"]
```

### 8.2 `.cargo/config.toml`

```toml
[build]
target = "thumbv7em-none-eabihf"
runner = "probe-rs run --chip STM32H743VITx"

[target.thumbv7em-none-eabihf]
runner = "probe-rs run --chip STM32H743VITx"
rustflags = ["-C", "link-arg=-Tlink.x", "-C", "link-arg=--nmagic"]
```

### 8.3 Build & Flash

```bash
# Build
cargo build --release

# Flash via ST-Link/V2-1 (on-board) or external probe
cargo run --release

# Or with probe-rs directly
probe-rs download --chip STM32H743VITx target/thumbv7em-none-eabihf/release/atom01-fw
probe-rs reset --chip STM32H743VITx
```

### 8.4 Logging

- Firmware uses `defmt` over RTT
- View live logs: `probe-rs rtt --chip STM32H743VITx`
- Default log level: `INFO`
- ISR context: `defmt::warn!` and above only (to avoid buffer overflow)

---

## 9. Risks & Mitigations

| # | Risk | Probability | Impact | Mitigation |
|---|---|---|---|---|
| R1 | INT8 quantization degrades policy accuracy | Medium | High | Phase 2 verifies <1% error; fallback FP32 with reduced batch |
| R2 | Embassy executor API churn | Low | Medium | Pin exact version; test on stable |
| R3 | 4 CAN buses not available on H743 | High | Medium | Document dual-bus fallback; defer 3rd/4th bus to SPI-MCP2515 |
| R4 | External SPI Flash too slow for runtime weight read | Medium | Medium | Load all weights to RAM at boot (~120 ms acceptable) |
| R5 | Ankle IK Newton solver diverges near joint limits | Low | Medium | Fallback to identity mapping + log warning |
| R6 | HiPNUC serial protocol mismatch | Low | High | Reference driver code in `atom01_deploy`; cross-check CRC |
| R7 | ST-Link firmware version incompatibility | Low | Low | Use OpenOCD fallback documented in README |
| R8 | RAM exceeds 1 MB after Phase 7 | Medium | High | Per-phase RSS measurement; defer SD card buffer to Phase 10 |

---

## 10. Open Questions → Resolved at Phase Boundaries

| # | Question | Default Decision | Revisit At |
|---|---|---|---|
| 1 | CAN bus count (4 buses native? SPI-MCP2515?) | **2-bus shared (CAN1+CAN2 native)** — see PLAN Phase 8.0 | Phase 8 start |
| 2 | External SPI Flash interface | **Standard SPI @ ≤50 MHz via SPI1** (W25Q64JV) | Phase 8 (driver impl) |
| 3 | IMU UART pinout | **USART4: PA0/PA1, 921600 baud** | Phase 7 (driver impl) |
| 4 | Boot mode | **Internal Flash only** (2 MB sufficient for firmware + constants; external SPI used as data storage, loaded to RAM at boot) | Phase 0 (linker script) |
| 5 | Watchdog behavior | **IWDG enabled, fed by idle task; 1s timeout → hard reset** | Phase 6 (init) |

Each row can be overridden during implementation if the user has a specific board wiring in mind. The defaults above are sensible for a generic STM32H743-NUCLEO-style carrier board.

---

## 11. References

| Topic | Source |
|---|---|
| Original C++ reference | `modules/atom01_deploy/src/inference/src/inference_node.cpp`, `robot_interface.cpp` |
| Ankle decoupling algorithm | `modules/atom01_deploy/src/inference/src/utils/decouple_atom01.cpp` |
| DM motor protocol | `modules/atom01_deploy/src/motors/src/drivers/dm/dm_motor_driver.cpp:358-413` |
| RL training (for network definition) | `modules/atom01_train/robolab/.../base_env.py:146-187` |
| CAN bus topology | `modules/atom01_deploy/src/inference/config/robot.yaml` |
| Embassy framework | https://embassy.dev/ |
| RTIC framework (alternative, not chosen) | https://rtic.rs/2/book/en/ |
| stm32h7xx-hal | https://docs.rs/stm32h7xx-hal/ |
| bxcan crate | https://docs.rs/bxcan/ |
| defmt logging | https://defmt.ferrous-systems.com/ |

---

## 12. Approval

This spec is **DRAFT** pending user review. Phase 0 implementation will not begin until user explicitly approves this document or provides revisions.
