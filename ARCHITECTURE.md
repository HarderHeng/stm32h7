# Project Architecture

## Layer Map

```
┌──────────────────────────────────────────────────────┐
│  bin/main.rs           Embassy USB-CDC shell          │  ← Application
├──────────────────────────────────────────────────────┤
│  pipeline.rs           Inference + control glue       │  ← Integration
├────────────────┬─────────────────────────────────────┤
│  observation.rs  obs ring buffer + frame stack        │
│  mlp.rs          INT8 4-layer MLP forward             │  ← Algorithm layer
│  ankle.rs        IK/FK/Jacobian for ankle mechanism   │     (no_std, no deps)
│  canproto.rs     DAMIAO motor MIT frame encode/decode │
├────────────────┼─────────────────────────────────────┤
│  config.rs          Constants (robot.yaml + inference.yaml) │  ← Config
├──────────────────────────────────────────────────────┤
│  hal.rs             Traits (CanBus/SerialPort/SpiFlash) + Mock │ ← HAL
├──────────────────────────────────────────────────────┤
│  drivers.rs         Hardware driver stubs             │  ← Drivers
│                     (embassy_stm32 impl placeholders) │     (target only)
└──────────────────────────────────────────────────────┘
```

## Module Contracts

### Algorithm Layer (host-testable in principle)

| Module | Input | Output | Dependencies |
|--------|-------|--------|-------------|
| `mlp` | `&[f32; 780]` stacked obs | `&mut [f32; 23]` raw action | `config` (scales, reorder), `weights/` (binaries) |
| `ankle` | motor angles `[f32; 2]` | `FkResult` (joint angles + Jacobian) | nothing |
| `canproto` | `MitCommand` + motor ID | `CanFrame` (8 bytes) | nothing |
| `observation` | `&[f32; 78]` per frame | `&[f32; 780]` flattened 10-frame stack | `config` (FRAME_STACK, OBS_DIM) |

### Driver Layer (target only, embassy-stm32)

| Module | Trait | Current State |
|--------|-------|--------------|
| `drivers` | `CanBus`, `SerialPort`, `SpiFlash` | **Stubs**. Bodies return errors. |

### Integration Layer

| Module | Responsibility | Current State |
|--------|---------------|--------------|
| `pipeline` | `step_inference()` @ 50Hz, `step_control()` @ 4ms, Embassy async task wrappers | **Full structure.** Bodies compile. |

## Data Flow (per 4ms control tick)

```
IMU (500Hz)                  Motor encoders (250Hz)
    │                              │
    └──────┬───────────────────────┘
           ↓
   pipeline::step_inference()     [every 5th tick = 50Hz]
      observation.push()          [78 floats → ring buffer]
      observation.flatten_into()  [780 floats]
      Mlp::forward_int8()         [23 raw action]
      Mlp::post_process()         [clip, scale, USD→URDF, +default]
           ↓
   pipeline::step_control()       [every tick = 250Hz]
      for each of 23 motors:
         MIT frame = canproto::encode_mit(motor_id, model, cmd)
         dispatch to correct CAN bus
           ↓
   CAN TX ISR → Motor feedback → CAN RX ISR
```

## Next Phase: Driver Bodies + Embassy Spawn

What's needed to go from "compiles" to "controls hardware":

1. **Fill in driver bodies** — replace `drivers.rs` stubs with real embassy-stm32 peripherals:
   - `CanBusHw` → wrap `embassy_stm32::can::Can` (bxCAN)
   - `ImuSerial` → wrap `embassy_stm32::usart::Uart` (USART4, 921600 baud)
   - `W25Q64` → wrap `embassy_stm32::spi::Spi` (SPI1)

2. **Spawn tasks in main.rs** — replace USB shell with real control:
   ```rust
   #[embassy_executor::main]
   async fn main(spawner: Spawner) {
       let mut pipeline = Pipeline::new();
       // spawn inference loop
       spawner.spawn(clocked_inference(pipeline)).unwrap();
       // spawn control loop
       spawner.spawn(clocked_control(pipeline)).unwrap();
   }
   ```

3. **Real weight loading** — re-run `tools/quantize_onnx.py --input .../policy.onnx --strict`
