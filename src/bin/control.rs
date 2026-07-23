//! Atom01 robot firmware — control mode main entry.
//!
//! Starts the Embassy executor, configures the clock tree, and spawns
//! the inference loop. The control loop is intentionally NOT spawned
//! yet because it requires real Embassy driver wiring (CAN, IMU, SPI
//! flash) that lives in `src/drivers.rs` as stubs. When the drivers
//! are implemented, add a `control_task` here per the architecture doc.
//!
//! **Pin mapping is board-specific** — adjust in `init()` before flashing.

#![no_std]
#![no_main]

use defmt::*;
use embassy_executor::Spawner;
use embassy_stm32::Config;
use embassy_time::{Duration, Ticker};
use {defmt_rtt as _, panic_probe as _};

use embassy_stm32h7::pipeline::Pipeline;

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    info!("Atom01 firmware — control mode");
    info!("  target: STM32H743");
    info!("  framework: Embassy async");

    let mut config = Config::default();
    {
        use embassy_stm32::rcc::*;
        config.rcc.hsi = Some(HSIPrescaler::DIV1);
        config.rcc.csi = true;
        config.rcc.pll1 = Some(Pll {
            source: PllSource::HSI,
            prediv: PllPreDiv::DIV4,
            mul: PllMul::MUL50,
            fracn: None,
            divp: Some(PllDiv::DIV2),
            divq: None,
            divr: None,
        });
        config.rcc.sys = Sysclk::PLL1_P;
        config.rcc.ahb_pre = AHBPrescaler::DIV2;
        config.rcc.apb1_pre = APBPrescaler::DIV2;
        config.rcc.apb2_pre = APBPrescaler::DIV2;
        config.rcc.apb3_pre = APBPrescaler::DIV2;
        config.rcc.apb4_pre = APBPrescaler::DIV2;
        config.rcc.voltage_scale = VoltageScale::Scale1;
    }
    let _p = embassy_stm32::init(config);

    // --- TODO: wire real Embassy peripherals ---
    // CAN1: PB8 (RX) / PB9 (TX), 1 Mbps
    // IMU: USART4 (PA0=TX / PA1=RX), 921600 baud
    // See src/drivers.rs doc comments for embassy peripheral hook points.

    info!("Init complete. Spawning inference task...");

    let pipe = Pipeline::new();

    // Spawn inference task (50 Hz). Control task (250 Hz) is deferred
    // until Phase 6/7 driver wiring lands — step_control() would dispatch
    // MIT frames to the (still-stub) CAN bus.
    spawner.spawn(inference_loop(pipe)).unwrap();

    info!("Inference task running. cmd_vel is a placeholder (zero) until");
    info!("Phase 7 wires a real source (USB shell / ROS2 / wired remote).");
}

#[embassy_executor::task]
async fn inference_loop(mut pipeline: Pipeline) {
    let mut ticker = Ticker::every(Duration::from_millis(20));
    // TODO: replace with a shared channel fed by the control source
    // (USB shell command, ROS2 topic, or wired remote).
    let cmd_vel = [0.0_f32; 3];
    loop {
        ticker.next().await;
        pipeline.step_inference(&cmd_vel);
    }
}
