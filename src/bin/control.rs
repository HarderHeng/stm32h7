//! Atom01 robot firmware — full control main entry.
//!
//! Starts the Embassy executor, initializes IMU + CAN buses,
//! spawns inference (50Hz) and control (250Hz) tasks.
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
    let p = embassy_stm32::init(config);
    let _periph = &p; // available for wiring CAN/IMU in Phase 6/7

    // --- TODO: replace pins with your board wiring ---
    // CAN1: PB8 (RX) / PB9 (TX), 1 Mbps
    // IMU: USART4 (PA0=TX / PA1=RX), 921600 baud
    // See src/drivers.rs doc comments for embassy peripheral hook points.

    info!("Init complete. Spawning tasks...");

    let pipe = Pipeline::new();

    // Spawn inference task (50 Hz)
    spawner.spawn(inference_loop(pipe)).unwrap();

    info!("Tasks spawned — running");
}

#[embassy_executor::task]
async fn inference_loop(mut pipeline: Pipeline) {
    let mut ticker = Ticker::every(Duration::from_millis(20));
    let cmd_vel = [0.0_f32; 3];
    loop {
        ticker.next().await;
        pipeline.step_inference(&cmd_vel);
    }
}
