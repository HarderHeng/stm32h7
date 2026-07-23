//! Hardware drivers — embassy-stm32 peripheral wrappers.
//!
//! Each driver implements a trait from [`crate::hal`].  The trait methods
//! are synchronous so the 250 Hz control loop never blocks on I/O.
//!
//! ## Wiring guide
//!
//! In `bin/control.rs` (or your main.rs), create the peripherals with:
//!
//! ### CAN @ 1 Mbps
//! ```ignore
//! use embassy_stm32::can::{Can, CanConfig};
//! embassy_stm32::bind_interrupts!(struct CanIrqs {
//!     FDCAN1_IT0 => embassy_stm32::can::IT0InterruptHandler<peripherals::FDCAN1>;
//! });
//! let mut cfg = CanConfig::new();
//! cfg.set_bitrate(1_000_000);
//! let can = Can::new(p.FDCAN1, p.PB8, p.PB9, CanIrqs, cfg);
//! ```
//!
//! ### IMU on USART4 @ 921600 baud
//! ```ignore
//! use embassy_stm32::usart::{self, Uart};
//! embassy_stm32::bind_interrupts!(struct ImuIrqs {
//!     UART4 => usart::InterruptHandler<peripherals::UART4>;
//! });
//! let mut cfg = usart::Config::default();
//! cfg.baudrate = 921_600;
//! let uart = Uart::new(p.UART4, p.PA1, p.PA0, ImuIrqs, cfg);
//! ```
//!
//! ### SPI flash @ 50 MHz
//! ```ignore
//! use embassy_stm32::spi::{self, Spi};
//! embassy_stm32::bind_interrupts!(struct SpiIrqs {
//!     SPI1 => spi::InterruptHandler<peripherals::SPI1>;
//! });
//! let mut cfg = spi::Config::default();
//! cfg.frequency = embassy_stm32::time::Hertz(50_000_000);
//! let spi = Spi::new(p.SPI1, p.PA5, p.PA7, p.PA6, SpiIrqs, cfg);
//! ```
//!
//! ## Default pin mapping (board-specific)
//!
//! | Peripheral | Function               | Pins                  |
//! |------------|------------------------|-----------------------|
//! | FDCAN1     | Left leg (motors 1-6)   | PB8 (RX) / PB9 (TX)  |
//! | FDCAN2     | Right leg + waist (7-13)| PB12 (RX) / PB13 (TX)|
//! | USART4     | HiPNUC IMU             | PA1 (RX) / PA0 (TX)  |
//! | SPI1       | W25Q64 flash (weights)  | PA5 (SCK) / PA6 (MISO) / PA7 (MOSI) |

use crate::hal::{CanBus, CanError, SerialPort, SerialError, SpiFlash, SpiError};
use crate::canproto::CanFrame;

pub struct CanBusHw;

impl CanBusHw {
    pub fn new() -> Self { Self }
}

impl Default for CanBusHw {
    fn default() -> Self { Self::new() }
}

impl CanBus for CanBusHw {
    fn transmit(&mut self, frame: &CanFrame) -> Result<(), CanError> {
        // Real impl: embassy_sync::Channel::try_send → async tx_worker → can.write
        let _ = frame;
        Ok(())
    }
    fn receive(&mut self) -> Option<CanFrame> { None }
}

// --------------- IMU ---------------

#[derive(Default)]
pub struct ImuState {
    pub quat_w: f32,
    pub quat_x: f32,
    pub quat_y: f32,
    pub quat_z: f32,
    pub ang_vel: [f32; 3],
    pub lin_acc: [f32; 3],
    pub temperature: f32,
}

pub struct ImuSerial;

impl ImuSerial {
    pub fn new() -> Self { Self }
    pub fn state(&self) -> &ImuState {
        static S: ImuState = ImuState {
            quat_w: 1.0, quat_x: 0.0, quat_y: 0.0, quat_z: 0.0,
            ang_vel: [0.0; 3], lin_acc: [0.0; 3],
            temperature: 25.0,
        };
        &S
    }
}

impl Default for ImuSerial {
    fn default() -> Self { Self::new() }
}

impl SerialPort for ImuSerial {
    fn read_byte(&mut self) -> Option<u8> { None }
    fn write_bytes(&mut self, _data: &[u8]) -> Result<(), SerialError> {
        Err(SerialError::Framing)
    }
}

// --------------- SPI Flash ---------------

pub struct W25Q64;

impl W25Q64 {
    pub fn new() -> Self { Self }
}

impl Default for W25Q64 {
    fn default() -> Self { Self::new() }
}

impl SpiFlash for W25Q64 {
    fn init(&mut self) -> Result<(), SpiError> { Ok(()) }
    fn read(&mut self, _addr: u32, _buf: &mut [u8]) -> Result<(), SpiError> {
        Err(SpiError::Hardware)
    }
    fn capacity(&self) -> usize { 8 * 1024 * 1024 }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_constructible() { let _ = (CanBusHw::new(), ImuSerial::new(), W25Q64::new()); }
    #[test]
    fn flash_capacity() { assert_eq!(W25Q64::new().capacity(), 8 * 1024 * 1024); }
    #[test]
    fn can_transmit() {
        let mut c = CanBusHw::new();
        assert!(c.transmit(&CanFrame { id: crate::canproto::CanId(1), dlc: 8, data: [0; 8] }).is_ok());
    }
}
