//! Hardware abstraction layer traits.
//!
//! All real hardware drivers implement these traits. Host-side tests use
//! the mock implementations in [`mock`] to run algorithm crates without
//! any MCU dependency.
//!
//! Reference: `modules/atom01_deploy/src/motors/src/drivers/dm/dm_motor_driver.cpp`
//! and `modules/atom01_deploy/src/inference/src/robot_interface.cpp`.

use crate::canproto::CanFrame;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanError {
    /// TX FIFO full / arbitration lost
    BusBusy,
    /// Hardware bus-off, requires recovery
    BusOff,
    /// Other hardware error
    Hardware,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerialError {
    /// Parity, framing, or overrun
    Framing,
    /// Buffer full
    Overflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpiError {
    /// SPI peripheral error
    Hardware,
    /// Flash responded with error status
    FlashStatus,
    /// Device ID mismatch (wrong chip)
    WrongDeviceId,
}

/// CAN bus abstraction. Implemented by:
/// - `drivers::can::CanBusHw` for real bxCAN hardware
/// - `mock::MockCanBus` for host-side tests
pub trait CanBus {
    fn transmit(&mut self, frame: &CanFrame) -> Result<(), CanError>;
    fn receive(&mut self) -> Option<CanFrame>;
}

/// Asynchronous serial port abstraction. Implemented by:
/// - `drivers::imu::ImuSerial` for real USART4
/// - `mock::MockSerial` for host-side tests
pub trait SerialPort {
    /// Read one byte if available, non-blocking.
    fn read_byte(&mut self) -> Option<u8>;
    /// Write bytes to the port, may queue for DMA.
    fn write_bytes(&mut self, data: &[u8]) -> Result<(), SerialError>;
}

/// External SPI Flash abstraction. Implemented by:
/// - `drivers::spi_flash::W25Q64` for real hardware
/// - `mock::MockSpiFlash` for host-side tests
pub trait SpiFlash {
    fn init(&mut self) -> Result<(), SpiError>;
    fn read(&mut self, addr: u32, buf: &mut [u8]) -> Result<(), SpiError>;
    fn capacity(&self) -> usize;
}

pub mod mock {
    //! Mock HAL implementations for host-side testing.
    //!
    //! Each mock maintains an in-memory state and records every call.
    //! Tests can inspect recorded calls via the inspect API.

    use super::*;
    use heapless::Vec;

    /// Mock CAN bus that records every transmitted frame.
    pub struct MockCanBus {
        pub tx_log: Vec<CanFrame, 256>,
        pub rx_queue: Vec<CanFrame, 256>,
        pub bus_off: bool,
    }

impl Default for MockCanBus {
    fn default() -> Self { Self::new() }
}

impl MockCanBus {
    pub const fn new() -> Self {
        Self {
            tx_log: Vec::new(),
            rx_queue: Vec::new(),
            bus_off: false,
        }
    }

        /// Inject a frame that `receive()` will return.
        pub fn inject_rx(&mut self, frame: CanFrame) {
            let _ = self.rx_queue.push(frame);
        }

        pub fn tx_count(&self) -> usize {
            self.tx_log.len()
        }
    }

    impl CanBus for MockCanBus {
        fn transmit(&mut self, frame: &CanFrame) -> Result<(), CanError> {
            if self.bus_off {
                return Err(CanError::BusOff);
            }
            self.tx_log.push(*frame).map_err(|_| CanError::BusBusy)
        }

        fn receive(&mut self) -> Option<CanFrame> {
            if self.rx_queue.is_empty() {
                None
            } else {
                Some(self.rx_queue.remove(0))
            }
        }
    }

    /// Mock serial port backed by a byte buffer.
    pub struct MockSerial {
        pub rx_buffer: Vec<u8, 1024>,
        pub tx_log: Vec<u8, 1024>,
    }

impl Default for MockSerial {
    fn default() -> Self { Self::new() }
}

impl MockSerial {
    pub const fn new() -> Self {
        Self {
            rx_buffer: Vec::new(),
            tx_log: Vec::new(),
        }
    }

        pub fn inject_rx(&mut self, data: &[u8]) {
            for &b in data {
                let _ = self.rx_buffer.push(b);
            }
        }
    }

    impl SerialPort for MockSerial {
        fn read_byte(&mut self) -> Option<u8> {
            if self.rx_buffer.is_empty() {
                None
            } else {
                Some(self.rx_buffer.remove(0))
            }
        }

        fn write_bytes(&mut self, data: &[u8]) -> Result<(), SerialError> {
            for &b in data {
                self.tx_log.push(b).map_err(|_| SerialError::Overflow)?;
            }
            Ok(())
        }
    }

    /// Mock SPI Flash backed by a fixed-size byte buffer.
    pub struct MockSpiFlash {
        pub data: Vec<u8, 8192>,
        pub initialized: bool,
    }

impl Default for MockSpiFlash {
    fn default() -> Self { Self::new() }
}

impl MockSpiFlash {
    pub const fn new() -> Self {
        Self {
            data: Vec::new(),
            initialized: false,
        }
    }

        pub fn preload(&mut self, data: &[u8]) {
            self.data.clear();
            for &b in data {
                let _ = self.data.push(b);
            }
        }
    }

    impl SpiFlash for MockSpiFlash {
        fn init(&mut self) -> Result<(), SpiError> {
            self.initialized = true;
            Ok(())
        }

        fn read(&mut self, addr: u32, buf: &mut [u8]) -> Result<(), SpiError> {
            // Use checked arithmetic so a hostile / buggy addr doesn't
            // silently wrap around and read from offset 0.
            let start = addr as usize;
            let end = match start.checked_add(buf.len()) {
                Some(e) => e,
                None => return Err(SpiError::Hardware),
            };
            if end > self.data.len() {
                return Err(SpiError::Hardware);
            }
            buf.copy_from_slice(&self.data[start..end]);
            Ok(())
        }

        fn capacity(&self) -> usize {
            self.data.len()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn mock_can_records_tx() {
            let mut bus = MockCanBus::new();
            let frame = crate::canproto::CanFrame {
                id: crate::canproto::CanId(1),
                dlc: 8,
                data: [0; 8],
            };
            bus.transmit(&frame).unwrap();
            assert_eq!(bus.tx_count(), 1);
        }

        #[test]
        fn mock_serial_round_trip() {
            let mut s = MockSerial::new();
            s.inject_rx(b"hello");
            assert_eq!(s.read_byte(), Some(b'h'));
            assert_eq!(s.read_byte(), Some(b'e'));
            s.write_bytes(b"ok").unwrap();
            assert_eq!(s.tx_log.as_slice(), b"ok");
        }

        #[test]
        fn mock_spi_rejects_overflow_addr() {
            // addr + buf.len() must not wrap around on 32-bit platforms.
            // Without checked_add, addr=u32::MAX would silently read from
            // offset 0 (or panic in release mode without checks).
            let mut flash = MockSpiFlash::new();
            flash.preload(&[1, 2, 3, 4]);
            let mut buf = [0u8; 8];
            assert_eq!(flash.read(u32::MAX, &mut buf), Err(SpiError::Hardware));
            // And a legitimate small read still works.
            let mut buf2 = [0u8; 2];
            assert_eq!(flash.read(1, &mut buf2), Ok(()));
            assert_eq!(buf2, [2, 3]);
        }
    }
}
