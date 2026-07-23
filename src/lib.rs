//! Atom01 robot firmware library root.
//!
//! Module layout:
//! - `hal`: Hardware abstraction traits (CanBus, SerialPort, SpiFlash) + mock impls
//! - `mlp`: Pure-Rust MLP forward pass for the RL policy (host-testable)
//! - `ankle`: Pure-Rust 4-bar parallel-mechanism decoupling for ankle joints (host-testable)
//! - `canproto`: Pure-Rust DAMIAO motor MIT frame encode/decode (host-testable)
//! - `drivers`: Real hardware driver implementations (no_std, target-only)

#![no_std]

pub mod hal;
pub mod mlp;
pub mod ankle;
pub mod canproto;
pub mod drivers;
pub mod observation;
pub mod pipeline;

pub mod config;

// Re-exports for ergonomic `use atom01_fw::Mlp` style usage
pub use ankle::{AnkleDecoupler, AnkleParams, FkResult, JacobianResult, Side};
pub use canproto::{CanFrame, CanId, DecodeError, MitCommand, MitFeedback, MotorModel};
pub use hal::{CanBus, CanError, SerialError, SerialPort, SpiError, SpiFlash};
pub use mlp::{Mlp, OUTPUT_DIM};
