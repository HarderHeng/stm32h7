//! DAMIAO (DM) motor MIT frame encode/decode.
//!
//! Reference C++ source:
//! - `modules/atom01_deploy/src/motors/src/drivers/dm/dm_motor_driver.cpp:358-413`
//!
//! MIT frame format (8 bytes, CAN 2.0):
//!   data[0..1] = pos  (16-bit unsigned, ±12.5 rad)
//!   data[2..3] = vel  (12-bit unsigned, ±20/25 rad/s)
//!   data[3..4] = kp   (12-bit unsigned, [0, 500])
//!   data[5..6] = kd   (12-bit unsigned, [0, 5])
//!   data[6..7] = tau  (12-bit unsigned, ±28/200 Nm)
//!
//! can_id = motor_id (1..23).

/// CAN identifier. Wraps `u16` to prevent accidental mixing with other ints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanId(pub u16);

/// 8-byte CAN data frame (CAN 2.0, standard or extended ID).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanFrame {
    pub id: CanId,
    pub dlc: u8,
    pub data: [u8; 8],
}

/// Motor model — determines physical ranges used by `range_map`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotorModel {
    /// DM 4340P (hip yaw/roll/pitch, knee, waist) — 14 units on Atom01.
    Dm4340P,
    /// DM 10010L (ankle pitch/roll, shoulder, elbow) — 9 units on Atom01.
    Dm10010L,
}

impl MotorModel {
    pub const fn pos_max(self) -> f32 {
        12.5
    }
    pub const fn spd_max(self) -> f32 {
        match self {
            MotorModel::Dm4340P => 20.0,
            MotorModel::Dm10010L => 25.0,
        }
    }
    pub const fn tau_max(self) -> f32 {
        match self {
            MotorModel::Dm4340P => 28.0,
            MotorModel::Dm10010L => 200.0,
        }
    }
}

/// MIT control command for one motor.
#[derive(Debug, Clone, Copy)]
pub struct MitCommand {
    /// Target position (rad). Range: ±`pos_max()`.
    pub pos: f32,
    /// Target velocity (rad/s). Used as velocity feed-forward in MIT law.
    /// Range: ±`spd_max()`.
    pub vel: f32,
    /// Position gain. Range: [0, 500].
    pub kp: f32,
    /// Velocity gain. Range: [0, 5].
    pub kd: f32,
    /// Torque feed-forward (Nm). Range: ±`tau_max()`.
    pub tau: f32,
}

/// Feedback frame decoded from motor response (can_id = motor_id + 16).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MitFeedback {
    pub motor_id: u8,
    pub position: f32, // rad
    pub velocity: f32, // rad/s
    pub torque: f32,   // Nm
    pub error_code: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// Wrong CAN ID (not motor_id + 16)
    WrongCanId,
    /// Wrong DLC (not 8)
    WrongDlc,
}

/// Linear range map: `x ∈ [in_min, in_max]` → `[out_min, out_max]` (saturating).
/// Reproduces the C++ `range_map` in dm_motor_driver.cpp:36-40.
///
/// Uses round-half-up via `(raw + 0.5) as u32` to avoid `libm` dependency
/// in `no_std`. Since `raw >= 0`, this matches round-half-away-from-zero
/// for our domain. `raw` is always in `[out_min, out_max]`, so the cast
/// never saturates.
#[inline]
pub fn range_map(x: f32, in_min: f32, in_max: f32, out_min: u32, out_max: u32) -> u32 {
    let x = clip_f32(x, in_min, in_max);
    let t = (x - in_min) / (in_max - in_min);
    let span = (out_max - out_min) as f32;
    let raw = out_min as f32 + t * span;
    (raw + 0.5) as u32
}

#[inline]
fn clip_f32(x: f32, lo: f32, hi: f32) -> f32 {
    if x.is_nan() { (lo + hi) * 0.5 } else if x < lo { lo } else if x > hi { hi } else { x }
}

/// Encode a MIT command into an 8-byte CAN frame for the given motor ID.
///
/// Reference: `dm_motor_driver.cpp:365-409` (the exact bit layout used by
/// DAMIAO firmware).
pub fn encode_mit(motor_id: u8, model: MotorModel, cmd: &MitCommand) -> CanFrame {
    let pos_max = model.pos_max();
    let spd_max = model.spd_max();
    let tau_max = model.tau_max();

    let p = range_map(cmd.pos, -pos_max, pos_max, 0, 0xFFFF) as u16;
    let v = range_map(cmd.vel, -spd_max, spd_max, 0, 0x0FFF) as u16;
    let kp = range_map(cmd.kp, 0.0, 500.0, 0, 0x0FFF) as u16;
    let kd = range_map(cmd.kd, 0.0, 5.0, 0, 0x0FFF) as u16;
    let t = range_map(cmd.tau, -tau_max, tau_max, 0, 0x0FFF) as u16;

    CanFrame {
        id: CanId(motor_id as u16),
        dlc: 8,
        data: [
            (p >> 8) as u8,
            p as u8,
            (v >> 4) as u8,
            (((v & 0x000F) << 4) as u8) | ((kp >> 8) & 0x000F) as u8,
            kp as u8,
            (kd >> 4) as u8,
            (((kd & 0x000F) << 4) as u8) | ((t >> 8) & 0x000F) as u8,
            t as u8,
        ],
    }
}

/// Decode a feedback CAN frame (can_id = motor_id + 16) into MitFeedback.
///
/// DAMIAO MIT feedback layout (8 bytes):
///   byte[0..2]: position (16-bit, ±12.566 rad)
///   byte[2]:    velocity top 8 bits (of 12-bit velocity)
///   byte[3]:    velocity low nibble | torque high nibble
///   byte[4]:    torque low 8 bits (of 12-bit torque)
///   byte[5]:    T_mos (8 bits)
///   byte[6]:    T_rotor (8 bits)
///   byte[7]:    error code (high nibble)
pub fn decode_mit_feedback(frame: &CanFrame) -> Result<MitFeedback, DecodeError> {
    if frame.dlc != 8 {
        return Err(DecodeError::WrongDlc);
    }
    // Feedback uses can_id = motor_id + master_id_offset (16).
    // Atom01 has 23 motors, so valid can_id range is 16..=39.
    if !(16..=16 + 23).contains(&frame.id.0) {
        return Err(DecodeError::WrongCanId);
    }
    let motor_id = (frame.id.0 - 16) as u8;

    let p_raw = ((frame.data[0] as u16) << 8) | (frame.data[1] as u16);
    let v_raw = (((frame.data[2] as u16) << 4) | ((frame.data[3] as u16) >> 4)) & 0x0FFF;
    let t_raw = (((frame.data[3] as u16) << 8) | (frame.data[4] as u16)) & 0x0FFF;
    let error_code = (frame.data[7] >> 4) & 0x0F;

    let position = (p_raw as f32 / 0xFFFF as f32 - 0.5) * 2.0 * 12.5;
    let velocity = (v_raw as f32 / 0x0FFF as f32 - 0.5) * 2.0 * 20.0;
    let torque = (t_raw as f32 / 0x0FFF as f32 - 0.5) * 2.0 * 28.0;

    Ok(MitFeedback {
        motor_id,
        position,
        velocity,
        torque,
        error_code,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_map_zero_maps_to_midpoint() {
        // 0 in [-12.5, 12.5] → (0 + 12.5) / 25 * 0xFFFF = 0x7FFF8 ≈ 0x7FFF8 → 0x7FFF8 round to 0x7FFF8
        let v = range_map(0.0, -12.5, 12.5, 0, 0xFFFF);
        assert!((v as i32 - 0x7FFF) <= 1);
    }

    #[test]
    fn range_map_extremes() {
        assert_eq!(range_map(12.5, -12.5, 12.5, 0, 0xFFFF), 0xFFFF);
        assert_eq!(range_map(-12.5, -12.5, 12.5, 0, 0xFFFF), 0);
    }

    #[test]
    fn range_map_saturates_outside_range() {
        assert_eq!(range_map(100.0, -12.5, 12.5, 0, 0xFFFF), 0xFFFF);
        assert_eq!(range_map(-100.0, -12.5, 12.5, 0, 0xFFFF), 0);
    }

    #[test]
    fn encode_zero_command_yields_midpoint_bits() {
        let cmd = MitCommand { pos: 0.0, vel: 0.0, kp: 0.0, kd: 0.0, tau: 0.0 };
        let f = encode_mit(1, MotorModel::Dm4340P, &cmd);
        assert_eq!(f.dlc, 8);
        assert_eq!(f.id, CanId(1));
        // pos=0 → 0x7FFF, vel/kp/kd/tau=0 → 0
        assert_eq!(u16::from_be_bytes([f.data[0], f.data[1]]), 0x7FFF);
        assert_eq!(f.data[2], 0x00); // vel high byte
        assert_eq!(f.data[7], 0x00); // tau low byte
    }

    #[test]
    fn encode_nan_command_becomes_safe_midpoint() {
        // NaN must NOT propagate through range_map/clip_f32 — instead it
        // falls back to the midpoint of the field's valid range, so a motor
        // command carrying NaN becomes "go to home pose with zero control"
        // rather than an extreme position that fights the motor's PD law.
        let cmd = MitCommand { pos: f32::NAN, vel: f32::NAN, kp: f32::NAN, kd: f32::NAN, tau: f32::NAN };
        let f = encode_mit(1, MotorModel::Dm4340P, &cmd);
        // Midpoint encoding rounds up: 0.5 * 0xFFFF = 32767.5 → 0x8000.
        let p = u16::from_be_bytes([f.data[0], f.data[1]]);
        assert!((p as i32 - 0x7FFF).abs() <= 1, "pos got 0x{:04X}", p);
    }

    #[test]
    fn encode_known_reference_values() {
        // Hand-computed: pos=1.0, vel=0.0, kp=100, kd=3.3, tau=0.0 for DM4340P
        // Expected bits:
        //   pos: round((1.0 + 12.5) / 25 * 65535) = round(54090.75) = 54091 = 0xD343
        //   vel: 0 → 0
        //   kp:  round(100 / 500 * 4095) = round(819) = 819 = 0x333
        //   kd:  round(3.3 / 5 * 4095) = round(2702.7) = 2703 = 0xA8F
        //   tau: 0 → 0
        let cmd = MitCommand { pos: 1.0, vel: 0.0, kp: 100.0, kd: 3.3, tau: 0.0 };
        let f = encode_mit(1, MotorModel::Dm4340P, &cmd);
        let p = u16::from_be_bytes([f.data[0], f.data[1]]);
        assert_eq!(p, 0xD343, "pos encoding mismatch: got 0x{:04X}", p);
        let kp_high = (f.data[3] & 0x0F) as u16;
        let kp_low = f.data[4] as u16;
        assert_eq!((kp_high << 8) | kp_low, 0x333, "kp encoding mismatch");
    }

    #[test]
    fn decode_feedback_extremes() {
        // pos=12.5, vel=20, tau=28 (DM4340P maxes), error=0.
        // byte[2]=0xFF (vel top 8 of 12-bit 0xFFF), byte[3]=0xFF (vel_lo=0xF<<4 | tau_top=0xF),
        // byte[4]=0xFF (tau lo 8 bits).
        let frame = CanFrame {
            id: CanId(1 + 16),
            dlc: 8,
            data: [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00],
        };
        let fb = decode_mit_feedback(&frame).unwrap();
        assert_eq!(fb.motor_id, 1);
        assert!((fb.position - 12.5).abs() < 1e-3);
        assert!((fb.velocity - 20.0).abs() < 1e-3);
        assert!((fb.torque - 28.0).abs() < 1e-3);
        assert_eq!(fb.error_code, 0);
    }

    #[test]
    fn decode_feedback_minimum_extremes() {
        let frame = CanFrame {
            id: CanId(7 + 16),
            dlc: 8,
            data: [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        };
        let fb = decode_mit_feedback(&frame).unwrap();
        assert_eq!(fb.motor_id, 7);
        assert!((fb.position - (-12.5)).abs() < 1e-3);
        assert!((fb.velocity - (-20.0)).abs() < 1e-3);
        assert!((fb.torque - (-28.0)).abs() < 1e-3);
    }

    #[test]
    fn decode_feedback_zero_signals() {
        // pos=0 (16-bit midpoint 0x7FFF), vel=0 (12-bit midpoint 0x800), tau=0.
        // byte[2] = (0x800 >> 4) & 0xFF = 0x80.
        // byte[3] = ((0x800 & 0xF) << 4) | ((0x800 >> 8) & 0xF) = 0x08.
        // byte[4] = 0x800 & 0xFF = 0x00.
        let frame = CanFrame {
            id: CanId(3 + 16),
            dlc: 8,
            data: [0x7F, 0xFF, 0x80, 0x08, 0x00, 0x00, 0x00, 0x00],
        };
        let fb = decode_mit_feedback(&frame).unwrap();
        assert_eq!(fb.motor_id, 3);
        // Quantization bias at midpoint: ~0.005 rad/s, ~0.007 Nm.
        assert!(fb.position.abs() < 1e-2, "pos got {}", fb.position);
        assert!(fb.velocity.abs() < 1e-2, "vel got {}", fb.velocity);
        assert!(fb.torque.abs() < 1e-2, "tau got {}", fb.torque);
    }

    #[test]
    fn decode_feedback_known_position_half() {
        // pos=6.25: pos_raw = round((6.25+12.5)/25 * 65535) = 0xBFFF.
        let frame = CanFrame {
            id: CanId(2 + 16),
            dlc: 8,
            data: [0xBF, 0xFF, 0x80, 0x08, 0x00, 0x00, 0x00, 0x00],
        };
        let fb = decode_mit_feedback(&frame).unwrap();
        assert!((fb.position - 6.25).abs() < 1e-3);
        assert!(fb.velocity.abs() < 1e-2);
        assert!(fb.torque.abs() < 1e-2);
    }

    #[test]
    fn decode_feedback_error_code_in_byte7_high_nibble() {
        let frame = CanFrame {
            id: CanId(5 + 16),
            dlc: 8,
            data: [0x7F, 0xFF, 0x80, 0x08, 0x00, 0x00, 0x00, 0xA0],
        };
        let fb = decode_mit_feedback(&frame).unwrap();
        assert_eq!(fb.error_code, 0xA, "error_code got 0x{:X}", fb.error_code);
        assert!(fb.velocity.abs() < 1e-2);
    }

    #[test]
    fn decode_rejects_wrong_dlc() {
        let frame = CanFrame { id: CanId(17), dlc: 4, data: [0; 8] };
        assert_eq!(decode_mit_feedback(&frame), Err(DecodeError::WrongDlc));
    }

    #[test]
    fn decode_rejects_can_id_below_master_offset() {
        let frame = CanFrame { id: CanId(15), dlc: 8, data: [0; 8] };
        assert_eq!(decode_mit_feedback(&frame), Err(DecodeError::WrongCanId));
    }

    #[test]
    fn decode_rejects_can_id_above_max_motor() {
        // motor_id + 16 must be ≤ 16 + 23 = 39 (23 motors on Atom01).
        // can_id=300 would (without guard) produce motor_id = (300-16) as u8 = 28 mod 256.
        let frame = CanFrame { id: CanId(300), dlc: 8, data: [0; 8] };
        assert_eq!(decode_mit_feedback(&frame), Err(DecodeError::WrongCanId));
    }

    #[test]
    fn decode_accepts_max_valid_can_id() {
        // motor_id 23 → can_id 39.
        let frame = CanFrame { id: CanId(39), dlc: 8, data: [0; 8] };
        assert_eq!(decode_mit_feedback(&frame).unwrap().motor_id, 23);
    }

    #[test]
    fn motor_model_ranges() {
        assert_eq!(MotorModel::Dm4340P.spd_max(), 20.0);
        assert_eq!(MotorModel::Dm10010L.spd_max(), 25.0);
        assert_eq!(MotorModel::Dm4340P.tau_max(), 28.0);
        assert_eq!(MotorModel::Dm10010L.tau_max(), 200.0);
    }
}
