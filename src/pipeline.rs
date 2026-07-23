//! Embassy task pipeline that ties all algorithm modules together.
//!
//! Defines two async tasks + shared state:
//!   - `inference_task` @ 50 Hz: build obs (78) → frame-stack (780) → MLP forward → post-process
//!   - `control_task`   @ 4 ms: read motor feedback → decouple ankles → 23× MIT frames → CAN TX
//!
//! All hardware I/O is routed through the [`crate::drivers`] module's stubs,
//! which must be filled in with real embassy-stm32 peripherals (Phase 6/7).
//!
//! Reference:
//!   - `modules/atom01_deploy/src/inference/src/inference_node.cpp` (task setup)
//!   - `modules/atom01_deploy/src/inference/src/robot_interface.cpp` (control loop)

use crate::config::{CONTROL_PERIOD_MS, DEFAULT_JOINT_ANGLES, INFERENCE_PERIOD_MS, KP, KD, MOTOR_SIGN, OBS_DIM};
use crate::ankle::{AnkleDecoupler, Side};
use crate::canproto::{CanFrame, MitCommand, MotorModel};
use crate::hal::CanBus;
use crate::mlp::{Mlp, OUTPUT_DIM, INPUT_DIM};
use crate::observation::Observation;
use crate::drivers::ImuSerial;

#[derive(Default)]
pub struct ImuState {
    pub quat: [f32; 4],     // (w, x, y, z)
    pub ang_vel: [f32; 3], // body frame
    pub updated: bool,
}

#[derive(Default)]
pub struct JointState {
    pub position: [f32; 23],
    pub velocity: [f32; 23],
    pub torque: [f32; 23],
    pub feedback_stale: bool,
}

pub struct Pipeline {
    pub obs: Observation,
    pub imu: ImuState,
    pub joint: JointState,
    pub action: [f32; OUTPUT_DIM], // post-processed, ready for motors
    pub raw_action: [f32; OUTPUT_DIM],
    pub stacked: [f32; INPUT_DIM],
    pub ankle_left: AnkleDecoupler,
    pub ankle_right: AnkleDecoupler,
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl Pipeline {
    pub fn new() -> Self {
        Self {
            obs: Observation::new(),
            imu: ImuState::default(),
            joint: JointState::default(),
            action: [0.0; OUTPUT_DIM],
            raw_action: [0.0; OUTPUT_DIM],
            stacked: [0.0; INPUT_DIM],
            ankle_left: AnkleDecoupler::new(true),
            ankle_right: AnkleDecoupler::new(false),
        }
    }

    /// Build a 78-dim observation vector from current IMU + joint state + cmd_vel.
    /// Output layout (matches inference.yaml:5):
    ///   ang_vel:3 + gravity_b:3 + cmd_vel:3 + dof_pos:23 + dof_vel:23 + last_action:23
    pub fn build_observation(&self, cmd_vel: &[f32; 3], out: &mut [f32; OBS_DIM]) {
        let mut i = 0;
        // ang_vel
        out[i..i + 3].copy_from_slice(&self.imu.ang_vel);
        i += 3;
        // gravity_b = q_body^-1 * [0, 0, -1]
        let g = compute_gravity_body(&self.imu.quat);
        out[i..i + 3].copy_from_slice(&g);
        i += 3;
        // cmd_vel
        out[i..i + 3].copy_from_slice(cmd_vel);
        i += 3;
        // dof_pos - default
        for j in 0..23 {
            out[i + j] = self.joint.position[j] - DEFAULT_JOINT_ANGLES[j];
        }
        i += 23;
        // dof_vel
        out[i..i + 23].copy_from_slice(&self.joint.velocity);
        i += 23;
        // last_action
        out[i..i + 23].copy_from_slice(&self.action);
    }

    /// One inference step: build obs → frame-stack → MLP forward → post-process.
/// Call at 50 Hz (every 20 ms).
    pub fn step_inference(&mut self, cmd_vel: &[f32; 3]) {
        let mut obs_frame = [0.0_f32; OBS_DIM];
        self.build_observation(cmd_vel, &mut obs_frame);
        self.obs.push(&obs_frame);
        self.obs.flatten_into(&mut self.stacked);
        Mlp::forward_int8(&self.stacked, &mut self.raw_action);

        if self.raw_action.iter().any(|v| !v.is_finite()) {
            for v in self.raw_action.iter_mut() { *v = 0.0; }
        }

        Mlp::post_process(&self.raw_action, &mut self.action);
    }

    /// One control step: build 23 MIT frames from current action + joint state.
    /// Returns the frames ready to be transmitted on the correct CAN bus.
    ///
    /// 4 ankle motors (indices 4,5,10,11) receive torque commands from FK→PD→J^T;
    /// the other 19 motors receive MIT position commands with kp/kd gains.
    ///
    /// Call at 250 Hz (every 4 ms).
    pub fn step_control(&mut self) -> [CanFrame; 23] {
        let mut frames = [empty_frame(); 23];
        const ANKLE_INDICES: [usize; 4] = [4, 5, 10, 11];

        for (side, urdf_base) in [(true, 4usize), (false, 10usize)] {
            let tau = self.compute_ankle_torques(side);
            for (offset, &t) in tau.iter().enumerate() {
                let motor_idx = urdf_base + offset;
                frames[motor_idx] = crate::canproto::encode_mit(
                    Self::motor_id_for(motor_idx),
                    motor_model_for(motor_idx),
                    &MitCommand { pos: 0.0, vel: 0.0, kp: 0.0, kd: 0.0, tau: t },
                );
            }
        }

        for motor_idx in 0..23 {
            if ANKLE_INDICES.contains(&motor_idx) {
                continue;
            }
            let signed_target = self.action[motor_idx] * MOTOR_SIGN[motor_idx] as f32;
            frames[motor_idx] = crate::canproto::encode_mit(
                Self::motor_id_for(motor_idx),
                motor_model_for(motor_idx),
                &MitCommand {
                    pos: signed_target,
                    vel: 0.0,
                    kp: KP[motor_idx],
                    kd: KD[motor_idx],
                    tau: 0.0,
                },
            );
        }
        frames
    }

    /// Convert URDF/motor index (0..=22) to the DAMIAO CAN motor_id (1..=23).
    /// Panics in debug if out of range; the cast is intentional at the call
    /// sites so a mis-indexed loop is caught immediately.
    fn motor_id_for(idx: usize) -> u8 {
        debug_assert!(idx < 23, "motor index {idx} out of range (max 22)");
        (idx + 1) as u8
    }

    /// Compute ankle motor torques from joint-space PD law.
    /// Called by `step_control` for the 4 ankle motors (indices 4,5,10,11).
    /// Returns `[tau_pitch_motor, tau_roll_motor]` in motor space.
    pub fn compute_ankle_torques(&self, side: Side) -> [f32; 2] {
        let ankle = if side { &self.ankle_left } else { &self.ankle_right };
        let urdf_base = if side { 4 } else { 10 };

        let motor_angles = [
            self.joint.position[urdf_base] * MOTOR_SIGN[urdf_base] as f32,
            self.joint.position[urdf_base + 1] * MOTOR_SIGN[urdf_base + 1] as f32,
        ];
        let motor_vels = [
            self.joint.velocity[urdf_base] * MOTOR_SIGN[urdf_base] as f32,
            self.joint.velocity[urdf_base + 1] * MOTOR_SIGN[urdf_base + 1] as f32,
        ];
        let fk = ankle.forward_kinematics(&motor_angles);
        let tau_joint = [
            KP[urdf_base] * (self.action[urdf_base] - fk.pitch) + KD[urdf_base] * (0.0 - motor_vels[0]),
            KP[urdf_base + 1] * (self.action[urdf_base + 1] - fk.roll) + KD[urdf_base + 1] * (0.0 - motor_vels[1]),
        ];
        // fk.jacobian is J_motor2joint. tau_motor = J^T · tau_joint.
        ankle.joint_torque_to_motor(&tau_joint, &fk.jacobian.into())
    }
}

fn motor_model_for(motor_idx: usize) -> MotorModel {
    // robot.yaml:18-22 maps motor index → model
    // DM4340P (index 1): 14 hip/knee/waist motors → indices 0,1,2,3,6,7,8,9,12
    // DM10010L (index 0): 9 ankle/shoulder/elbow motors → indices 4,5,10,11,13,14,15,16,17,18,19,20,21,22
    match motor_idx {
        0 | 1 | 2 | 3 | 6 | 7 | 8 | 9 | 12 => MotorModel::Dm4340P,
        _ => MotorModel::Dm10010L,
    }
}

fn compute_gravity_body(quat: &[f32; 4]) -> [f32; 3] {
    // Quaternion convention (w, x, y, z) — match upstream atom01_deploy.
    // For unit q, q_body_inv = (w, -x, -y, -z). Rotates world gravity [0, 0, -1]
    // into body frame. The sign of gz depends on the IMU frame convention
    // (NED → +z is down, ENU → +z is up); verify against upstream before
    // flashing. For NED this returns (0, 0, 1) at identity quaternion,
    // meaning gravity points along body +z — correct for an upright NED robot.
    let (w, x, y, z) = (quat[0], -quat[1], -quat[2], -quat[3]);
    [
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    ]
}

fn empty_frame() -> CanFrame {
    CanFrame { id: crate::canproto::CanId(0), dlc: 0, data: [0; 8] }
}

/// Inference task — call from Embassy executor at 50 Hz.
/// Placeholder; replace with `embassy_executor::task` attribute in main.rs.
pub async fn inference_task(pipeline: &mut Pipeline, cmd_vel: &[f32; 3]) {
    pipeline.step_inference(cmd_vel);
}

/// Control task — call from Embassy executor at 250 Hz.
/// Reads motor feedback, dispatches MIT frames to CAN buses.
pub async fn control_task(pipeline: &mut Pipeline, _imu: &mut ImuSerial, _can_buses: &mut [&mut dyn CanBus; 4]) {
    let frames = pipeline.step_control();
    for frame in frames.iter() {
        // Bus dispatch (Phase 7): choose bus by motor ID
        // For now, this is a stub — real wiring needs embassy_stm32::can::Can
        let _ = frame;
    }
}

pub const INFERENCE_PERIOD_US: u32 = INFERENCE_PERIOD_MS * 1000;
pub const CONTROL_PERIOD_US: u32 = CONTROL_PERIOD_MS * 1000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_creation_is_clean() {
        let p = Pipeline::new();
        assert_eq!(p.action, [0.0; OUTPUT_DIM]);
        assert_eq!(p.obs.frame_count(), 0);
    }

    #[test]
    fn step_inference_produces_post_processed_action() {
        let mut p = Pipeline::new();
        let cmd = [0.0_f32; 3];
        p.step_inference(&cmd);
        // With zero input and weights, raw_action should be near zero
        // → post-processed action ≈ DEFAULT_JOINT_ANGLES
        for i in 0..OUTPUT_DIM {
            let diff = (p.action[i] - DEFAULT_JOINT_ANGLES[i]).abs();
            assert!(diff < 1.0, "joint {}: action {} vs default {}",
                    i, p.action[i], DEFAULT_JOINT_ANGLES[i]);
        }
    }

    #[test]
    fn step_control_emits_23_mit_frames() {
        let mut p = Pipeline::new();
        let frames = p.step_control();
        assert_eq!(frames.len(), 23);
        for (i, frame) in frames.iter().enumerate() {
            assert_eq!(frame.dlc, 8, "frame {} has wrong dlc", i);
            assert!(frame.id.0 >= 1 && frame.id.0 <= 23, "frame {} has bad id {}", i, frame.id.0);
        }
    }

    #[test]
    fn ankle_motors_get_torque_only_commands() {
        let mut p = Pipeline::new();
        let frames = p.step_control();
        for &ankle_idx in &[4usize, 5, 10, 11] {
            let f = frames[ankle_idx];
            // pos = 0 → data[0..1] = 0x7FFF (midpoint of ±12.5 range)
            let p_raw = u16::from_be_bytes([f.data[0], f.data[1]]);
            assert!((p_raw as i32 - 0x7FFF).abs() <= 1,
                "ankle {} pos midpoint: got 0x{:04X}", ankle_idx, p_raw);
            // kp = 0 → byte[3] high nibble (top 4 bits of 12-bit kp) = 0,
            //         byte[4] (low 8 bits of kp) = 0
            let kp_high = (f.data[3] & 0x0F) as u16;
            let kp_low = f.data[4] as u16;
            assert_eq!((kp_high << 8) | kp_low, 0, "ankle {} kp should be 0", ankle_idx);
        }
    }

    #[test]
    fn ankle_torques_reflect_decoupling_error() {
        let mut p = Pipeline::new();
        // Default ankle_pitch_l target is -0.2 (DEFAULT_JOINT_ANGLES[4]).
        // Pretend the joint drifted to +0.3, so PD error = -0.5.
        p.joint.position[4] = 0.3;
        p.joint.position[5] = 0.1;

        let tau_left = p.compute_ankle_torques(true);
        assert!(tau_left[0].is_finite());
        assert!(tau_left[1].is_finite());
        // Positive KP means tau has the same sign as (action - joint).
        // action_pitch = -0.2, fk.pitch ≈ 0.3 → error negative → tau negative.
        assert!(tau_left[0] < 0.0, "expected negative tau_pitch, got {}", tau_left[0]);
    }

    #[test]
    fn step_control_uses_compute_ankle_torques_path() {
        // Regression: step_control must no longer pass joint angle as torque
        // (the old code used `tau: signed_target` with wrong units).
        let mut p = Pipeline::new();
        p.joint.position[4] = 0.5;
        let frames = p.step_control();
        // Read tau_raw from byte[6..8] of ankle motor 4's frame.
        let f = &frames[4];
        let tau_raw = (((f.data[6] & 0x0F) as u16) << 8) | (f.data[7] as u16);
        // Old buggy code would saturate tau_raw (~0xFFF). New code produces
        // a finite PD output proportional to the joint error.
        assert!(tau_raw < 0x0FFF, "ankle tau_raw should not saturate, got 0x{:X}", tau_raw);
    }

    #[test]
    fn non_ankle_motors_get_position_commands() {
        let mut p = Pipeline::new();
        let frames = p.step_control();
        // Pick a non-ankle motor (e.g., motor 0) and verify kp is non-zero
        let f = frames[0];
        // kp high nibble in data[3] should be non-zero (since KP[0] = 100)
        let kp_high = f.data[3] & 0x0F;
        assert!(kp_high > 0, "non-ankle motor should have non-zero kp");
    }

    #[test]
    fn motor_model_assignment_matches_robot_yaml() {
        // Indices 0,1,2,3 (left leg) should be DM4340P
        for &i in &[0usize, 1, 2, 3] {
            assert!(matches!(motor_model_for(i), MotorModel::Dm4340P),
                    "motor {} should be DM4340P", i);
        }
        // Index 4 (left ankle pitch) should be DM10010L
        assert!(matches!(motor_model_for(4), MotorModel::Dm10010L));
        // Index 12 (waist) should be DM4340P
        assert!(matches!(motor_model_for(12), MotorModel::Dm4340P));
    }
}
