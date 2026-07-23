//! Configuration constants sourced from upstream `atom01_deploy` project.
//!
//! All values here MUST be kept in sync with:
//! - `modules/atom01_deploy/src/inference/config/inference.yaml`
//! - `modules/atom01_deploy/src/inference/config/robot.yaml`
//!
//! Any change to those YAMLs requires regenerating the constants here.

// Action post-processing constants (inference.yaml:19-21)
// Source: atom01_deploy/src/inference/config/inference.yaml
pub const ACTION_SCALE: f32 = 0.25;
pub const CLIP_ACTIONS: f32 = 100.0;


/// USD→URDF index permutation (inference.yaml:21)
/// Maps policy output index `i` to URDF joint index `usd2urdf[i]`.
pub const USD2URDF: [usize; 23] = [
    0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22,
];


/// Default joint angles for home pose (inference.yaml:26-30)
pub const DEFAULT_JOINT_ANGLES: [f32; 23] = [
    // Left leg
    0.0, 0.0, -0.1, 0.3, -0.2, 0.0,
    // Right leg
    0.0, 0.0, -0.1, 0.3, -0.2, 0.0,
    // Torso
    0.0,
    // Left arm
    0.18, 0.06, 0.0, 0.78, 0.0,
    // Right arm
    0.18, -0.06, 0.0, 0.78, 0.0,
];



/// PD gains per motor (robot.yaml:32-35)
pub const KP: [f32; 23] = [
    100.0, 100.0, 100.0, 150.0, 40.0, 40.0, 100.0, 100.0, 100.0, 150.0, 40.0, 40.0, 150.0,
    40.0, 40.0, 40.0, 30.0, 20.0, 40.0, 40.0, 40.0, 30.0, 20.0,
];

pub const KD: [f32; 23] = [
    3.3, 3.3, 3.3, 5.0, 2.0, 2.0, 3.3, 3.3, 3.3, 5.0, 2.0, 2.0, 5.0, 2.0, 2.0, 2.0, 1.5, 1.0,
    2.0, 2.0, 2.0, 1.5, 1.0,
];

/// Motor direction sign correction (robot.yaml:42-46)
/// -1 reverses the sign before sending to the motor driver.
pub const MOTOR_SIGN: [i8; 23] = [
    1, 1, 1, 1, 1, 1,
    1, 1, -1, -1, -1, -1, 1,
    1, 1, 1, 1, 1,
    -1, 1, 1, -1, 1,
];





/// Observation vector layout (inference.yaml:5)
/// Layout: ang_vel(3) + gravity_b(3) + cmd_vel(3) + dof_pos(23) + dof_vel(23) + last_action(23) = 78
pub const OBS_DIM: usize = 78;

/// Frame stack history (inference.yaml:6)
pub const FRAME_STACK: usize = 10;

/// Control loop periods (milliseconds)
pub const INFERENCE_PERIOD_MS: u32 = 20; // 50 Hz
pub const CONTROL_PERIOD_MS: u32 = 4; // 250 Hz


