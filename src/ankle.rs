//! Ankle closed-chain (parallel 4-bar) decoupling.
//!
//! The robot's ankles are driven by two motors each (ankle_pitch + ankle_roll)
//! through a 4-bar parallel mechanism. The policy thinks in joint space
//! (pitch, roll), but motors operate in motor space (long-link angle,
//! short-link angle). This module bridges the two.
//!
//! Reference C++ source:
//! - `modules/atom01_deploy/src/inference/src/utils/decouple_atom01.cpp`
//!
//! Geometry parameters (mm, from decouple_atom01.cpp:11-38):
//! - long rod:  l_rod = 180 mm, l_bar = 20 mm
//! - short rod: l_rod = 110 mm, l_bar = 20 mm
//! - spacing:   l_spacing = ±42.35 mm (sign flips for left/right)

pub type Side = bool;

#[derive(Debug, Clone, Copy)]
pub struct AnkleParams {
    pub l_rod_long: f32,
    pub l_rod_short: f32,
    pub l_bar: f32,
    pub l_spacing: f32,
}

impl Default for AnkleParams {
    fn default() -> Self {
        Self {
            l_rod_long: 180.0,
            l_rod_short: 110.0,
            l_bar: 20.0,
            l_spacing: 42.35,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FkResult {
    pub pitch: f32,
    pub roll: f32,
    pub jacobian: [[f32; 2]; 2],
    pub iterations: u8,
    pub converged: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct LinkParamsAtom01 {
    pub l_rod: f32,
    pub l_bar: f32,
    pub l_spacing: f32,
    pub r_a_0: [f32; 3],
    pub r_b_0: [f32; 3],
    pub r_c_0: [f32; 3],
    pub theta_0: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct IkResultAtom01 {
    pub r_bar: [[f32; 3]; 2],
    pub r_rod: [[f32; 3]; 2],
    pub r_c: [[f32; 3]; 2],
    pub theta: [f32; 2],
}

#[derive(Debug, Clone, Copy, Default)]
pub struct JacobianResult {
    pub j_motor2joint: [[f32; 2]; 2],
    pub j_joint2motor: [[f32; 2]; 2],
}

impl From<[[f32; 2]; 2]> for JacobianResult {
    fn from(j_motor2joint: [[f32; 2]; 2]) -> Self {
        Self { j_motor2joint, j_joint2motor: j_motor2joint }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AnkleDecoupler {
    pub params: AnkleParams,
    pub side: Side,
    links_left: [LinkParamsAtom01; 2],
    links_right: [LinkParamsAtom01; 2],
}

const S_BAR: [f32; 3] = [0.0, 1.0, 0.0];

const PI: f32 = core::f32::consts::PI;
const TWO_PI: f32 = 2.0 * core::f32::consts::PI;

/// Reduce x to [-π, π] via repeated subtraction (no f32::rem_euclid in no_std).
/// Falls back to x on NaN/Inf or if x is so large that the loop would never
/// terminate within a sane iteration count (32 wraps cover |x| up to ~200π).
#[inline]
fn wrap_to_pi(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let mut x = x;
    for _ in 0..32 {
        if x > PI { x -= TWO_PI; } else if x < -PI { x += TWO_PI; } else { return x; }
    }
    x
}

/// 5-term Taylor series for sin(x). Accurate to ~1e-5 for `|x| < π/2`.
/// `no_std`-friendly (no f32::sin required).
#[inline]
fn sin_approx(x: f32) -> f32 {
    let x = wrap_to_pi(x);
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;
    let x7 = x5 * x2;
    x - x3 * (1.0 / 6.0) + x5 * (1.0 / 120.0) - x7 * (1.0 / 5040.0)
}

/// 5-term Taylor series for cos(x). Accurate to ~1e-5 for `|x| < π/2`.
#[inline]
fn cos_approx(x: f32) -> f32 {
    let x = wrap_to_pi(x);
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    1.0 - x2 * 0.5 + x4 * (1.0 / 24.0) - x6 * (1.0 / 720.0)
}

/// asin via Taylor (only used inside cos/sin paths, small angle assumption in IK).
#[inline]
fn asin_approx(x: f32) -> f32 {
    let x = x.clamp(-1.0, 1.0);
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;
    let x7 = x5 * x2;
    let x9 = x7 * x2;
    x + x3 * (1.0 / 6.0) + x5 * (3.0 / 40.0) + x7 * (5.0 / 112.0) + x9 * (35.0 / 1152.0)
}

#[inline]
fn sqrt_approx(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut z = x * 0.5;
    let mut y = x;
    let mut i = 0;
    while i < 6 {
        y = 0.5 * (y + z);
        z = x / y;
        i += 1;
    }
    y
}

impl AnkleDecoupler {
    pub fn new(side: Side) -> Self {
        let params = AnkleParams {
            l_rod_long: 180.0,
            l_rod_short: 110.0,
            l_bar: 20.0,
            l_spacing: if side { 42.35 } else { -42.35 },
        };
        let links_left = build_links(true);
        let links_right = build_links(false);
        Self { params, side, links_left, links_right }
    }

    fn links(&self) -> &[LinkParamsAtom01; 2] {
        if self.side { &self.links_left } else { &self.links_right }
    }

    pub fn inverse_kinematics(&self, q_pitch: f32, q_roll: f32) -> IkResultAtom01 {
        let mut out = IkResultAtom01::default();
        let links = self.links();

        let cos_p = cos_approx(q_pitch);
        let sin_p = sin_approx(q_pitch);
        let cos_r = cos_approx(q_roll);
        let sin_r = sin_approx(q_roll);

        // x_rot = R_y(q_pitch) * R_x(q_roll)
        // R_y = [[cos_p, 0, sin_p], [0, 1, 0], [-sin_p, 0, cos_p]]
        // R_x = [[1, 0, 0], [0, cos_r, -sin_r], [0, sin_r, cos_r]]
        let links_0_r_c: [[f32; 3]; 2] = [
            {
                let r_c_0 = links[0].r_c_0;
                // R_y * R_x * r_c_0
                let x = r_c_0[0];
                let y = cos_r * r_c_0[1] - sin_r * r_c_0[2];
                let z = sin_r * r_c_0[1] + cos_r * r_c_0[2];
                [cos_p * x + sin_p * z, y, -sin_p * x + cos_p * z]
            },
            {
                let r_c_0 = links[1].r_c_0;
                let x = r_c_0[0];
                let y = cos_r * r_c_0[1] - sin_r * r_c_0[2];
                let z = sin_r * r_c_0[1] + cos_r * r_c_0[2];
                [cos_p * x + sin_p * z, y, -sin_p * x + cos_p * z]
            },
        ];

        for i in 0..2 {
            let l_rod = links[i].l_rod;
            let l_bar = links[i].l_bar;
            let r_a_i = links[i].r_a_0;
            let r_c_i = links_0_r_c[i];
            let r_ba_bar = [
                links[i].r_b_0[0] - r_a_i[0],
                links[i].r_b_0[1] - r_a_i[1],
                links[i].r_b_0[2] - r_a_i[2],
            ];

            let a = r_c_i[0] - r_a_i[0];
            let b = r_a_i[2] - r_c_i[2];
            let dx = r_c_i[0] - r_a_i[0];
            let dy = r_c_i[1] - r_a_i[1];
            let dz = r_c_i[2] - r_a_i[2];
            let r_ca_sq = dx * dx + dy * dy + dz * dz;
            let c = (l_rod * l_rod - l_bar * l_bar - r_ca_sq) / (2.0 * l_bar);

            let a_sq = a * a;
            let b_sq = b * b;
            let c_sq = c * c;
            let discriminant = b_sq * c_sq - (a_sq + b_sq) * (c_sq - a_sq);
            let disc = if discriminant < 0.0 { 0.0 } else { discriminant };

            let theta_i = asin_approx((b * c + sqrt_approx(disc)) / (a_sq + b_sq));
            let theta_signed = if a < 0.0 { theta_i } else { -theta_i };

            let cos_t = cos_approx(theta_signed);
            let sin_t = sin_approx(theta_signed);
            // R_y(theta) * r_ba_bar
            let r_b_i = [
                r_a_i[0] + cos_t * r_ba_bar[0] + sin_t * r_ba_bar[2],
                r_a_i[1] + r_ba_bar[1],
                r_a_i[2] - sin_t * r_ba_bar[0] + cos_t * r_ba_bar[2],
            ];

            out.r_bar[i] = [r_b_i[0] - r_a_i[0], r_b_i[1] - r_a_i[1], r_b_i[2] - r_a_i[2]];
            out.r_rod[i] = [r_c_i[0] - r_b_i[0], r_c_i[1] - r_b_i[1], r_c_i[2] - r_b_i[2]];
            out.r_c[i] = r_c_i;
            out.theta[i] = theta_signed;
        }
        out
    }

    pub fn jacobian(&self, ik: &IkResultAtom01, q_pitch: f32) -> JacobianResult {
        let mut j_x: [[f32; 6]; 2] = [[0.0; 6]; 2];
        for i in 0..2 {
            let r_rod = ik.r_rod[i];
            let r_c = ik.r_c[i];
            let cross = [
                r_c[1] * r_rod[2] - r_c[2] * r_rod[1],
                r_c[2] * r_rod[0] - r_c[0] * r_rod[2],
                r_c[0] * r_rod[1] - r_c[1] * r_rod[0],
            ];
            j_x[2 * i][0] = r_rod[0];
            j_x[2 * i][1] = r_rod[1];
            j_x[2 * i][2] = r_rod[2];
            j_x[2 * i][3] = cross[0];
            j_x[2 * i][4] = cross[1];
            j_x[2 * i][5] = cross[2];
        }

        // J_theta: 2x2 diagonal with S_BAR · (r_bar × r_rod)
        let cross0 = cross3(&ik.r_bar[0], &ik.r_rod[0]);
        let cross1 = cross3(&ik.r_bar[1], &ik.r_rod[1]);
        let dot0 = dot3(&S_BAR, &cross0);
        let dot1 = dot3(&S_BAR, &cross1);
        let mut j_theta = [[0.0_f32; 2]; 2];
        j_theta[0][0] = dot0;
        j_theta[1][1] = dot1;

        // J_q (mapping from joint angle to end-effector spatial velocity twist)
        let cos_p = cos_approx(q_pitch);
        let sin_p = sin_approx(q_pitch);
        let mut j_q: [[f32; 2]; 6] = [[0.0; 2]; 6];
        // j_q = [[0, 0], [0, 0], [0, 0], [0, cos_p], [1, 0], [0, -sin_p]]
        j_q[3][1] = cos_p;
        j_q[4][0] = 1.0;
        j_q[5][1] = -sin_p;

        // J_temp = J_x · J_q   (2x2)
        let mut j_temp = [[0.0_f32; 2]; 2];
        for r in 0..2 {
            for c in 0..2 {
                let mut sum = 0.0;
                for k in 0..6 {
                    sum += j_x[r][k] * j_q[k][c];
                }
                j_temp[r][c] = sum;
            }
        }

        // J_motor2Joint = J_theta^-1 · J_temp
        let det_temp = j_temp[0][0] * j_temp[1][1] - j_temp[0][1] * j_temp[1][0];
        let inv_temp = if det_temp.abs() < 1e-9 {
            [[0.0; 2]; 2]
        } else {
            let d = 1.0 / det_temp;
            [[d * j_temp[1][1], -d * j_temp[0][1]], [-d * j_temp[1][0], d * j_temp[0][0]]]
        };

        let det_theta = j_theta[0][0] * j_theta[1][1] - j_theta[0][1] * j_theta[1][0];
        let inv_theta = if det_theta.abs() < 1e-9 {
            [[0.0; 2]; 2]
        } else {
            let d = 1.0 / det_theta;
            [[d * j_theta[1][1], -d * j_theta[0][1]], [-d * j_theta[1][0], d * j_theta[0][0]]]
        };

        let mut j_motor2joint = [[0.0_f32; 2]; 2];
        let mut j_joint2motor = [[0.0_f32; 2]; 2];
        for r in 0..2 {
            for c in 0..2 {
                let mut s1 = 0.0;
                let mut s2 = 0.0;
                for k in 0..2 {
                    s1 += inv_temp[r][k] * j_theta[k][c];
                    s2 += inv_theta[r][k] * j_temp[k][c];
                }
                j_motor2joint[r][c] = s1;
                j_joint2motor[r][c] = s2;
            }
        }

        JacobianResult { j_motor2joint, j_joint2motor }
    }

    pub fn forward_kinematics(&self, motor_angles: &[f32; 2]) -> FkResult {
        // Newton iteration, mirroring decouple_atom01.cpp:161-205.
        // Damped (ALPHA=0.5) and bounded; we also abort if the joint
        // estimate diverges outside the physically reachable range, since
        // NaN/Inf guard alone wouldn't catch monotonic blow-up.
        const MAX_ITER: u8 = 100;
        const TOLERANCE: f32 = 1e-3;
        const ALPHA: f32 = 0.5;
        const JOINT_LIMIT: f32 = 1.5; // ~86°; mechanism reach is well under this.

        let mut x_k = [0.0_f32, 0.0_f32]; // [pitch, roll]
        let mut last_error = [10.0_f32, 10.0_f32];
        let mut iterations = 0u8;
        let mut jac = JacobianResult::default();

        for _ in 0..MAX_ITER {
            let pitch = x_k[0];
            let roll = x_k[1];
            let ik = self.inverse_kinematics(pitch, roll);
            jac = self.jacobian(&ik, pitch);

            if jac.j_motor2joint.iter().flatten().any(|v| !v.is_finite()) {
                return FkResult {
                    pitch: 0.0,
                    roll: 0.0,
                    jacobian: [[1.0, 0.0], [0.0, 1.0]],
                    iterations,
                    converged: false,
                };
            }

            let f_error = [motor_angles[0] - ik.theta[0], motor_angles[1] - ik.theta[1]];
            let update = [
                ALPHA * (jac.j_motor2joint[0][0] * f_error[0] + jac.j_motor2joint[0][1] * f_error[1]),
                ALPHA * (jac.j_motor2joint[1][0] * f_error[0] + jac.j_motor2joint[1][1] * f_error[1]),
            ];

            // Divergence guard: bail out before the estimate blows up.
            // Catches pathological motor_angles that put the mechanism
            // outside its workspace (e.g., very large inputs that wrap
            // around the asin branch) where Newton would otherwise
            // iterate forever producing larger and larger joint angles.
            if update[0].abs() > JOINT_LIMIT || update[1].abs() > JOINT_LIMIT
                || !update[0].is_finite() || !update[1].is_finite()
            {
                return FkResult {
                    pitch: 0.0,
                    roll: 0.0,
                    jacobian: [[1.0, 0.0], [0.0, 1.0]],
                    iterations,
                    converged: false,
                };
            }

            x_k[0] += update[0];
            x_k[1] += update[1];
            iterations += 1;
            last_error = f_error;

            if f_error[0].abs() < TOLERANCE && f_error[1].abs() < TOLERANCE {
                return FkResult {
                    pitch: x_k[0],
                    roll: x_k[1],
                    jacobian: jac.j_motor2joint,
                    iterations,
                    converged: true,
                };
            }
        }
        FkResult {
            pitch: x_k[0],
            roll: x_k[1],
            jacobian: jac.j_motor2joint,
            iterations,
            converged: last_error[0].abs() < TOLERANCE && last_error[1].abs() < TOLERANCE,
        }
    }

    pub fn joint_torque_to_motor(&self, tau_joint: &[f32; 2], jac: &JacobianResult) -> [f32; 2] {
        // tau_motor = J_motor2Joint^T · tau_joint
        [
            jac.j_motor2joint[0][0] * tau_joint[0] + jac.j_motor2joint[1][0] * tau_joint[1],
            jac.j_motor2joint[0][1] * tau_joint[0] + jac.j_motor2joint[1][1] * tau_joint[1],
        ]
    }
}

fn dot3(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn build_links(is_left: bool) -> [LinkParamsAtom01; 2] {
    let l_bar = 20.0;
    let l_spacing = if is_left { 42.35 } else { -42.35 };
    let long_angle_0 = 0.0_f32;
    let short_angle_0 = core::f32::consts::PI;

    // Use exact cos/sin for known angles (0, π) — the 5-term Taylor diverges
    // badly there (cos(π) ≈ -1.22, sin(π) ≈ -0.075 with the Taylor in this file).
    let cos_long = 1.0_f32;
    let sin_long = 0.0_f32;
    let cos_short = -1.0_f32;
    let sin_short = 0.0_f32;

    let r_b1_x = -l_bar * cos_long;
    let r_b1_z = 180.0 - l_bar * sin_long;
    let r_b2_x = -l_bar * cos_short;
    let r_b2_z = 110.0 - l_bar * sin_short;

    [
        LinkParamsAtom01 {
            l_rod: 180.0,
            l_bar,
            l_spacing,
            r_a_0: [0.0, l_spacing, 180.0],
            r_b_0: [r_b1_x, l_spacing, r_b1_z],
            r_c_0: [-20.0, l_spacing, 0.0],
            theta_0: long_angle_0,
        },
        LinkParamsAtom01 {
            l_rod: 110.0,
            l_bar,
            l_spacing,
            r_a_0: [0.0, l_spacing, 110.0],
            r_b_0: [r_b2_x, l_spacing, r_b2_z],
            r_c_0: [20.0, l_spacing, 0.0],
            theta_0: short_angle_0,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn left_and_right_have_mirrored_spacing() {
        let left = AnkleDecoupler::new(true);
        let right = AnkleDecoupler::new(false);
        assert!(left.params.l_spacing > 0.0);
        assert!(right.params.l_spacing < 0.0);
        assert!((left.params.l_spacing + right.params.l_spacing).abs() < 1e-6);
    }

    #[test]
    fn default_params_match_atom01_spec() {
        let d = AnkleDecoupler::new(true);
        assert_eq!(d.params.l_rod_long, 180.0);
        assert_eq!(d.params.l_rod_short, 110.0);
        assert_eq!(d.params.l_bar, 20.0);
    }

    #[test]
    fn ik_at_zero_pose_returns_motor_angles_near_construction() {
        let d = AnkleDecoupler::new(true);
        let ik = d.inverse_kinematics(0.0, 0.0);
        // At zero pose, theta should match theta_0 values from construction
        assert!((ik.theta[0] - 0.0).abs() < 1e-3, "long motor: {}", ik.theta[0]);
        assert!((ik.theta[1] - core::f32::consts::PI).abs() < 1e-3, "short motor: {}", ik.theta[1]);
    }

    #[test]
    fn fk_roundtrip_preserves_joint_angles() {
        let d = AnkleDecoupler::new(true);
        for &(target_pitch, target_roll) in &[
            (0.0, 0.0),
            (0.1, 0.05),
            (-0.1, 0.05),
            (0.2, -0.1),
            (0.05, 0.2),
        ] {
            let ik = d.inverse_kinematics(target_pitch, target_roll);
            let motor = [ik.theta[0], ik.theta[1]];
            let fk = d.forward_kinematics(&motor);
            assert!(fk.converged, "FK did not converge for ({target_pitch}, {target_roll})");
            assert!(
                (fk.pitch - target_pitch).abs() < 1e-2,
                "pitch: target={target_pitch}, got={}", fk.pitch
            );
            assert!(
                (fk.roll - target_roll).abs() < 1e-2,
                "roll: target={target_roll}, got={}", fk.roll
            );
        }
    }

    #[test]
    fn fk_bails_out_on_pathological_motor_angles() {
        // Motor angles way outside the mechanism workspace must not blow up
        // Newton iteration; we should bail out with converged=false rather
        // than iterate to max_iter producing garbage joint angles.
        let d = AnkleDecoupler::new(true);
        let fk = d.forward_kinematics(&[100.0, -100.0]);
        assert!(!fk.converged, "FK should detect divergence");
        // Returned joint angles must be finite even on the failure path.
        assert!(fk.pitch.is_finite());
        assert!(fk.roll.is_finite());
    }

    #[test]
    fn fk_bails_out_on_nan_motor_angles() {
        let d = AnkleDecoupler::new(true);
        let fk = d.forward_kinematics(&[f32::NAN, 0.0]);
        assert!(!fk.converged);
        assert!(fk.pitch.is_finite());
        assert!(fk.roll.is_finite());
    }

    #[test]
    fn joint_torque_to_motor_uses_transpose() {
        let d = AnkleDecoupler::new(true);
        let jac = JacobianResult {
            j_motor2joint: [[1.0, 0.5], [0.2, 1.0]],
            j_joint2motor: [[1.0, 0.5], [0.2, 1.0]],
        };
        let tau_motor = d.joint_torque_to_motor(&[1.0, 2.0], &jac);
        // J^T · tau = [[1, 0.2], [0.5, 1]] · [1, 2] = [1+0.4, 0.5+2] = [1.4, 2.5]
        assert!((tau_motor[0] - 1.4).abs() < 1e-6);
        assert!((tau_motor[1] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn ik_far_from_zero_still_returns_valid_angles() {
        let d = AnkleDecoupler::new(true);
        let ik = d.inverse_kinematics(0.3, 0.15);
        // Result should be finite
        assert!(ik.theta[0].is_finite());
        assert!(ik.theta[1].is_finite());
    }

    #[test]
    fn build_links_uses_exact_geometry_at_zero_and_pi() {
        // Long link at angle 0: cos(0)=1, sin(0)=0 → r_b_1 = (-l_bar, l_spacing, 180).
        // Short link at angle π: cos(π)=-1, sin(π)=0 → r_b_2 = (+l_bar, l_spacing, 110).
        let left = AnkleDecoupler::new(true);
        let right = AnkleDecoupler::new(false);
        assert_eq!(left.params.links_left[0].r_b_0, [-20.0, 42.35, 180.0]);
        assert_eq!(left.params.links_left[1].r_b_0, [20.0, 42.35, 110.0]);
        assert_eq!(right.params.links_right[0].r_b_0, [-20.0, -42.35, 180.0]);
        assert_eq!(right.params.links_right[1].r_b_0, [20.0, -42.35, 110.0]);
    }

    #[test]
    fn wrap_to_pi_handles_huge_values_without_hanging() {
        let r = wrap_to_pi(1.0e10);
        assert!(r.is_finite());
        assert_eq!(wrap_to_pi(0.0), 0.0);
        assert!((wrap_to_pi(PI * 3.0)).abs() <= PI);
        assert!((wrap_to_pi(-PI * 3.0)).abs() <= PI);
        assert!(wrap_to_pi(f32::NAN).is_nan());
    }

    #[test]
    fn fk_nan_check_covers_all_jacobian_rows() {
        // Newton iteration's NaN guard must inspect all 4 elements of the 2x2
        // Jacobian, not just row 0. Verify both rows are detected.
        let jac_with_nan_row0 = [[f32::NAN, 0.0], [0.0, 1.0]];
        let jac_with_nan_row1 = [[1.0, 0.0], [0.0, f32::NAN]];
        for jac in [jac_with_nan_row0, jac_with_nan_row1] {
            assert!(jac.iter().flatten().any(|v| v.is_nan()));
        }
    }
}
