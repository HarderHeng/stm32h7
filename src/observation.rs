//! Observation builder with 10-frame history ring buffer.
//!
//! 78-dim observation layout (matches inference.yaml:5):
//!   ang_vel:3 + gravity_b:3 + cmd_vel:3 + dof_pos:23 + dof_vel:23 + last_action:23 = 78
//!
//! Frame stack: 10 frames × 78 floats = 780 floats input to MLP.
//!
//! No heap allocation; ring buffer is a fixed `[f32; FRAME_STACK * OBS_DIM]` stack array.

use crate::config::{FRAME_STACK, OBS_DIM};

pub const STACKED_DIM: usize = OBS_DIM * FRAME_STACK;

pub struct Observation {
    /// Ring buffer of past frames. Index 0 is the oldest, FRAME_STACK-1 is newest.
    buf: [f32; STACKED_DIM],
    write_idx: usize,
    frame_count: usize,
}

impl Observation {
    pub const fn new() -> Self {
        Self {
            buf: [0.0; STACKED_DIM],
            write_idx: 0,
            frame_count: 0,
        }
    }

    /// Push a new 78-float observation frame into the ring buffer.
    pub fn push(&mut self, obs: &[f32; OBS_DIM]) {
        let offset = self.write_idx * OBS_DIM;
        self.buf[offset..offset + OBS_DIM].copy_from_slice(obs);
        self.write_idx = (self.write_idx + 1) % FRAME_STACK;
        if self.frame_count < FRAME_STACK {
            self.frame_count += 1;
        }
    }

    /// Flatten the ring buffer into a chronological 780-float stack
    /// (oldest first, newest last). Used as MLP input.
    pub fn flatten_into(&self, dst: &mut [f32; STACKED_DIM]) {
        if self.frame_count < FRAME_STACK {
            let pad_floats = (FRAME_STACK - self.frame_count) * OBS_DIM;
            for v in &mut dst[..pad_floats] { *v = 0.0; }
            let live_floats = self.frame_count * OBS_DIM;
            dst[pad_floats..pad_floats + live_floats]
                .copy_from_slice(&self.buf[..live_floats]);
        } else {
            for i in 0..FRAME_STACK {
                let src_idx = ((self.write_idx + i) % FRAME_STACK) * OBS_DIM;
                let dst_idx = i * OBS_DIM;
                dst[dst_idx..dst_idx + OBS_DIM]
                    .copy_from_slice(&self.buf[src_idx..src_idx + OBS_DIM]);
            }
        }
    }

    pub fn frame_count(&self) -> usize {
        self.frame_count
    }
}

impl Default for Observation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_observation_has_zero_history() {
        let obs = Observation::new();
        assert_eq!(obs.frame_count(), 0);
        let mut dst = [0.0_f32; STACKED_DIM];
        obs.flatten_into(&mut dst);
        assert!(dst.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn push_increments_frame_count_until_full() {
        let mut obs = Observation::new();
        let frame = [1.0_f32; OBS_DIM];
        for i in 1..=FRAME_STACK {
            obs.push(&frame);
            assert_eq!(obs.frame_count(), i.min(FRAME_STACK));
        }
        assert_eq!(obs.frame_count(), FRAME_STACK);
    }

    #[test]
    fn flatten_chronological_order_after_fill() {
        let mut obs = Observation::new();
        for i in 0..FRAME_STACK {
            let mut frame = [0.0_f32; OBS_DIM];
            frame[0] = i as f32;
            obs.push(&frame);
        }
        let mut dst = [0.0_f32; STACKED_DIM];
        obs.flatten_into(&mut dst);
        for i in 0..FRAME_STACK {
            assert_eq!(dst[i * OBS_DIM], i as f32);
        }
    }

    #[test]
    fn flatten_pads_with_zeros_when_not_full() {
        let mut obs = Observation::new();
        let mut frame = [0.0_f32; OBS_DIM];
        frame[0] = 7.0;
        obs.push(&frame);
        let mut dst = [0.0_f32; STACKED_DIM];
        obs.flatten_into(&mut dst);
        // First 9 frames are padding (zeros), last frame is the 7.0
        for i in 0..FRAME_STACK - 1 {
            assert_eq!(dst[i * OBS_DIM], 0.0);
        }
        assert_eq!(dst[(FRAME_STACK - 1) * OBS_DIM], 7.0);
    }
}
