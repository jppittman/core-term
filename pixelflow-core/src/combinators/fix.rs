//! # Fix Combinator
//!
//! Iterate until convergence (fixed-point iteration).

use crate::backend::{BatchArithmetic, SimdBatch};
use crate::{Field, Manifold};

/// Iterate until convergence.
///
/// Evaluates `step` repeatedly (using W as state) until `done` returns true.
/// Uses per-lane retirement for efficiency.
#[derive(Clone, Copy, Debug)]
pub struct Fix<Seed, Step, Done> {
    /// Initial state value.
    pub seed: Seed,
    /// Step function: computes next state from current (via W).
    pub step: Step,
    /// Termination condition: returns true when lane has converged.
    pub done: Done,
}

impl<Seed: Manifold, Step: Manifold, Done: Manifold> Manifold for Fix<Seed, Step, Done> {
    #[inline(always)]
    fn eval(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        let _ = w; // Unused input; we use W for iteration state
        let mut state = self.seed.eval(x, y, z, w);

        // Track which lanes are still active via proper bit mask
        // 0.0 >= 0.0 is true, so this gives us all-1s (0xFFFFFFFF per lane)
        let zero = 0.0f32.eval(x, y, z, w);
        let mut active = BatchArithmetic::cmp_ge(zero, zero);

        loop {
            // Check termination
            let done_mask = self.done.eval(x, y, z, state);

            // Retire converged lanes: active = active AND NOT done
            // When done_mask is all-1s, select 0 (all-0s); otherwise keep active
            active = done_mask.select(0.0, active);

            // Early exit if all lanes converged
            if !active.any() {
                break;
            }

            // Step only active lanes
            let next = self.step.eval(x, y, z, state);
            state = BatchArithmetic::select(active, next, state);
        }

        state
    }
}
