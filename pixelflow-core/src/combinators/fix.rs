//! # Fix Combinator
//!
//! Iterate until convergence (fixed-point iteration).

use crate::{Field, Manifold};

type Field4 = (Field, Field, Field, Field);

/// Iterate until convergence.
///
/// Evaluates `step` repeatedly (using W as state) until `done` returns true.
/// Uses per-lane retirement for efficiency.
#[derive(Clone, Debug)]
pub struct Fix<Seed, Step, Done> {
    /// Initial state value.
    pub seed: Seed,
    /// Step function: computes next state from current (via W).
    pub step: Step,
    /// Termination condition: returns true when lane has converged.
    pub done: Done,
}

impl<Seed, Step, Done> Manifold<Field4> for Fix<Seed, Step, Done>
where
    Seed: Manifold<Field4, Output = Field>,
    Step: Manifold<Field4, Output = Field>,
    Done: Manifold<Field4, Output = Field>,
{
    type Output = Field;
    #[inline(always)]
    fn eval(&self, p: Field4) -> Field {
        let (x, y, z, _w) = p;
        let mut state = self.seed.eval(p);

        // Track which lanes are still active (all-1s mask initially)
        let zero = Field::from(0.0f32);
        let mut active = zero.ge(zero); // 0 >= 0 is always true -> all-1s mask

        loop {
            // Check termination
            let done_mask = self.done.eval((x, y, z, state));

            // Retire converged lanes: active = active AND NOT done
            active = Field::select_raw(done_mask, zero, active);

            // Early exit if all lanes converged
            if !active.any() {
                break;
            }

            // Step only active lanes
            let next = self.step.eval((x, y, z, state));
            state = Field::select_raw(active, next, state);
        }

        state
    }
}
