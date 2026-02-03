//! Mathematical rewrite rules organized by algebraic structure.
//!
//! This module provides categorical, trait-based rule derivation. Instead of
//! enumerating identities, we declare algebraic properties and derive rules.
//!
//! ## Module Organization
//!
//! - [`algebra`]: Core algebraic structures (InversePair, Commutative, Identity, etc.)
//! - [`parity`]: Even/odd function symmetry (sin is odd, cos is even)
//! - [`trig`]: Trigonometric identities (angle addition, Pythagorean)
//! - [`exp`]: Exponential/logarithmic identities (inverse pairs, homomorphisms)
//!
//! ## What's NOT Here
//!
//! CPU instruction selection rules (FMA, rsqrt) are NOT mathematical identities.
//! They belong in the compiler where the target architecture is known.
//! See `pixelflow_compiler::rewrite_rules` for those.
//!
//! ## Design Philosophy
//!
//! Rules are derived from algebraic properties, not enumerated:
//!
//! ```text
//! // One trait declaration...
//! impl InversePair for AddNeg {
//!     fn base() -> &'static dyn Op { &ops::Add }
//!     fn inverse() -> &'static dyn Op { &ops::Neg }
//!     fn derived() -> &'static dyn Op { &ops::Sub }
//!     fn identity() -> f32 { 0.0 }
//! }
//!
//! // ...yields four rules:
//! // - Canonicalize: a - b → a + neg(b)
//! // - Involution: neg(neg(x)) → x
//! // - Cancellation: (x + a) - a → x
//! // - InverseAnnihilation: x + neg(x) → 0
//! ```
//!
//! ## Categorical Structure
//!
//! The traits reflect mathematical categories:
//!
//! - **InversePair**: Group structure (operation + inverse + identity)
//! - **Parity**: Z₂ action (negation symmetry)
//! - **AngleAddition**: Lie group structure (angle as group element)
//! - **FunctionInverse**: Bijection (forward/backward maps)
//! - **Homomorphism**: Structure-preserving maps between algebraic structures
//!
//! The deep insight: Many identities are the same identity in different
//! presentations. For example, the exp Homomorphism (exp(a+b) = exp(a)*exp(b))
//! IS the trig angle addition rule via Euler's identity.

pub mod algebra;
pub mod exp;
pub mod parity;
pub mod trig;

use crate::egraph::rewrite::Rewrite;

// Re-export key types for convenience
pub use algebra::{
    InversePair, AddNeg, MulRecip,
    Commutative, Identity, Annihilator, Associative,
    algebra_rules, inverse_pair_rules, basic_algebra_rules,
};
pub use parity::{
    Parity, ParityKind,
    SinParity, CosParity, TanParity, AsinParity, AtanParity, AbsParity,
    parity_rules,
};
pub use trig::{
    AngleAddition, Sign, AngleExpansion,
    SinAngleAddition, CosAngleAddition,
    trig_rules,
};
pub use exp::{
    FunctionInverse, Homomorphism,
    ExpLn, Exp2Log2, ExpHomomorphism, LnHomomorphism,
    exp_rules,
};

/// All mathematical rewrite rules.
///
/// This is the primary entry point for getting all math rules. Categories:
/// - Algebra (14 rules): InversePair, commutative, identity, annihilator, etc.
/// - Parity (4 rules): sin, cos, tan, abs negation symmetry
/// - Trig (3 rules): angle addition, Pythagorean
/// - Exp (7 rules): inverse pairs, homomorphisms, power combine
///
/// Total: ~28 rules
///
/// Note: CPU instruction fusion rules (FMA, rsqrt) are NOT included here.
/// Those are compiler-specific and live in `pixelflow_compiler::rewrite_rules`.
pub fn all_math_rules() -> Vec<Box<dyn Rewrite>> {
    let mut rules = Vec::new();
    rules.extend(algebra_rules());
    rules.extend(parity_rules());
    rules.extend(trig_rules());
    rules.extend(exp_rules());
    rules
}

/// Core arithmetic rules only (fast, always applicable).
///
/// Use this for quick optimization passes where trig/exp rules
/// aren't needed.
pub fn core_rules() -> Vec<Box<dyn Rewrite>> {
    algebra_rules()
}

/// Transcendental function rules (trig, exp, log).
///
/// Use this when optimizing expressions with transcendental functions.
pub fn transcendental_rules() -> Vec<Box<dyn Rewrite>> {
    let mut rules = Vec::new();
    rules.extend(parity_rules());
    rules.extend(trig_rules());
    rules.extend(exp_rules());
    rules
}
