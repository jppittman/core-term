//! Operation definitions for e-graph expressions.
//!
//! Each operation is a unit struct implementing the `Op` trait.
//! Adding a new operation = define struct + impl Op. That's it.

/// Trait for operations.
pub trait Op: 'static + Send + Sync {
    fn name(&self) -> &'static str;
    fn default_cost(&self) -> usize { 1 }

    // Algebraic properties for rewrite rules
    fn is_commutative(&self) -> bool { false }
    fn is_associative(&self) -> bool { false }
    fn identity(&self) -> Option<f32> { None }
    fn annihilator(&self) -> Option<f32> { None }
    fn is_idempotent(&self) -> bool { false }
}

impl core::fmt::Debug for dyn Op {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Op({})", self.name())
    }
}

// === Arithmetic ===

pub struct Add;
impl Op for Add {
    fn name(&self) -> &'static str { "add" }
    fn default_cost(&self) -> usize { 4 }
    fn is_commutative(&self) -> bool { true }
    fn is_associative(&self) -> bool { true }
    fn identity(&self) -> Option<f32> { Some(0.0) }
}

pub struct Sub;
impl Op for Sub {
    fn name(&self) -> &'static str { "sub" }
    fn default_cost(&self) -> usize { 4 }
}

pub struct Mul;
impl Op for Mul {
    fn name(&self) -> &'static str { "mul" }
    fn default_cost(&self) -> usize { 5 }
    fn is_commutative(&self) -> bool { true }
    fn is_associative(&self) -> bool { true }
    fn identity(&self) -> Option<f32> { Some(1.0) }
    fn annihilator(&self) -> Option<f32> { Some(0.0) }
}

pub struct Div;
impl Op for Div {
    fn name(&self) -> &'static str { "div" }
    fn default_cost(&self) -> usize { 15 }
}

pub struct Neg;
impl Op for Neg {
    fn name(&self) -> &'static str { "neg" }
}

pub struct Recip;
impl Op for Recip {
    fn name(&self) -> &'static str { "recip" }
    fn default_cost(&self) -> usize { 5 }
}

// === Roots ===

pub struct Sqrt;
impl Op for Sqrt {
    fn name(&self) -> &'static str { "sqrt" }
    fn default_cost(&self) -> usize { 15 }
}

pub struct Rsqrt;
impl Op for Rsqrt {
    fn name(&self) -> &'static str { "rsqrt" }
    fn default_cost(&self) -> usize { 5 }
}

// === Misc math ===

pub struct Abs;
impl Op for Abs {
    fn name(&self) -> &'static str { "abs" }
}

pub struct Min;
impl Op for Min {
    fn name(&self) -> &'static str { "min" }
    fn default_cost(&self) -> usize { 4 }
    fn is_commutative(&self) -> bool { true }
    fn is_associative(&self) -> bool { true }
    fn is_idempotent(&self) -> bool { true }
}

pub struct Max;
impl Op for Max {
    fn name(&self) -> &'static str { "max" }
    fn default_cost(&self) -> usize { 4 }
    fn is_commutative(&self) -> bool { true }
    fn is_associative(&self) -> bool { true }
    fn is_idempotent(&self) -> bool { true }
}

pub struct MulAdd;
impl Op for MulAdd {
    fn name(&self) -> &'static str { "mul_add" }
    fn default_cost(&self) -> usize { 5 }
}

// === Rounding ===

pub struct Floor;
impl Op for Floor {
    fn name(&self) -> &'static str { "floor" }
}

pub struct Ceil;
impl Op for Ceil {
    fn name(&self) -> &'static str { "ceil" }
}

pub struct Round;
impl Op for Round {
    fn name(&self) -> &'static str { "round" }
}

pub struct Fract;
impl Op for Fract {
    fn name(&self) -> &'static str { "fract" }
}

// === Trig ===

pub struct Sin;
impl Op for Sin {
    fn name(&self) -> &'static str { "sin" }
    fn default_cost(&self) -> usize { 15 }
}

pub struct Cos;
impl Op for Cos {
    fn name(&self) -> &'static str { "cos" }
    fn default_cost(&self) -> usize { 15 }
}

pub struct Tan;
impl Op for Tan {
    fn name(&self) -> &'static str { "tan" }
    fn default_cost(&self) -> usize { 15 }
}

pub struct Asin;
impl Op for Asin {
    fn name(&self) -> &'static str { "asin" }
    fn default_cost(&self) -> usize { 15 }
}

pub struct Acos;
impl Op for Acos {
    fn name(&self) -> &'static str { "acos" }
    fn default_cost(&self) -> usize { 15 }
}

pub struct Atan;
impl Op for Atan {
    fn name(&self) -> &'static str { "atan" }
    fn default_cost(&self) -> usize { 15 }
}

pub struct Atan2;
impl Op for Atan2 {
    fn name(&self) -> &'static str { "atan2" }
    fn default_cost(&self) -> usize { 15 }
}

// === Exponential/Log ===

pub struct Exp;
impl Op for Exp {
    fn name(&self) -> &'static str { "exp" }
    fn default_cost(&self) -> usize { 15 }
}

pub struct Exp2;
impl Op for Exp2 {
    fn name(&self) -> &'static str { "exp2" }
    fn default_cost(&self) -> usize { 15 }
}

pub struct Ln;
impl Op for Ln {
    fn name(&self) -> &'static str { "ln" }
    fn default_cost(&self) -> usize { 15 }
}

pub struct Log2;
impl Op for Log2 {
    fn name(&self) -> &'static str { "log2" }
    fn default_cost(&self) -> usize { 15 }
}

pub struct Log10;
impl Op for Log10 {
    fn name(&self) -> &'static str { "log10" }
    fn default_cost(&self) -> usize { 15 }
}

pub struct Pow;
impl Op for Pow {
    fn name(&self) -> &'static str { "pow" }
    fn default_cost(&self) -> usize { 15 }
}

pub struct Hypot;
impl Op for Hypot {
    fn name(&self) -> &'static str { "hypot" }
    fn default_cost(&self) -> usize { 15 }
}

// === Comparison ===

pub struct Lt;
impl Op for Lt {
    fn name(&self) -> &'static str { "lt" }
    fn default_cost(&self) -> usize { 4 }
}

pub struct Le;
impl Op for Le {
    fn name(&self) -> &'static str { "le" }
    fn default_cost(&self) -> usize { 4 }
}

pub struct Gt;
impl Op for Gt {
    fn name(&self) -> &'static str { "gt" }
    fn default_cost(&self) -> usize { 4 }
}

pub struct Ge;
impl Op for Ge {
    fn name(&self) -> &'static str { "ge" }
    fn default_cost(&self) -> usize { 4 }
}

pub struct Eq;
impl Op for Eq {
    fn name(&self) -> &'static str { "eq" }
    fn default_cost(&self) -> usize { 4 }
}

pub struct Ne;
impl Op for Ne {
    fn name(&self) -> &'static str { "ne" }
    fn default_cost(&self) -> usize { 4 }
}

// === Control flow ===

pub struct Select;
impl Op for Select {
    fn name(&self) -> &'static str { "select" }
    fn default_cost(&self) -> usize { 4 }
}

pub struct Clamp;
impl Op for Clamp {
    fn name(&self) -> &'static str { "clamp" }
    fn default_cost(&self) -> usize { 4 }
}

// === Aggregates ===

pub struct Tuple;
impl Op for Tuple {
    fn name(&self) -> &'static str { "tuple" }
    fn default_cost(&self) -> usize { 0 }
}
