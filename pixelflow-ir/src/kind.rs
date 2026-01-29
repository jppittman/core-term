//! Operation Kind Enumeration.
//!
//! This enum provides a uniform representation of all operations.
//! It is used for storage in the e-graph and as the base for feature indices.

/// Unified enumeration of all IR operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum OpKind {
    // --- Basic Arithmetic ---
    Var = 0,
    Const = 1,
    Add = 2,
    Sub = 3,
    Mul = 4,
    Div = 5,
    Neg = 6,
    Sqrt = 7,
    Rsqrt = 8,
    Abs = 9,
    Min = 10,
    Max = 11,
    MulAdd = 12,
    MulRsqrt = 13,

    // --- Extended Math ---
    Recip = 14,
    Floor = 15,
    Ceil = 16,
    Round = 17,
    Fract = 18,

    // --- Trigonometry ---
    Sin = 19,
    Cos = 20,
    Tan = 21,
    Asin = 22,
    Acos = 23,
    Atan = 24,
    Atan2 = 30,

    // --- Exponentials ---
    Exp = 25,
    Exp2 = 26,
    Ln = 27,
    Log2 = 28,
    Log10 = 29,
    Pow = 31,
    Hypot = 32,

    // --- Comparison ---
    Lt = 33,
    Le = 34,
    Gt = 35,
    Ge = 36,
    Eq = 37,
    Ne = 38,

    // --- Control Flow ---
    Select = 39,
    Clamp = 40,

    // --- Structure ---
    Tuple = 41,
}

impl OpKind {
    /// Total number of operations.
    pub const COUNT: usize = 42;

    /// Convert index to OpKind.
    pub fn from_index(idx: usize) -> Option<Self> {
        if idx >= Self::COUNT {
            return None;
        }
        // SAFETY: repr(u8) and contiguous 0..=41
        unsafe { core::mem::transmute(idx as u8) }
    }

    /// Get the arity of the operation.
    pub fn arity(self) -> usize {
        match self {
            Self::Var | Self::Const | Self::Tuple => 0,
            
            Self::Neg | Self::Sqrt | Self::Rsqrt | Self::Abs |
            Self::Recip | Self::Floor | Self::Ceil | Self::Round | 
            Self::Fract | Self::Sin | Self::Cos | Self::Tan | 
            Self::Asin | Self::Acos | Self::Atan | Self::Exp | 
            Self::Exp2 | Self::Ln | Self::Log2 | Self::Log10 => 1,
            
            Self::Add | Self::Sub | Self::Mul | Self::Div |
            Self::Min | Self::Max | Self::MulRsqrt |
            Self::Atan2 | Self::Pow | Self::Hypot |
            Self::Lt | Self::Le | Self::Gt | Self::Ge | 
            Self::Eq | Self::Ne => 2,
            
            Self::MulAdd | Self::Select | Self::Clamp => 3,
        }
    }

    /// Get the display name of the operation.
    pub fn name(self) -> &'static str {
        match self {
            Self::Var => "var",
            Self::Const => "const",
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Neg => "neg",
            Self::Sqrt => "sqrt",
            Self::Rsqrt => "rsqrt",
            Self::Abs => "abs",
            Self::Min => "min",
            Self::Max => "max",
            Self::MulAdd => "mul_add",
            Self::MulRsqrt => "mul_rsqrt",
            Self::Recip => "recip",
            Self::Floor => "floor",
            Self::Ceil => "ceil",
            Self::Round => "round",
            Self::Fract => "fract",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Asin => "asin",
            Self::Acos => "acos",
            Self::Atan => "atan",
            Self::Atan2 => "atan2",
            Self::Exp => "exp",
            Self::Exp2 => "exp2",
            Self::Ln => "ln",
            Self::Log2 => "log2",
            Self::Log10 => "log10",
            Self::Pow => "pow",
            Self::Hypot => "hypot",
            Self::Lt => "lt",
            Self::Le => "le",
            Self::Gt => "gt",
            Self::Ge => "ge",
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Select => "select",
            Self::Clamp => "clamp",
            Self::Tuple => "tuple",
        }
    }
}