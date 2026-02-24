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

    /// Convert to array index.
    #[inline]
    #[must_use] 
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Convert index to OpKind.
    #[must_use] 
    pub fn from_index(idx: usize) -> Option<Self> {
        if idx >= Self::COUNT {
            return None;
        }
        // SAFETY: repr(u8) and contiguous 0..=41
        unsafe { core::mem::transmute(idx as u8) }
    }

    /// Get the arity of the operation.
    #[must_use] 
    pub const fn arity(self) -> usize {
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
    #[must_use] 
    pub const fn name(self) -> &'static str {
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

    /// Parse OpKind from its string name.
    #[must_use] 
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "var" => Some(Self::Var),
            "const" => Some(Self::Const),
            "add" => Some(Self::Add),
            "sub" => Some(Self::Sub),
            "mul" => Some(Self::Mul),
            "div" => Some(Self::Div),
            "neg" => Some(Self::Neg),
            "sqrt" => Some(Self::Sqrt),
            "rsqrt" => Some(Self::Rsqrt),
            "abs" => Some(Self::Abs),
            "min" => Some(Self::Min),
            "max" => Some(Self::Max),
            "mul_add" => Some(Self::MulAdd),
            "mul_rsqrt" => Some(Self::MulRsqrt),
            "recip" => Some(Self::Recip),
            "floor" => Some(Self::Floor),
            "ceil" => Some(Self::Ceil),
            "round" => Some(Self::Round),
            "fract" => Some(Self::Fract),
            "sin" => Some(Self::Sin),
            "cos" => Some(Self::Cos),
            "tan" => Some(Self::Tan),
            "asin" => Some(Self::Asin),
            "acos" => Some(Self::Acos),
            "atan" => Some(Self::Atan),
            "atan2" => Some(Self::Atan2),
            "exp" => Some(Self::Exp),
            "exp2" => Some(Self::Exp2),
            "ln" => Some(Self::Ln),
            "log2" => Some(Self::Log2),
            "log10" => Some(Self::Log10),
            "pow" => Some(Self::Pow),
            "hypot" => Some(Self::Hypot),
            "lt" => Some(Self::Lt),
            "le" => Some(Self::Le),
            "gt" => Some(Self::Gt),
            "ge" => Some(Self::Ge),
            "eq" => Some(Self::Eq),
            "ne" => Some(Self::Ne),
            "select" => Some(Self::Select),
            "clamp" => Some(Self::Clamp),
            "tuple" => Some(Self::Tuple),
            _ => None,
        }
    }

    /// Get the default cost estimate for this operation (in cycles).
    #[must_use] 
    pub const fn default_cost(self) -> usize {
        match self {
            Self::Var | Self::Const | Self::Tuple => 0,
            Self::Neg | Self::Abs | Self::Floor | Self::Ceil |
            Self::Round | Self::Fract => 1,
            Self::Add | Self::Sub | Self::Min | Self::Max |
            Self::Lt | Self::Le | Self::Gt | Self::Ge |
            Self::Eq | Self::Ne | Self::Select | Self::Clamp => 4,
            Self::Mul | Self::MulAdd | Self::MulRsqrt | Self::Recip | Self::Rsqrt => 5,
            Self::Div | Self::Sqrt | Self::Sin | Self::Cos | Self::Tan |
            Self::Asin | Self::Acos | Self::Atan | Self::Atan2 |
            Self::Exp | Self::Exp2 | Self::Ln | Self::Log2 | Self::Log10 |
            Self::Pow | Self::Hypot => 15,
        }
    }

    /// Returns true if the operation is commutative (a op b == b op a).
    #[must_use] 
    pub const fn is_commutative(self) -> bool {
        matches!(self, Self::Add | Self::Mul | Self::Min | Self::Max | Self::Eq | Self::Ne)
    }

    /// Returns true if the operation is associative ((a op b) op c == a op (b op c)).
    #[must_use] 
    pub const fn is_associative(self) -> bool {
        matches!(self, Self::Add | Self::Mul | Self::Min | Self::Max)
    }

    /// Returns the identity element if one exists (a op identity == a).
    #[must_use] 
    pub const fn identity(self) -> Option<f32> {
        match self {
            Self::Add | Self::Sub => Some(0.0),
            Self::Mul | Self::Div => Some(1.0),
            _ => None,
        }
    }

    /// Returns the annihilator element if one exists (a op annihilator == annihilator).
    #[must_use] 
    pub const fn annihilator(self) -> Option<f32> {
        match self {
            Self::Mul => Some(0.0),
            _ => None,
        }
    }

    /// Returns true if the operation is idempotent (a op a == a).
    #[must_use] 
    pub const fn is_idempotent(self) -> bool {
        matches!(self, Self::Min | Self::Max | Self::Abs)
    }
}