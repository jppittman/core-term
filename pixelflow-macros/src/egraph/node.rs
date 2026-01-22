//! Core e-graph data structures: EClassId, Op, and ENode.

/// Identifier for an equivalence class in the e-graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EClassId(pub(crate) u32);

impl EClassId {
    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

/// Identifies the kind of operation for generic algebraic rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Recip,
    Min,
    Max,
    Sqrt,
    Rsqrt,
    Abs,
    MulAdd,
}

impl Op {
    /// Is this operation commutative? (a op b == b op a)
    pub fn is_commutative(&self) -> bool {
        matches!(self, Op::Add | Op::Mul | Op::Min | Op::Max)
    }

    /// Is this operation associative? ((a op b) op c == a op (b op c))
    pub fn is_associative(&self) -> bool {
        matches!(self, Op::Add | Op::Mul | Op::Min | Op::Max)
    }

    /// Identity element for this operation (a op I == a)
    pub fn identity(&self) -> Option<f32> {
        match self {
            Op::Add => Some(0.0),
            Op::Mul => Some(1.0),
            _ => None,
        }
    }

    /// Annihilator element for this operation (a op Z == Z)
    pub fn annihilator(&self) -> Option<f32> {
        match self {
            Op::Mul => Some(0.0),
            _ => None,
        }
    }

    /// Idempotence (a op a == a)
    pub fn is_idempotent(&self) -> bool {
        matches!(self, Op::Min | Op::Max)
    }

    /// Does this operation have an inverse?
    /// Returns (inverse_op, identity_element).
    pub fn inverse(&self) -> Option<(Op, f32)> {
        match self {
            Op::Add => Some((Op::Neg, 0.0)),
            Op::Mul => Some((Op::Recip, 1.0)),
            _ => None,
        }
    }

    /// Construct a binary ENode from this operator.
    pub fn make_binary(&self, a: EClassId, b: EClassId) -> Option<ENode> {
        match self {
            Op::Add => Some(ENode::Add(a, b)),
            Op::Sub => Some(ENode::Sub(a, b)),
            Op::Mul => Some(ENode::Mul(a, b)),
            Op::Div => Some(ENode::Div(a, b)),
            Op::Min => Some(ENode::Min(a, b)),
            Op::Max => Some(ENode::Max(a, b)),
            Op::Neg | Op::Recip | Op::Sqrt | Op::Rsqrt | Op::Abs => None,
            Op::MulAdd => None,
        }
    }

    /// Construct a unary ENode from this operator.
    pub fn make_unary(&self, a: EClassId) -> Option<ENode> {
        match self {
            Op::Neg => Some(ENode::Neg(a)),
            Op::Recip => Some(ENode::Recip(a)),
            Op::Sqrt => Some(ENode::Sqrt(a)),
            Op::Rsqrt => Some(ENode::Rsqrt(a)),
            Op::Abs => Some(ENode::Abs(a)),
            _ => None,
        }
    }
}

/// An expression node in the e-graph.
///
/// Children point to e-classes (not other nodes), enabling compact
/// representation of equivalent expressions.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ENode {
    // === Core ===
    Var(u8),
    Const(u32),

    // === Arithmetic (optimizable) ===
    Add(EClassId, EClassId),
    Sub(EClassId, EClassId),
    Mul(EClassId, EClassId),
    Div(EClassId, EClassId),
    Neg(EClassId),
    Recip(EClassId),
    Sqrt(EClassId),
    Rsqrt(EClassId),
    Abs(EClassId),
    Min(EClassId, EClassId),
    Max(EClassId, EClassId),
    MulAdd(EClassId, EClassId, EClassId),

    // === Pass-through unary ===
    Floor(EClassId),
    Ceil(EClassId),
    Round(EClassId),
    Fract(EClassId),
    Sin(EClassId),
    Cos(EClassId),
    Tan(EClassId),
    Asin(EClassId),
    Acos(EClassId),
    Atan(EClassId),
    Exp(EClassId),
    Exp2(EClassId),
    Ln(EClassId),
    Log2(EClassId),
    Log10(EClassId),

    // === Pass-through binary ===
    Atan2(EClassId, EClassId),
    Pow(EClassId, EClassId),
    Hypot(EClassId, EClassId),

    // === Comparisons ===
    Lt(EClassId, EClassId),
    Le(EClassId, EClassId),
    Gt(EClassId, EClassId),
    Ge(EClassId, EClassId),
    Eq(EClassId, EClassId),
    Ne(EClassId, EClassId),

    // === Selection/Ternary ===
    Select(EClassId, EClassId, EClassId),
    Clamp(EClassId, EClassId, EClassId),
}

impl ENode {
    pub fn constant(val: f32) -> Self {
        ENode::Const(val.to_bits())
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            ENode::Const(bits) => Some(f32::from_bits(*bits)),
            _ => None,
        }
    }

    pub fn is_const(&self, val: f32) -> bool {
        self.as_f32() == Some(val)
    }

    pub fn op(&self) -> Option<Op> {
        match self {
            ENode::Add(_, _) => Some(Op::Add),
            ENode::Sub(_, _) => Some(Op::Sub),
            ENode::Mul(_, _) => Some(Op::Mul),
            ENode::Div(_, _) => Some(Op::Div),
            ENode::Neg(_) => Some(Op::Neg),
            ENode::Recip(_) => Some(Op::Recip),
            ENode::Min(_, _) => Some(Op::Min),
            ENode::Max(_, _) => Some(Op::Max),
            ENode::Sqrt(_) => Some(Op::Sqrt),
            ENode::Rsqrt(_) => Some(Op::Rsqrt),
            ENode::Abs(_) => Some(Op::Abs),
            ENode::MulAdd(_, _, _) => Some(Op::MulAdd),
            _ => None,
        }
    }

    pub fn binary_operands(&self) -> Option<(EClassId, EClassId)> {
        match self {
            ENode::Add(a, b)
            | ENode::Sub(a, b)
            | ENode::Mul(a, b)
            | ENode::Div(a, b)
            | ENode::Min(a, b)
            | ENode::Max(a, b)
            | ENode::Atan2(a, b)
            | ENode::Pow(a, b)
            | ENode::Hypot(a, b)
            | ENode::Lt(a, b)
            | ENode::Le(a, b)
            | ENode::Gt(a, b)
            | ENode::Ge(a, b)
            | ENode::Eq(a, b)
            | ENode::Ne(a, b) => Some((*a, *b)),
            _ => None,
        }
    }

    pub fn children(&self) -> Vec<EClassId> {
        match self {
            ENode::Var(_) | ENode::Const(_) => vec![],
            ENode::Add(a, b)
            | ENode::Sub(a, b)
            | ENode::Mul(a, b)
            | ENode::Div(a, b)
            | ENode::Min(a, b)
            | ENode::Max(a, b)
            | ENode::Atan2(a, b)
            | ENode::Pow(a, b)
            | ENode::Hypot(a, b)
            | ENode::Lt(a, b)
            | ENode::Le(a, b)
            | ENode::Gt(a, b)
            | ENode::Ge(a, b)
            | ENode::Eq(a, b)
            | ENode::Ne(a, b) => vec![*a, *b],
            ENode::Neg(a)
            | ENode::Recip(a)
            | ENode::Sqrt(a)
            | ENode::Rsqrt(a)
            | ENode::Abs(a)
            | ENode::Floor(a)
            | ENode::Ceil(a)
            | ENode::Round(a)
            | ENode::Fract(a)
            | ENode::Sin(a)
            | ENode::Cos(a)
            | ENode::Tan(a)
            | ENode::Asin(a)
            | ENode::Acos(a)
            | ENode::Atan(a)
            | ENode::Exp(a)
            | ENode::Exp2(a)
            | ENode::Ln(a)
            | ENode::Log2(a)
            | ENode::Log10(a) => vec![*a],
            ENode::MulAdd(a, b, c) | ENode::Select(a, b, c) | ENode::Clamp(a, b, c) => {
                vec![*a, *b, *c]
            }
        }
    }
}
