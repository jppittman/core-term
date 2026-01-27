//! Concrete Operation Structs (ZSTs).
//!
//! Each struct represents a specific operation in the IR as a Zero-Sized Type.
//! These implement the `Op` trait with type-level constants.

use crate::traits::Op;
use crate::kind::OpKind;

macro_rules! define_op {
    ($name:ident, $str_name:expr, $arity:expr, $kind:expr) => {
        #[doc = $str_name]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub struct $name;
        
        impl Op for $name {
            const ARITY: usize = $arity;
            const NAME: &'static str = $str_name;
            const KIND: OpKind = $kind;
        }
    };
}

// --- Basic Arithmetic ---
define_op!(Var, "var", 0, OpKind::Var);
define_op!(Const, "const", 0, OpKind::Const);
define_op!(Add, "add", 2, OpKind::Add);
define_op!(Sub, "sub", 2, OpKind::Sub);
define_op!(Mul, "mul", 2, OpKind::Mul);
define_op!(Div, "div", 2, OpKind::Div);
define_op!(Neg, "neg", 1, OpKind::Neg);
define_op!(Sqrt, "sqrt", 1, OpKind::Sqrt);
define_op!(Rsqrt, "rsqrt", 1, OpKind::Rsqrt);
define_op!(Abs, "abs", 1, OpKind::Abs);
define_op!(Min, "min", 2, OpKind::Min);
define_op!(Max, "max", 2, OpKind::Max);
define_op!(MulAdd, "mul_add", 3, OpKind::MulAdd);
define_op!(MulRsqrt, "mul_rsqrt", 2, OpKind::MulRsqrt);

// --- Extended Math ---
define_op!(Recip, "recip", 1, OpKind::Recip);
define_op!(Floor, "floor", 1, OpKind::Floor);
define_op!(Ceil, "ceil", 1, OpKind::Ceil);
define_op!(Round, "round", 1, OpKind::Round);
define_op!(Fract, "fract", 1, OpKind::Fract);

// --- Trigonometry ---
define_op!(Sin, "sin", 1, OpKind::Sin);
define_op!(Cos, "cos", 1, OpKind::Cos);
define_op!(Tan, "tan", 1, OpKind::Tan);
define_op!(Asin, "asin", 1, OpKind::Asin);
define_op!(Acos, "acos", 1, OpKind::Acos);
define_op!(Atan, "atan", 1, OpKind::Atan);
define_op!(Atan2, "atan2", 2, OpKind::Atan2);

// --- Exponentials ---
define_op!(Exp, "exp", 1, OpKind::Exp);
define_op!(Exp2, "exp2", 1, OpKind::Exp2);
define_op!(Ln, "ln", 1, OpKind::Ln);
define_op!(Log2, "log2", 1, OpKind::Log2);
define_op!(Log10, "log10", 1, OpKind::Log10);
define_op!(Pow, "pow", 2, OpKind::Pow);
define_op!(Hypot, "hypot", 2, OpKind::Hypot);

// --- Comparison ---
define_op!(Lt, "lt", 2, OpKind::Lt);
define_op!(Le, "le", 2, OpKind::Le);
define_op!(Gt, "gt", 2, OpKind::Gt);
define_op!(Ge, "ge", 2, OpKind::Ge);
define_op!(Eq, "eq", 2, OpKind::Eq);
define_op!(Ne, "ne", 2, OpKind::Ne);

// --- Control Flow ---
define_op!(Select, "select", 3, OpKind::Select);
define_op!(Clamp, "clamp", 3, OpKind::Clamp);

// --- Structure ---
define_op!(Tuple, "tuple", 0, OpKind::Tuple);
