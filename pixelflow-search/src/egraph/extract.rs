//! Expression tree for extracted expressions.

/// A simplified expression tree extracted from the e-graph.
#[derive(Clone, Debug, PartialEq)]
pub enum ExprTree {
    Var(u8),
    Const(f32),
    Add(Box<ExprTree>, Box<ExprTree>),
    Sub(Box<ExprTree>, Box<ExprTree>),
    Mul(Box<ExprTree>, Box<ExprTree>),
    Div(Box<ExprTree>, Box<ExprTree>),
    Neg(Box<ExprTree>),
    Recip(Box<ExprTree>),
    Sqrt(Box<ExprTree>),
    Rsqrt(Box<ExprTree>),
    Abs(Box<ExprTree>),
    Min(Box<ExprTree>, Box<ExprTree>),
    Max(Box<ExprTree>, Box<ExprTree>),
    MulAdd(Box<ExprTree>, Box<ExprTree>, Box<ExprTree>),
    Floor(Box<ExprTree>),
    Ceil(Box<ExprTree>),
    Round(Box<ExprTree>),
    Fract(Box<ExprTree>),
    Sin(Box<ExprTree>),
    Cos(Box<ExprTree>),
    Tan(Box<ExprTree>),
    Asin(Box<ExprTree>),
    Acos(Box<ExprTree>),
    Atan(Box<ExprTree>),
    Exp(Box<ExprTree>),
    Exp2(Box<ExprTree>),
    Ln(Box<ExprTree>),
    Log2(Box<ExprTree>),
    Log10(Box<ExprTree>),
    Atan2(Box<ExprTree>, Box<ExprTree>),
    Pow(Box<ExprTree>, Box<ExprTree>),
    Hypot(Box<ExprTree>, Box<ExprTree>),
    Lt(Box<ExprTree>, Box<ExprTree>),
    Le(Box<ExprTree>, Box<ExprTree>),
    Gt(Box<ExprTree>, Box<ExprTree>),
    Ge(Box<ExprTree>, Box<ExprTree>),
    Eq(Box<ExprTree>, Box<ExprTree>),
    Ne(Box<ExprTree>, Box<ExprTree>),
    Select(Box<ExprTree>, Box<ExprTree>, Box<ExprTree>),
    Clamp(Box<ExprTree>, Box<ExprTree>, Box<ExprTree>),
    Tuple(Vec<ExprTree>),
}

#[cfg(test)]
impl ExprTree {
    pub fn eval(&self, vars: &[f32; 4]) -> f32 {
        match self {
            ExprTree::Var(i) => vars[*i as usize],
            ExprTree::Const(c) => *c,
            ExprTree::Add(a, b) => a.eval(vars) + b.eval(vars),
            ExprTree::Sub(a, b) => a.eval(vars) - b.eval(vars),
            ExprTree::Mul(a, b) => a.eval(vars) * b.eval(vars),
            ExprTree::Div(a, b) => a.eval(vars) / b.eval(vars),
            ExprTree::Neg(a) => -a.eval(vars),
            ExprTree::Recip(a) => 1.0 / a.eval(vars),
            ExprTree::Sqrt(a) => a.eval(vars).sqrt(),
            ExprTree::Rsqrt(a) => 1.0 / a.eval(vars).sqrt(),
            ExprTree::Abs(a) => a.eval(vars).abs(),
            ExprTree::Min(a, b) => a.eval(vars).min(b.eval(vars)),
            ExprTree::Max(a, b) => a.eval(vars).max(b.eval(vars)),
            ExprTree::MulAdd(a, b, c) => a.eval(vars) * b.eval(vars) + c.eval(vars),
            _ => todo!("Implement eval for more variants if needed"),
        }
    }
}
