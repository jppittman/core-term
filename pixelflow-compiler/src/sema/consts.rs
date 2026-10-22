//! Counts and `const`s: a `usize` expression checked as rustc's const
//! evaluator checks it, a fold's bounds, and every `const`'s value.

use crate::ast::{BinaryOp, ConstItem, Expr, Name, PLAN, RangeExpr, UnaryOp, same_name};
use pixelflow_ir::{Binder, Fold};
use proc_macro2::{Span, TokenStream};
use quote::quote;
use std::collections::HashMap;
use std::ops::Range;
use syn::Ident;

use super::analyze::{ConstValue, Ty};
use super::body::a_count_is_not_a_value;

/// A count: a `usize` evaluated at expansion, or one that names an entry's
/// structural parameters and so has a value only once the host function is
/// instantiated with them (plan §1.4).
#[derive(Debug, Clone)]
enum Count {
    /// Evaluated here, each operation checked.
    Known(u64),
    /// A Rust `usize` expression over literals and structural parameters:
    /// rustc evaluates it, per instantiation, in a `const` block, and checks
    /// each operation there as it checks any `const`.
    Structural(TokenStream),
}

impl Count {
    /// The count as a Rust `usize` expression.
    fn tokens(&self) -> TokenStream {
        match self {
            Count::Known(value) => {
                let value = proc_macro2::Literal::u64_unsuffixed(*value);
                quote!(#value)
            }
            Count::Structural(tokens) => tokens.clone(),
        }
    }
}

/// One of a count's operations on two known counts, `None` where rustc's
/// const evaluator would refuse it.
type CheckedOp = fn(u64, u64) -> Option<u64>;

/// Integer arithmetic over counts: what a `usize` const's initializer and a
/// fold's bounds are built from. Each operation on two known counts is
/// checked here, as rustc's const evaluator checks it, so a count that would
/// wrap is an error rather than a different count; one that names a
/// structural parameter is checked by rustc itself, when the host function
/// is instantiated.
///
/// A trait because its scopes differ only in what a name means: a `const`
/// being evaluated may name another, evaluated on demand; a fold's bound
/// names a `usize` const, already evaluated, or a structural parameter.
/// Anything else is not constant.
trait UsizeScope {
    /// The value of `name` in a `usize` expression, or why it has none.
    fn usize_named(&mut self, name: &Ident) -> syn::Result<Count>;

    /// The refusal of something a `usize` expression is not built from.
    fn not_a_count(&self, span: Span) -> syn::Error;

    /// An integer literal, a name, `+ - * /` and parentheses.
    fn eval_usize(&mut self, expr: &Expr) -> syn::Result<Count> {
        let binary = match expr {
            Expr::Literal(literal) => return literal.usize_value().map(Count::Known),
            Expr::Ident(name) => return self.usize_named(name),
            Expr::Paren(inner) => return self.eval_usize(inner),
            // Constant, but no count: rustc's refusal of the same tokens.
            Expr::Unary(unary) => match unary.op {
                UnaryOp::Neg => {
                    return Err(syn::Error::new(
                        unary.span,
                        "cannot apply unary operator `-` to type `usize`\n\
                         \n\
                         note: unsigned values cannot be negated; a range's bounds and a \
                         `usize` const are counts",
                    ));
                }
            },
            Expr::Binary(binary) => binary,
            other => return Err(self.not_a_count(other.span())),
        };
        let lhs = self.eval_usize(&binary.lhs)?;
        let rhs = self.eval_usize(&binary.rhs)?;
        let (checked, symbol): (CheckedOp, TokenStream) = match binary.op {
            BinaryOp::Add => (u64::checked_add, quote!(+)),
            BinaryOp::Sub => (u64::checked_sub, quote!(-)),
            BinaryOp::Mul => (u64::checked_mul, quote!(*)),
            BinaryOp::Div => (u64::checked_div, quote!(/)),
            BinaryOp::Lt
            | BinaryOp::Le
            | BinaryOp::Gt
            | BinaryOp::Ge
            | BinaryOp::Eq
            | BinaryOp::Ne
            | BinaryOp::BitAnd
            | BinaryOp::BitOr => return Err(self.not_a_count(binary.span)),
        };
        let (Count::Known(lhs), Count::Known(rhs)) = (&lhs, &rhs) else {
            let (lhs, rhs) = (lhs.tokens(), rhs.tokens());
            return Ok(Count::Structural(quote!((#lhs #symbol #rhs))));
        };
        if binary.op == BinaryOp::Div && *rhs == 0 {
            return Err(syn::Error::new(
                binary.span,
                format!("attempt to divide `{lhs}_usize` by zero"),
            ));
        }
        checked(*lhs, *rhs).map(Count::Known).ok_or_else(|| {
            syn::Error::new(
                binary.span,
                format!(
                    "attempt to compute `{lhs}_usize {symbol} {rhs}_usize`, which would overflow"
                ),
            )
        })
    }
}

/// A fold's bounds: evaluated at expansion, or, where they name an entry's
/// structural parameters, when its host function is instantiated.
#[derive(Debug, Clone)]
pub(crate) enum Bounds {
    /// Both evaluated, the range runs forwards, and the IR admits it
    /// ([`Fold::admits`]).
    Known(Range<u32>),
    /// At least one names a structural parameter.
    Structural(StructuralRange),
}

/// `lo..hi` as Rust `usize` expressions over literals and an entry's
/// structural parameters: what the host function evaluates, per
/// instantiation, in a `const` block — where rustc checks each operation,
/// and the refusals [`range_bounds`] makes of a known range (backwards, past
/// 2²⁴) are made of this one.
#[derive(Debug, Clone)]
pub(crate) struct StructuralRange {
    pub(crate) lo: TokenStream,
    pub(crate) hi: TokenStream,
}

impl StructuralRange {
    /// The range as its source reads, for a diagnostic, and as the key two
    /// folds over one range share.
    pub(crate) fn text(&self) -> String {
        format!("{}..{}", self.lo, self.hi)
    }
}

/// A fold's bounds, each a `usize` built from integer literals, `usize`
/// consts and the entry's structural parameters, and — when both are known
/// here — the range running forwards.
///
/// The one evaluation of them, and the one admission: a range is refused
/// here when it runs backwards, or when the IR cannot hold it — its ends are
/// `u32` today (widening them is A5 of the plan, deprioritized), and its
/// index is an `f32` lane, exact only to [`Fold::EXACT_BOUND`], the tighter
/// of the two. Lowering reads the range this returns, so the two stages
/// cannot disagree on one.
///
/// Ranges are constant (plan §1.5): the program's shape is known when it is
/// compiled, and a structural parameter is part of the program. A reversed
/// range is refused rather than read as the empty fold it would be in Rust:
/// an empty range is written `a..a`, and `b..a` is almost always a slip
/// (clippy's `reversed_empty_ranges` is deny-by-default for the same
/// reason), and the IR's own `Fold` refuses one.
///
/// # Errors
///
/// A bound that is not a constant `usize` expression, an overflow, or a range
/// that runs backwards or reaches past [`Fold::EXACT_BOUND`].
pub(crate) fn range_bounds(range: &RangeExpr, scope: RangeScope<'_>) -> syn::Result<Bounds> {
    let mut scope = scope;
    let lo = scope.eval_usize(&range.lo)?;
    let hi = scope.eval_usize(&range.hi)?;
    let (Count::Known(lo), Count::Known(hi)) = (&lo, &hi) else {
        return Ok(Bounds::Structural(StructuralRange {
            lo: lo.tokens(),
            hi: hi.tokens(),
        }));
    };
    if lo > hi {
        return Err(syn::Error::new(
            range.span,
            format!(
                "the range `{lo}..{hi}` runs backwards\n\
                 \n\
                 note: `{lo}..{hi}` holds no index, so a fold over it would be its monoid's \
                 identity; the empty range is written `a..a`, and a reversed one is refused as \
                 the slip it usually is"
            ),
        ));
    }
    let admitted = match (u32::try_from(*lo), u32::try_from(*hi)) {
        (Ok(lo), Ok(hi)) => Some(lo..hi),
        _ => None,
    };
    admitted
        .filter(Fold::admits)
        .map(Bounds::Known)
        .ok_or_else(|| {
            let bound = Fold::EXACT_BOUND;
            syn::Error::new(
                range.span,
                format!(
                    "the range `{lo}..{hi}` reaches past {bound} (2^24): a fold's index is an \
                     `f32` lane, which names every integer only that far, so past it indices \
                     would round together and the fold would not be the one written \
                     (`Fold::EXACT_BOUND`)\n\
                     note: a fold's ends are also `u32` in the IR today (`Fold`); 64-bit fold \
                     ends are A5 of {PLAN}, deprioritized, and would not widen the lane"
                ),
            )
        })
}

/// The refusal of a fold opened with every binder of the IR already taken.
///
/// `sema` makes it for the folds one body nests; lowering makes it again
/// where an inlined helper's fold nests inside a caller's, the one depth
/// `sema` does not see.
pub(crate) fn folds_nested_too_deep(span: Span) -> syn::Error {
    syn::Error::new(
        span,
        format!(
            "folds nested more than {} deep: the IR binds at most that many indices at once \
             (`Binder::COUNT`)",
            Binder::COUNT
        ),
    )
}

/// What a fold's bound may name: a `usize` const, already evaluated, or one
/// of the entry's structural parameters.
#[derive(Clone, Copy)]
pub(crate) struct RangeScope<'a> {
    pub(crate) consts: &'a HashMap<Name, ConstValue>,
    pub(crate) structural: &'a [Ident],
}

impl UsizeScope for RangeScope<'_> {
    fn usize_named(&mut self, name: &Ident) -> syn::Result<Count> {
        match self.consts.get(&Name::of(name)) {
            Some(ConstValue::Usize(value)) => Ok(Count::Known(*value)),
            Some(ConstValue::F32(_)) => Err(syn::Error::new(
                name.span(),
                format!(
                    "`{name}` is an `f32` const, and a range's bounds are `usize`s\n\
                     \n\
                     help: declare the count as `const {name}: usize`"
                ),
            )),
            None if self.structural.iter().any(|s| same_name(s, name)) => {
                Ok(Count::Structural(quote!(#name)))
            }
            None => Err(syn::Error::new(
                name.span(),
                format!(
                    "a range's bounds are constant, and `{name}` is not a `const`\n\
                     \n\
                     note: a bound is built from integer literals, `usize` consts, the entry's \
                     structural parameters (`const N: usize`), `+ - * /` and parentheses; a \
                     fold's index, a parameter, a `let` and a coordinate are not constant\n\
                     note: ranges are constant: a program's shape is known when it is \
                     compiled (§1.5 of {PLAN})"
                ),
            )),
        }
    }

    fn not_a_count(&self, span: Span) -> syn::Error {
        syn::Error::new(
            span,
            format!(
                "a range's bounds are constant\n\
                 \n\
                 note: a bound is built from integer literals, `usize` consts, the entry's \
                 structural parameters, `+ - * /` and parentheses\n\
                 note: ranges are constant: a program's shape is known when it is compiled \
                 (§1.5 of {PLAN})"
            ),
        )
    }
}

/// Every `const`'s value.
///
/// A `const` is evaluated here, at expansion, and gets the value rustc gives
/// the same tokens. An `f32` const is evaluated per operation in `f32`,
/// since rustc evaluates an `f32` `const` in `f32` too. The discipline is
/// exactly the one rustc's const evaluator has: each operation rounds once,
/// in `f32`, and a product and a sum are never contracted into one rounding
/// — which matters because this crate is built with `-fp-contract=fast`
/// (`.cargo/config.toml`), and an FMA gives `b * c + d` a value two
/// roundings never reach. Its literals were rounded once by the parser, and
/// nothing here rounds again. A `usize` const is evaluated in 64-bit
/// integers, each operation checked ([`UsizeScope`]). A const may name
/// another declared anywhere in the block; a cycle is refused.
pub(super) fn evaluate_consts(
    consts: &[ConstItem],
    types: &HashMap<Name, Ty>,
) -> syn::Result<HashMap<Name, ConstValue>> {
    let mut evaluator = ConstEvaluator {
        items: consts.iter().map(|c| (Name::of(&c.name), c)).collect(),
        types,
        values: HashMap::with_capacity(consts.len()),
        in_progress: Vec::new(),
    };
    for c in consts {
        evaluator.value_of(&c.name)?;
    }
    Ok(evaluator.values)
}

struct ConstEvaluator<'a> {
    items: HashMap<Name, &'a ConstItem>,
    /// Each const's declared type, which decides how it is evaluated.
    types: &'a HashMap<Name, Ty>,
    values: HashMap<Name, ConstValue>,
    /// The consts whose initializers are being evaluated, outermost first:
    /// naming one of them again is a cycle.
    in_progress: Vec<Name>,
}

impl UsizeScope for ConstEvaluator<'_> {
    fn usize_named(&mut self, name: &Ident) -> syn::Result<Count> {
        match self.value_of(name)? {
            ConstValue::Usize(value) => Ok(Count::Known(value)),
            ConstValue::F32(_) => Err(syn::Error::new(
                name.span(),
                format!(
                    "`{name}` is an `f32` const, where a `usize` is expected\n\
                     \n\
                     note: a `usize` const is built from integers and other `usize` consts"
                ),
            )),
        }
    }

    fn not_a_count(&self, span: Span) -> syn::Error {
        Self::not_constant(span)
    }
}

impl ConstEvaluator<'_> {
    /// `f32` arithmetic at expansion: what an `f32` const's initializer is
    /// built from — literals, `f32` consts, a `usize` const `as f32`,
    /// `+ - * /`, unary `-` and parentheses — each operation in `f32`, as
    /// rustc's const evaluator does it (see [`evaluate_consts`]).
    fn eval_f32(&mut self, expr: &Expr) -> syn::Result<f32> {
        match expr {
            Expr::Literal(literal) => literal.f32_value(),
            Expr::Ident(name) => self.f32_named(name),
            Expr::Paren(inner) => self.eval_f32(inner),
            // Rust's own `as`, which rounds to the nearest `f32` as rustc's
            // does: exact up to 2²⁴. A `const` cannot name a structural
            // parameter — they are an entry's — so a count this does not
            // know is refused rather than assumed away.
            Expr::Cast(cast) => {
                let Some(name) = cast.named() else {
                    return Err(Self::not_constant(cast.span));
                };
                match self.usize_named(name)? {
                    Count::Known(count) => Ok(count as f32),
                    Count::Structural(_) => Err(Self::not_constant(cast.span)),
                }
            }
            Expr::Unary(unary) => match unary.op {
                UnaryOp::Neg => Ok(-self.eval_f32(&unary.operand)?),
            },
            Expr::Binary(binary) => {
                let lhs = self.eval_f32(&binary.lhs)?;
                let rhs = self.eval_f32(&binary.rhs)?;
                match binary.op {
                    BinaryOp::Add => Ok(lhs + rhs),
                    BinaryOp::Sub => Ok(lhs - rhs),
                    BinaryOp::Mul => Ok(lhs * rhs),
                    BinaryOp::Div => Ok(lhs / rhs),
                    BinaryOp::Lt
                    | BinaryOp::Le
                    | BinaryOp::Gt
                    | BinaryOp::Ge
                    | BinaryOp::Eq
                    | BinaryOp::Ne
                    | BinaryOp::BitAnd
                    | BinaryOp::BitOr => Err(Self::not_constant(binary.span)),
                }
            }
            other => Err(Self::not_constant(other.span())),
        }
    }

    /// The `f32` const `name`'s value; a `usize` const is a count, and is a
    /// value only as `name as f32`.
    fn f32_named(&mut self, name: &Ident) -> syn::Result<f32> {
        match self.value_of(name)? {
            ConstValue::F32(value) => Ok(value),
            ConstValue::Usize(_) => Err(a_count_is_not_a_value(name)),
        }
    }

    /// The value of the const `name`, evaluating it on first demand.
    fn value_of(&mut self, name: &Ident) -> syn::Result<ConstValue> {
        let key = Name::of(name);
        if let Some(&value) = self.values.get(&key) {
            return Ok(value);
        }
        let Some(item) = self.items.get(&key).copied() else {
            return Err(syn::Error::new(
                name.span(),
                format!(
                    "cannot find a `const` named `{key}`\n\
                     \n\
                     note: a `const` initializer is evaluated at expansion, so it names \
                     only literals and other `const`s of this block"
                ),
            ));
        };
        if self.in_progress.contains(&key) {
            let cycle = self
                .in_progress
                .iter()
                .map(Name::to_string)
                .collect::<Vec<_>>()
                .join("` → `");
            return Err(syn::Error::new(
                name.span(),
                format!(
                    "`{key}` depends on itself\n\
                     \n\
                     note: `{cycle}` → `{key}`"
                ),
            ));
        }
        self.in_progress.push(key.clone());
        let value = match self.types.get(&key) {
            Some(Ty::Usize) => ConstValue::Usize(self.known_usize(&item.init)?),
            Some(Ty::F32) => ConstValue::F32(self.eval_f32(&item.init)?),
            Some(Ty::Bool | Ty::Record(_) | Ty::Kernel) | None => {
                return Err(syn::Error::new_spanned(
                    &item.ty,
                    "a `const` in a `kernel!` block is an `f32` or a `usize`",
                ));
            }
        };
        self.in_progress.pop();
        self.values.insert(key, value);
        Ok(value)
    }

    /// A `usize` initializer's value, known here. Only a structural parameter
    /// leaves a count unknown, and a `const` item cannot name one — they are
    /// an entry's — so one that is not known is refused rather than assumed
    /// away.
    fn known_usize(&mut self, expr: &Expr) -> syn::Result<u64> {
        match self.eval_usize(expr)? {
            Count::Known(value) => Ok(value),
            Count::Structural(_) => Err(Self::not_constant(expr.span())),
        }
    }

    fn not_constant(span: Span) -> syn::Error {
        syn::Error::new(
            span,
            "a `const` initializer is evaluated at expansion\n\
             \n\
             note: an `f32` const is built from literals, other `f32` consts, a `usize` \
             const `as f32`, `+ - * /`, unary `-` and parentheses, each operation in `f32`\n\
             note: a `usize` const is built from integer literals, other `usize` consts, \
             `+ - * /` and parentheses, each operation checked",
        )
    }
}
