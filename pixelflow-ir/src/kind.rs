//! Operation Kind Enumeration.
//!
//! This enum provides a uniform representation of all operations.
//! It is used for storage in the e-graph and as the base for feature indices.
//!
//! Per-op tables belong in [`OpMap`], not in a bare `[T; OpKind::COUNT]`.
//! A bare array makes each consumer responsible for turning an op into a
//! subscript, and every consumer that did got it wrong the same way.

use crate::traits::EmitStyle;
use core::ops::{Index, IndexMut};

/// Unified enumeration of all IR operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

    // --- Extended Math ---
    Recip = 13,
    Floor = 14,
    Ceil = 15,
    Round = 16,

    // --- Trigonometry ---
    Sin = 17,
    Cos = 18,
    Tan = 19,
    Asin = 20,
    Acos = 21,
    Atan = 22,
    Atan2 = 23,

    // --- Exponentials ---
    Exp = 24,
    Exp2 = 25,
    Ln = 26,
    Log2 = 27,
    Log10 = 28,
    Pow = 29,

    // --- Comparison ---
    Lt = 30,
    Le = 31,
    Gt = 32,
    Ge = 33,
    Eq = 34,
    Ne = 35,

    // --- Control Flow ---
    Select = 36,

    // --- Structure ---
    Tuple = 37,

    // --- Bit manipulation (integer-domain primitives) ---
    // These let exp/ln/log lower to arithmetic: each maps 1:1 to a single
    // hardware instruction. Lanes are reinterpreted as i32 where noted;
    // "reinterpret" itself is free (same register) and is not an op.
    /// Truncate f32 lanes to i32 (`cvttps2dq` / `fcvtzs`).
    TruncToInt = 38,
    /// Convert i32 lanes to f32 (`cvtdq2ps` / `scvtf`).
    IntToFloat = 39,
    /// Integer (i32) lane-wise add (`paddd` / `add .4s`).
    IAdd = 40,
    /// Logical shift-left of i32 lanes; RHS is a `Const` shift amount.
    Shl = 41,
    /// Logical shift-right of i32 lanes; RHS is a `Const` shift amount.
    Shr = 42,
    /// Bitwise AND of lane bit patterns (`andps` / `and`).
    BitAnd = 43,
    /// Bitwise OR of lane bit patterns (`orps` / `orr`).
    BitOr = 44,

    // --- Differentiation ---
    /// Symbolic derivative `∂(child0)/∂(var)` where `child1` is a `Const` whose
    /// value is the variable index (0=X, 1=Y, 2=Z, 3=W). The e-graph pushes
    /// `Dwrt` toward the leaves via the chain rule at optimization time;
    /// whatever survives (budget miss, unsupported op) falls to the runtime
    /// `lower_dwrt` pass, which either eliminates it or refuses to compile.
    /// It must never reach a backend.
    Dwrt = 45,

    // --- Bound memory (lattices) ---
    /// Buffer leaf: a slot referencing a `BufferDecl` in the arena's buffer
    /// table. The declared extents are static IR; the contents are bound at
    /// JIT-compile time. See `docs/designs/KERNELS_AND_LATTICES.md`.
    Buffer = 46,
    /// Read a bound buffer: `Gather(buffer, x, y)` where `buffer` is a
    /// `Buffer` leaf. Semantics match `DiscreteManifold::eval`: floor the
    /// indices, clamp to the declared extents, gather row-major.
    Gather = 47,
    /// Primitive gather: `RawGather(buffer, index)` reads `buffer`'s contents
    /// at the already-computed linear lane `index` (truncated to int), with no
    /// floor/clamp/row-major math. `Gather` lowers to index arithmetic (built
    /// from existing ops) plus this primitive — the analogue of `raw_mul` under
    /// `mul`. The index is trusted to be in bounds (the lowering clamps it).
    RawGather = 48,

    // --- Reduction (lattice fold) ---
    /// Fold a body over a bounded domain. Encoded
    /// `Nary(Reduce, [Const(combiner), Const(reduce_var), Const(extent), body])`:
    /// `combiner` is the monoid op index (`Add`/`Mul`/`Min`/`Max`), `body`
    /// references `Var(reduce_var)` (indices 4..8), and the fold runs over
    /// `0..extent`. The combiner is a *child* (a parameter), not baked into the
    /// opcode, so one `Reduce` covers every monoid and can later take an
    /// arbitrary combiner function. Lowered to an unrolled accumulation by
    /// `expand_reduce` before codegen — the analogue of `Gather -> RawGather`.
    Reduce = 49,
}

impl OpKind {
    /// Total number of operations.
    pub const COUNT: usize = 50;

    /// Monoid identity for an op usable as a reduction combiner
    /// (`Add`→0, `Mul`→1, `Min`→+∞, `Max`→−∞). `None` if `self` is not a valid
    /// combiner.
    #[must_use]
    pub(crate) const fn monoid_identity(self) -> Option<f32> {
        match self {
            Self::Add => Some(0.0),
            Self::Mul => Some(1.0),
            Self::Min => Some(f32::INFINITY),
            Self::Max => Some(f32::NEG_INFINITY),
            // Mask monoids — the existential and universal quantifiers over a
            // bounded domain. Masks are bitwise in both evaluation tiers, so
            // the identities are the all-clear and all-set bit patterns
            // (`BitAnd`'s is a NaN by float reading; it is never read as a
            // number, only ANDed).
            Self::BitOr => Some(0.0),
            Self::BitAnd => Some(f32::from_bits(u32::MAX)),
            _ => None,
        }
    }

    /// Whether folding this op on `args` would bake in a **platform-specific**
    /// answer.
    ///
    /// The FP contract (CLAUDE.md, "Floating point at the edges") is hardware
    /// semantics — but the hardwares disagree in specific places, so for these
    /// (op, operand) combinations the language promises nothing:
    ///
    /// - `Min`/`Max` with a NaN operand: x86 `minps`/`maxps` compute
    ///   `(a OP b) ? a : b`, yielding the SECOND operand; aarch64 `FMIN`/`FMAX`
    ///   propagate the NaN. `min(NaN, 1.0)` is `1.0` on x86, `NaN` on aarch64.
    /// - `Min`/`Max` on **opposite-signed zeros**: the same operand-order rule
    ///   returns the second zero on x86 (since `-0.0 < 0.0` is false), while
    ///   `FMIN` selects `-0.0` and `FMAX` selects `+0.0`. `==` cannot see this
    ///   — `-0.0 == 0.0` — but `1.0 / x` turns it into `+inf` vs `-inf`.
    /// - `Gt`/`Ge` with a NaN operand: x86 uses the unordered predicates
    ///   (imm8 6/5), TRUE for NaN; aarch64 `FCMGT`/`FCMGE` are ordered, FALSE.
    /// - `Round` at an exact tie: nearest-even on x86 (`vroundps` imm 0x00),
    ///   ties-away on aarch64 (`FRINTA`), and `(x + 0.5).floor()` in the
    ///   combinator tier — three answers at `±n.5`.
    /// - `Round` on `-0.5 <= x <= -0.0`: both JIT tiers round to `-0.0`,
    ///   preserving the sign, while the combinator formula yields `+0.0`.
    ///   Not a tie, so it needs its own condition. `(x + 0.5).floor()` is not
    ///   any IEEE rounding mode — `round(-1.5)` is -1 there against -2
    ///   everywhere else — so the real repair is in that tier, not here.
    /// - `Recip`/`Rsqrt` **always**: these are estimate instructions and the
    ///   estimates differ — `rcpps` ~12 bits, `vrcp14ps` ~14, aarch64
    ///   `FRECPE` + one `FRECPS` refinement. The exact `1.0 / x` the folder
    ///   computes is not any of those answers.
    /// - `MulAdd` where one rounding differs from two: `vfmadd`/`FMLA` round
    ///   the product and sum together, SSE2's `mulps`+`addps` rounds twice.
    ///   `mul_add(1.0000001, 4097.0, 4097.0)` is `8194.001` fused, `8194.0`
    ///   split.
    ///
    /// x86 has no ties-away rounding mode and NaN/zero blending costs extra
    /// instructions, so unifying any of these would spend hot-path work on
    /// cases the language does not promise.
    ///
    /// **Why a fold in particular must decline:** constant folding happens on
    /// the BUILD host at macro-expansion time, but the constant it produces
    /// executes on the TARGET. Folding one of these bakes the build machine's
    /// arithmetic into a binary that may run somewhere computing differently —
    /// so this is not merely an optimized-vs-unoptimized discrepancy.
    ///
    /// `eval_unary`/`eval_binary` still return a value for these; it is this
    /// build's behavior and no promise about any other.
    #[must_use]
    pub fn fold_is_platform_specific(self, args: &[f32]) -> bool {
        match self {
            Self::Min | Self::Max => match args {
                // Equal but distinguishable is exactly ±0.0: x86 picks by
                // operand order, ARM by sign.
                [a, b] => a.is_nan() || b.is_nan() || (a == b && a.to_bits() != b.to_bits()),
                _ => false,
            },
            Self::Gt | Self::Ge => args.iter().any(|a| a.is_nan()),
            // Ties (x86 nearest-even vs aarch64 ties-away), plus the interval
            // where the combinator formula loses the sign of zero: for
            // `-0.5 <= x <= -0.0` both JIT tiers round to `-0.0` (sign
            // preserved) while `(x + 0.5).floor()` gives `+0.0`. `1.0 / x`
            // makes that `-inf` versus `+inf`. `-0.5` is already covered as a
            // tie; the rest of the interval is not.
            Self::Round => args.iter().any(|a| {
                (a - libm::truncf(*a)).abs() == 0.5 || (a.is_sign_negative() && *a > -0.5)
            }),
            // Reciprocal ESTIMATES, and every target estimates differently:
            // SSE2/AVX2 `rcpps`/`vrcpps` give ~12 bits, AVX-512 `vrcp14ps`
            // ~14, and aarch64 emits `FRECPE` plus one `FRECPS` Newton step.
            // `eval_unary`'s exact `1.0 / x` matches none of them, so there is
            // no host whose answer is worth baking. Unconditional: an estimate
            // is only guaranteed close, never equal, so no argument is safe.
            Self::Recip | Self::Rsqrt => true,
            // An invalid float->int conversion diverges in exactly two
            // places, and only two. x86 `cvttps2dq` yields the "integer
            // indefinite" 0x8000_0000 for every invalid input; aarch64
            // `FCVTZS` saturates to the destination's min/max with NaN going
            // to 0; and Rust's `as i32`, which `eval_unary` uses, saturates
            // like aarch64. So:
            //
            //   NaN        x86 -> i32::MIN,  aarch64 -> 0          DIVERGES
            //   x >= 2^31  x86 -> i32::MIN,  aarch64 -> i32::MAX   DIVERGES
            //   x < -2^31  x86 -> i32::MIN,  aarch64 -> i32::MIN   agrees
            //   -inf       x86 -> i32::MIN,  aarch64 -> i32::MIN   agrees
            //
            // The negative overflow direction agrees by arithmetic accident:
            // the integer-indefinite pattern IS i32::MIN, which is also what
            // saturation produces there. Refusing it too would cost ~19% of
            // the f32 space for nothing, so the predicate names the two real
            // cases rather than the convenient superset `!is_finite()`.
            //
            // The limit is 2^31 rather than `i32::MAX as f32`, because that
            // cast rounds UP to 2^31 and would admit values above the range.
            Self::TruncToInt => match args {
                [x] => {
                    const I32_LIMIT: f32 = 2_147_483_648.0; // 2^31
                    x.is_nan() || *x >= I32_LIMIT
                }
                _ => false,
            },

            // A shift count outside `0..32` has three answers. The folder
            // reduces it mod 32 (`y as u32 & 31`, below); x86
            // (V)PSLLD/(V)PSRLD zero the whole destination for any count > 31
            // (Intel SDM); and aarch64's immediate encoding silently changes
            // ELEMENT SIZE — `emit_shl` computes `immh:immb = shift + 32`, so
            // at shift >= 32 `immh` becomes `1xxx`, which decodes as `.2D`
            // and shifts bits across the 32-bit lane boundary. A non-integral
            // count is refused for the same reason: the tiers disagree on how
            // to narrow it. The emitters now assert this range, so a refused
            // fold cannot be laundered into a bad encoding either.
            Self::Shl | Self::Shr => match args {
                // The range test comes first and `||` short-circuits, so the
                // cast below only runs on a value already known to be in
                // `0..32` — where `as u32` is exact and total. (`f32::fract`
                // would be the obvious spelling but is std-only; this crate
                // is `no_std`.)
                [_, count] => !(0.0..32.0).contains(count) || (*count as u32) as f32 != *count,
                _ => false,
            },
            // One rounding or two. `mul_add` is the single-rounding FMA every
            // first-class target has (`vfmadd`, `FMLA`); `x * y + z` is what
            // SSE2 emits without one. Value-aware, because they agree for most
            // inputs and folding is worth keeping where they do.
            Self::MulAdd => match args {
                [x, y, z] => libm::fmaf(*x, *y, *z).to_bits() != (x * y + z).to_bits(),
                _ => false,
            },
            _ => false,
        }
    }

    /// Whether `self` is a valid reduction combiner (an associative monoid op
    /// with an identity).
    #[must_use]
    pub(crate) const fn is_monoid(self) -> bool {
        self.monoid_identity().is_some()
    }

    /// Convert to array index. Dense over `0..COUNT`, so `[T; OpKind::COUNT]`
    /// indexed by this is total — see [`OpKind::from_index`].
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Every op, in [`OpKind::index`] order.
    ///
    /// Prefer this to walking `0..COUNT` and calling [`OpKind::from_index`] —
    /// that spelling is why the three ops sitting past the old discriminant
    /// gaps were never visited by any table-filling loop.
    pub fn all() -> impl Iterator<Item = Self> + Clone {
        (0..Self::COUNT).map(|i| Self::from_index(i).expect("index_is_dense_and_total"))
    }

    /// Convert index to `OpKind`. Inverse of [`OpKind::index`].
    #[must_use]
    pub fn from_index(idx: usize) -> Option<Self> {
        if idx >= Self::COUNT {
            return None;
        }
        // SAFETY: `OpKind` is `repr(u8)` with discriminants assigned densely
        // over `0..COUNT`, so every value passing the bound above names a
        // variant. `index_is_dense_and_total` is what keeps that true — it
        // fails the moment a discriminant is skipped or `COUNT` drifts.
        Some(unsafe { core::mem::transmute::<u8, Self>(idx as u8) })
    }

    /// Get the arity of the operation.
    #[must_use]
    pub const fn arity(self) -> usize {
        match self {
            Self::Var | Self::Const | Self::Tuple | Self::Buffer => 0,

            Self::Neg
            | Self::Sqrt
            | Self::Rsqrt
            | Self::Abs
            | Self::Recip
            | Self::Floor
            | Self::Ceil
            | Self::Round
            | Self::Sin
            | Self::Cos
            | Self::Tan
            | Self::Asin
            | Self::Acos
            | Self::Atan
            | Self::Exp
            | Self::Exp2
            | Self::Ln
            | Self::Log2
            | Self::Log10
            | Self::TruncToInt
            | Self::IntToFloat => 1,

            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::Div
            | Self::Min
            | Self::Max
            | Self::Atan2
            | Self::Pow
            | Self::Lt
            | Self::Le
            | Self::Gt
            | Self::Ge
            | Self::Eq
            | Self::Ne
            | Self::IAdd
            | Self::Shl
            | Self::Shr
            | Self::BitAnd
            | Self::BitOr
            | Self::Dwrt
            | Self::RawGather => 2,

            Self::MulAdd | Self::Select | Self::Gather => 3,

            // N-ary: [combiner, reduce_var, extent, body].
            Self::Reduce => 4,
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
            Self::Recip => "recip",
            Self::Floor => "floor",
            Self::Ceil => "ceil",
            Self::Round => "round",
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
            Self::Lt => "lt",
            Self::Le => "le",
            Self::Gt => "gt",
            Self::Ge => "ge",
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Select => "select",
            Self::Tuple => "tuple",
            Self::TruncToInt => "trunc_to_int",
            Self::IntToFloat => "int_to_float",
            Self::IAdd => "iadd",
            Self::Shl => "shl",
            Self::Shr => "shr",
            Self::BitAnd => "bitand",
            Self::BitOr => "bitor",
            Self::Dwrt => "dwrt",
            Self::Buffer => "buffer",
            Self::Gather => "gather",
            Self::RawGather => "raw_gather",
            Self::Reduce => "reduce",
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
            "recip" => Some(Self::Recip),
            "floor" => Some(Self::Floor),
            "ceil" => Some(Self::Ceil),
            "round" => Some(Self::Round),
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
            "pow" | "powf" => Some(Self::Pow),
            "lt" => Some(Self::Lt),
            "le" => Some(Self::Le),
            "gt" => Some(Self::Gt),
            "ge" => Some(Self::Ge),
            "eq" => Some(Self::Eq),
            "ne" => Some(Self::Ne),
            "select" => Some(Self::Select),
            "tuple" => Some(Self::Tuple),
            "trunc_to_int" => Some(Self::TruncToInt),
            "int_to_float" => Some(Self::IntToFloat),
            "iadd" => Some(Self::IAdd),
            "shl" => Some(Self::Shl),
            "shr" => Some(Self::Shr),
            "bitand" => Some(Self::BitAnd),
            "bitor" => Some(Self::BitOr),
            "dwrt" => Some(Self::Dwrt),
            "buffer" => Some(Self::Buffer),
            "gather" => Some(Self::Gather),
            "raw_gather" => Some(Self::RawGather),
            "reduce" => Some(Self::Reduce),
            _ => None,
        }
    }

    /// Get the default cost estimate for this operation (in cycles).
    #[must_use]
    pub const fn default_cost(self) -> usize {
        match self {
            Self::Var | Self::Const | Self::Tuple | Self::Buffer => 0,
            // Memory read: native gather on AVX2/AVX-512, scalar loads on
            // NEON/SSE2. Priced between an arithmetic op and a transcendental.
            Self::Gather | Self::RawGather => 10,
            // Reduction is lowered (unrolled) away before costing; price the
            // node itself at zero so a stray one never dominates extraction.
            Self::Reduce => 0,
            Self::Neg | Self::Abs | Self::Floor | Self::Ceil | Self::Round => 1,
            Self::Add
            | Self::Sub
            | Self::Min
            | Self::Max
            | Self::Lt
            | Self::Le
            | Self::Gt
            | Self::Ge
            | Self::Eq
            | Self::Ne
            | Self::Select => 4,
            Self::Mul | Self::MulAdd | Self::Recip | Self::Rsqrt => 5,
            // Bit-manip primitives: single cheap integer/convert instructions.
            Self::TruncToInt
            | Self::IntToFloat
            | Self::IAdd
            | Self::Shl
            | Self::Shr
            | Self::BitAnd
            | Self::BitOr => 1,
            // Dwrt must be rewritten away before extraction; price it so the
            // extractor never prefers a surviving derivative over a decomposed
            // form. (A surviving Dwrt is then caught by a validation pass.)
            Self::Dwrt => 1_000_000,
            Self::Div
            | Self::Sqrt
            | Self::Sin
            | Self::Cos
            | Self::Tan
            | Self::Asin
            | Self::Acos
            | Self::Atan
            | Self::Atan2
            | Self::Exp
            | Self::Exp2
            | Self::Ln
            | Self::Log2
            | Self::Log10
            | Self::Pow => 15,
        }
    }

    /// Returns true if the operation is commutative (a op b == b op a).
    #[must_use]
    pub const fn is_commutative(self) -> bool {
        matches!(
            self,
            Self::Add | Self::Mul | Self::Min | Self::Max | Self::Eq | Self::Ne
        )
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

    /// Returns true if this op should appear in randomly generated seed expressions
    /// fed to the JIT training pipeline.
    ///
    /// Excludes:
    /// - Var/Const (leaves, not ops)
    /// - Tuple (structural, not computational)
    /// - MulAdd (fused — should only arise from rewrite rules)
    /// - Lt/Le/Gt/Ge/Eq/Ne (return masks, not floats — type-invalid in arithmetic)
    /// - Select (needs mask input — only valid composed with a comparison)
    /// - Buffer/Gather (memory ops — require a bound buffer, not synthesizable)
    #[must_use]
    pub const fn is_seed_op(self) -> bool {
        !matches!(
            self,
            Self::Var
                | Self::Const
                | Self::Tuple
                | Self::MulAdd
                | Self::Lt
                | Self::Le
                | Self::Gt
                | Self::Ge
                | Self::Eq
                | Self::Ne
                | Self::Select
                | Self::Buffer
                | Self::Gather
                | Self::RawGather
                | Self::Reduce
        )
    }

    /// Get the emit style for code generation.
    #[must_use]
    pub const fn emit_style(self) -> EmitStyle {
        match self {
            // Special cases handled separately
            Self::Var | Self::Const | Self::Tuple => EmitStyle::Special,

            // Unary prefix: (-a)
            Self::Neg => EmitStyle::UnaryPrefix,

            // Unary method: (a).sqrt()
            Self::Sqrt
            | Self::Rsqrt
            | Self::Abs
            | Self::Recip
            | Self::Floor
            | Self::Ceil
            | Self::Round
            | Self::Sin
            | Self::Cos
            | Self::Tan
            | Self::Asin
            | Self::Acos
            | Self::Atan
            | Self::Exp
            | Self::Exp2
            | Self::Ln
            | Self::Log2
            | Self::Log10
            | Self::TruncToInt
            | Self::IntToFloat => EmitStyle::UnaryMethod,

            // Binary infix: (a + b)
            Self::Add => EmitStyle::BinaryInfix("+"),
            Self::Sub => EmitStyle::BinaryInfix("-"),
            Self::Mul => EmitStyle::BinaryInfix("*"),
            Self::Div => EmitStyle::BinaryInfix("/"),

            // Binary method: (a).min(b)
            Self::Min
            | Self::Max
            | Self::Atan2
            | Self::Pow
            | Self::Lt
            | Self::Le
            | Self::Gt
            | Self::Ge
            | Self::Eq
            | Self::Ne
            | Self::IAdd
            | Self::Shl
            | Self::Shr
            | Self::BitAnd
            | Self::BitOr => EmitStyle::BinaryMethod,

            // Differentiation: never emitted (rewritten away in the e-graph).
            Self::Dwrt => EmitStyle::Special,

            // Memory ops: emitted by the JIT binding path, not as method calls.
            Self::Buffer | Self::Gather | Self::RawGather => EmitStyle::Special,

            // Reduction: lowered to unrolled arithmetic before codegen.
            Self::Reduce => EmitStyle::Special,

            // Ternary method: (a).mul_add(b, c)
            Self::MulAdd | Self::Select => EmitStyle::TernaryMethod,
        }
    }

    // NOTE: KNOWN_METHODS was removed - now derived from ALL_OPS via known_method_names()

    /// Evaluate a unary operation on a constant argument.
    ///
    /// Returns `None` for non-unary operations or operations that can't be
    /// evaluated at compile time.
    #[must_use]
    pub fn eval_unary(self, x: f32) -> Option<f32> {
        match self {
            Self::Neg => Some(-x),
            Self::Sqrt => Some(libm::sqrtf(x)),
            Self::Rsqrt => Some(1.0 / libm::sqrtf(x)),
            Self::Abs => Some(x.abs()),
            Self::Recip => Some(1.0 / x),
            Self::Floor => Some(libm::floorf(x)),
            Self::Ceil => Some(libm::ceilf(x)),
            // x86's answer (`vroundps` imm 0x00 is nearest-even). aarch64
            // `FRINTA` is ties-away and the combinator tier is
            // `(x + 0.5).floor()`, so at a tie this is one of three behaviors
            // and no promise — see `fold_is_platform_specific`.
            Self::Round => Some(libm::rintf(x)),
            // Integer-domain primitives. A lane holds an f32 bit pattern; these
            // reinterpret it as i32, exactly as the hardware instructions do
            // (`cvttps2dq` / `cvtdq2ps`). They exist so exp/log can lower to
            // arithmetic, and the interpreter needs them for the same reason —
            // it evaluates that lowering.
            Self::TruncToInt => Some(f32::from_bits((x as i32) as u32)),
            Self::IntToFloat => Some((x.to_bits() as i32) as f32),

            // Transcendentals have no arm on purpose. `f32::sin` and friends
            // are std-only (absent from `core`) and would be a *second*
            // definition of a function the compiler already defines by
            // expansion. Callers lower first — `expand_transcendentals` — and
            // then only the primitives above are ever evaluated. Same
            // discipline as `Dwrt`, which is also lowered rather than
            // interpreted.
            _ => None,
        }
    }

    /// A comparison lane: all bits set for true, all clear for false.
    ///
    /// This is not a stylistic choice about how to spell a boolean — it is the
    /// only representation the consumers accept. `Select` is a *bitwise* blend
    /// on every backend (`andps`/`andnps`/`orps` on SSE2 and AVX2, one
    /// `vpternlogd 0xCA` on AVX-512, `BSL` on aarch64), and `BitAnd`/`BitOr`
    /// are literal bitwise ops, so a mask lane's job is to be a per-bit
    /// stencil. `1.0` is not one: `0xFFFFFFFF & 0x3f800000` is `0x3f800000`,
    /// and blending `7.0` against `9.0` through that pattern gives `4.5` — a
    /// value neither branch ever held.
    ///
    /// Every tier that *executes* a comparison already writes all-ones —
    /// `cmpps`, `FCMEQ`, and the combinator tier's SIMD compares alike. Folding
    /// in the same representation is what makes a folded comparison and an
    /// executed one interchangeable as operands.
    #[must_use]
    pub const fn mask(condition: bool) -> f32 {
        if condition {
            f32::from_bits(u32::MAX)
        } else {
            0.0
        }
    }

    /// Whether this op's result is a lane **bit pattern** rather than a number.
    ///
    /// Comparisons produce masks ([`Self::mask`] — all-ones, which reads as
    /// NaN); the integer-domain primitives reinterpret lanes as `i32` and are
    /// how exp/log lower to arithmetic; `Select`/`BitAnd`/`BitOr` pass patterns
    /// through. For all of them a non-finite float reading is the intended
    /// output rather than an overflow, which is what separates them from
    /// `1e38 * 1e38` — the case a folder is right to refuse.
    #[must_use]
    pub const fn is_bitwise_domain(self) -> bool {
        matches!(
            self,
            Self::Lt
                | Self::Le
                | Self::Gt
                | Self::Ge
                | Self::Eq
                | Self::Ne
                | Self::Select
                | Self::BitAnd
                | Self::BitOr
                | Self::TruncToInt
                | Self::IntToFloat
                | Self::IAdd
                | Self::Shl
                | Self::Shr
        )
    }

    /// Evaluate a binary operation on constant arguments.
    ///
    /// Returns `None` for non-binary operations.
    #[must_use]
    pub fn eval_binary(self, x: f32, y: f32) -> Option<f32> {
        match self {
            Self::Add => Some(x + y),
            Self::Sub => Some(x - y),
            Self::Mul => Some(x * y),
            Self::Div => Some(x / y),
            // `minps`/`maxps` semantics on x86: `(a OP b) ? a : b`, so a NaN in
            // EITHER operand yields the second. aarch64 `FMIN`/`FMAX` propagate
            // the NaN instead — see `fold_is_platform_specific`. This arm
            // is x86's answer and is not a promise for other targets.
            Self::Min => Some(if x < y { x } else { y }),
            Self::Max => Some(if x > y { x } else { y }),
            Self::Lt => Some(Self::mask(x < y)),
            Self::Le => Some(Self::mask(x <= y)),
            // Gt/Ge are x86's UNORDERED predicates (imm8 6 = NLE_US, 5 =
            // NLT_US), so unlike Lt/Le they are TRUE when either operand is
            // NaN. ARM's FCMGT/FCMGE are ordered and give FALSE — see
            // `fold_is_platform_specific`.
            // Spelled as the ordered comparison plus the unordered case rather
            // than as `!(x <= y)`: same answer, but it says which part is the
            // NaN behavior instead of hiding it inside a negation.
            Self::Gt => Some(Self::mask(x > y || x.is_nan() || y.is_nan())),
            Self::Ge => Some(Self::mask(x >= y || x.is_nan() || y.is_nan())),
            // Exact, as every emitter compares. The old `(x-y).abs() < EPSILON`
            // made the folder disagree with the JIT for ordinary finite values:
            // 0.5 and its next representable neighbour differ by less than
            // EPSILON, so folding said "equal" where the hardware says "not".
            // NaN is false here and in `cmpps`/`FCMEQ` alike.
            Self::Eq => Some(Self::mask(x == y)),
            // Exact, and true for NaN — matching `NEQ_UQ` and an inverted
            // `FCMEQ`. Same epsilon bug as `Eq` above.
            Self::Ne => Some(Self::mask(x != y)),
            // Integer-domain binaries on lane bit patterns. The shift count is
            // the RHS `Const`'s numeric value (the schedule reads it as
            // `*v as u32 as u8`), and both shifts are logical — `vpslld` /
            // `vpsrld`, zero-filling.
            Self::IAdd => Some(f32::from_bits(
                (x.to_bits() as i32).wrapping_add(y.to_bits() as i32) as u32,
            )),
            Self::Shl => Some(f32::from_bits(x.to_bits() << (y as u32 & 31))),
            Self::Shr => Some(f32::from_bits(x.to_bits() >> (y as u32 & 31))),
            // Bitwise on lane bit patterns, matching the SIMD backends. On
            // canonical masks (1.0/0.0 here, all-ones/zero in the JIT) this
            // is logical AND/OR in both representations.
            Self::BitAnd => Some(f32::from_bits(x.to_bits() & y.to_bits())),
            Self::BitOr => Some(f32::from_bits(x.to_bits() | y.to_bits())),
            _ => None,
        }
    }

    /// Evaluate a ternary operation on constant arguments.
    #[must_use]
    pub fn eval_ternary(self, x: f32, y: f32, z: f32) -> Option<f32> {
        match self {
            // Two roundings, which is what a target without FMA emits (SSE2:
            // `mulps` then `addps`). AVX2+FMA, AVX-512 and aarch64 all round
            // once (`vfmadd`, `FMLA`), so the answers differ by target and
            // `fold_is_platform_specific` declines exactly where they do.
            Self::MulAdd => Some(x * y + z),
            // The bitwise blend every backend emits — `andps`/`andnps`/`orps`
            // on SSE2 and AVX2, one `vpternlogd 0xCA` on AVX-512, `BSL` on
            // aarch64. Spelling it `if x != 0.0` would be right only for a
            // canonical [`Self::mask`] and silently wrong for anything else.
            Self::Select => Some(f32::from_bits(
                (x.to_bits() & y.to_bits()) | (!x.to_bits() & z.to_bits()),
            )),
            _ => None,
        }
    }
}

// ============================================================================
// OpMap
// ============================================================================

/// A total map from every [`OpKind`] to a `T`.
///
/// This exists because the obvious alternative — `[T; OpKind::COUNT]`
/// subscripted by `OpKind::index()` — spread the same defect to every consumer
/// that reached for it. Four of them did: the latency-prior cost table, the
/// learned cost model, the NNUE op embeddings, and its gradient buffer.
///
/// Two things go wrong with the bare array, and both went wrong here:
///
/// 1. **The subscript escapes.** `index() -> usize` hands out an integer with
///    no memory of the bound it is valid against, so each consumer re-derives
///    the same unchecked cast. When the discriminants had gaps, every one of
///    them was out of bounds for the last three ops — and the workaround that
///    appeared was a local `if op.index() < OpKind::COUNT` guard that
///    *silently skipped* those ops rather than failing.
/// 2. **Positional literals drift.** A `[T; COUNT]` written out in order is
///    aligned to the discriminants only by convention, and nothing checks the
///    convention. The latency prior was written densely while `index()` was
///    sparse, so 13 of 50 ops read a neighbour's cost — `Dwrt` was priced at
///    10 instead of the 1000 that keeps extraction away from it.
///
/// [`OpMap::from_fn`] closes the second hole: an exhaustive `match` on
/// `OpKind` is checked by the compiler, so a table cannot silently misalign
/// and cannot silently omit an op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpMap<T> {
    slots: [T; OpKind::COUNT],
}

impl<T> OpMap<T> {
    /// Build a table by answering for every op. Write the body as a `match`
    /// and the compiler will not let you forget one.
    pub fn from_fn(mut f: impl FnMut(OpKind) -> T) -> Self {
        Self {
            slots: core::array::from_fn(|i| {
                f(OpKind::from_index(i).expect("index_is_dense_and_total"))
            }),
        }
    }

    /// Iterate in [`OpKind::index`] order — the order [`OpMap::as_slice`]
    /// serializes in.
    pub fn iter(&self) -> impl Iterator<Item = (OpKind, &T)> {
        self.slots
            .iter()
            .enumerate()
            .map(|(i, v)| (OpKind::from_index(i).expect("index_is_dense_and_total"), v))
    }

    /// The backing slots in `index()` order. For serialization; prefer
    /// indexing by [`OpKind`] for lookups.
    pub fn as_slice(&self) -> &[T] {
        &self.slots
    }

    /// Mutable backing slots in `index()` order. For deserialization.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.slots
    }
}

impl<T: Copy> OpMap<T> {
    /// A table where every op answers `value`.
    pub const fn splat(value: T) -> Self {
        Self {
            slots: [value; OpKind::COUNT],
        }
    }
}

impl<T: Default + Copy> Default for OpMap<T> {
    fn default() -> Self {
        Self::splat(T::default())
    }
}

impl<T> Index<OpKind> for OpMap<T> {
    type Output = T;

    #[inline]
    fn index(&self, op: OpKind) -> &T {
        // The only place in the workspace where an op becomes a subscript.
        &self.slots[op.index()]
    }
}

impl<T> IndexMut<OpKind> for OpMap<T> {
    #[inline]
    fn index_mut(&mut self, op: OpKind) -> &mut T {
        &mut self.slots[op.index()]
    }
}

// EmitStyle is imported from crate::traits - single source of truth

/// Every op name the surface language accepts as a method.
///
/// `Special` ops are excluded: they are structural (`Var`, `Const`, `Buffer`,
/// `Reduce`, ...) and have no method spelling.
///
/// This used to walk a parallel array of one zero-sized type per op, each
/// implementing an `OpMeta`/`Op`/arity-marker trait family, purely to answer
/// this question. `OpKind` already knows every op's name and emit style, so
/// the family and its 230-line module are gone.
pub fn known_method_names() -> impl Iterator<Item = &'static str> {
    OpKind::all()
        .filter(|op| !matches!(op.emit_style(), EmitStyle::Special))
        .map(OpKind::name)
}

#[cfg(test)]
mod index_space {
    use super::OpKind;

    /// `index()` must be dense over `0..COUNT`, and `from_index` must be its
    /// exact inverse.
    ///
    /// Both halves are load-bearing, and both were false before 2026-08-02.
    /// Every `[T; OpKind::COUNT]` in the workspace is subscripted by `index()`
    /// — the NNUE op embeddings, the latency-prior cost table, the extraction
    /// feature sets — so a *gap* in the discriminants pushes the ops after it
    /// past the end of every one of those arrays. `Gather`/`RawGather`/`Reduce`
    /// sat past three gaps and indexed 50..=52 into `[_; 50]`; the only reason
    /// that was not a live panic is that the e-graph refuses `Buffer` leaves
    /// earlier, and every `Gather` has one.
    ///
    /// The second half is worse: `from_index` bounds-checks against `COUNT`
    /// and then transmutes, so any index that is inside `COUNT` but names no
    /// variant is undefined behaviour rather than a `None`.
    #[test]
    fn index_is_dense_and_total() {
        for i in 0..OpKind::COUNT {
            let op = OpKind::from_index(i)
                .unwrap_or_else(|| panic!("from_index({i}) is None inside 0..COUNT — gap"));
            assert_eq!(op.index(), i, "{op:?} does not round-trip at index {i}");
        }
        assert_eq!(
            OpKind::from_index(OpKind::COUNT),
            None,
            "COUNT must be one past the last discriminant"
        );
    }
}

#[cfg(test)]
mod op_map {
    use super::{OpKind, OpMap};

    #[test]
    fn iter_yields_every_op_paired_with_its_own_slot_in_index_order() {
        let map = OpMap::from_fn(OpKind::index);
        let collected: Vec<(OpKind, usize)> = map.iter().map(|(op, v)| (op, *v)).collect();

        assert_eq!(collected.len(), OpKind::COUNT);
        for (i, (op, value)) in collected.iter().enumerate() {
            assert_eq!(op.index(), i, "iter() is not in index() order at slot {i}");
            assert_eq!(*value, i, "slot {i} does not hold its own op's value");
        }
    }

    #[test]
    fn as_slice_exposes_the_same_values_iter_yields_in_the_same_order() {
        let map = OpMap::from_fn(|op| op.index());

        let via_slice: Vec<usize> = map.as_slice().to_vec();
        let via_iter: Vec<usize> = map.iter().map(|(_, v)| *v).collect();

        assert_eq!(via_slice, via_iter);
    }

    #[test]
    fn as_mut_slice_writes_are_visible_through_indexing_by_op() {
        let mut map: OpMap<usize> = OpMap::splat(0);
        map.as_mut_slice()[OpKind::Add.index()] = 99;

        assert_eq!(map[OpKind::Add], 99, "write through as_mut_slice was lost");
        assert_eq!(
            map[OpKind::Sub],
            0,
            "as_mut_slice write bled into a neighboring op's slot"
        );
    }

    #[test]
    fn default_matches_splat_of_the_value_type_default() {
        let defaulted: OpMap<usize> = OpMap::default();
        let splatted: OpMap<usize> = OpMap::splat(usize::default());

        assert_eq!(defaulted, splatted);
    }
}

#[cfg(test)]
mod method_names {
    use super::{EmitStyle, OpKind, known_method_names};

    #[test]
    fn every_returned_name_round_trips_through_from_name() {
        for name in known_method_names() {
            assert_eq!(
                OpKind::from_name(name).map(OpKind::name),
                Some(name),
                "known_method_names() produced {name:?}, which from_name() cannot parse back"
            );
        }
    }

    #[test]
    fn excludes_every_op_whose_emit_style_is_special() {
        let names: std::collections::HashSet<&str> = known_method_names().collect();

        for op in OpKind::all() {
            if matches!(op.emit_style(), EmitStyle::Special) {
                assert!(
                    !names.contains(op.name()),
                    "{op:?} has EmitStyle::Special and must not have a method spelling"
                );
            }
        }
    }

    #[test]
    fn includes_an_ordinary_binary_op_by_its_method_name() {
        let names: Vec<&str> = known_method_names().collect();
        assert!(
            names.contains(&"min"),
            "Min is not EmitStyle::Special and should surface as a known method"
        );
    }
}

#[cfg(test)]
mod eval_unary_libm_arms {
    use super::OpKind;

    #[test]
    fn ceil_rounds_toward_positive_infinity() {
        assert_eq!(OpKind::Ceil.eval_unary(1.2), Some(2.0));
        assert_eq!(OpKind::Ceil.eval_unary(-1.2), Some(-1.0));
    }

    #[test]
    fn round_ties_to_even_matching_x86s_vroundps() {
        assert_eq!(OpKind::Round.eval_unary(2.5), Some(2.0));
        assert_eq!(OpKind::Round.eval_unary(3.5), Some(4.0));
        assert_eq!(OpKind::Round.eval_unary(1.2), Some(1.0));
    }

    #[test]
    fn int_to_float_reinterprets_bits_rather_than_converting_the_value() {
        // 1.0f32's bit pattern read as i32 is nowhere near 1.0.
        let one_bits = 1.0f32.to_bits() as i32;
        assert_eq!(
            OpKind::IntToFloat.eval_unary(1.0),
            Some(one_bits as f32),
            "IntToFloat must reinterpret x's bits as i32, not convert x's value"
        );
    }

    #[test]
    fn recip_is_exact_reciprocal_not_an_estimate() {
        assert_eq!(OpKind::Recip.eval_unary(4.0), Some(0.25));
    }
}

#[cfg(test)]
mod min_max_platform_specific {
    use super::OpKind;

    /// `pixelflow-codegen`'s `min_max_nan_handling_agrees_between_tiers`
    /// exercises the same inputs but *gates* its own assertion on this
    /// function's result (`if op.fold_is_platform_specific(..) { continue }`)
    /// — that structurally can't prove what the function should return, only
    /// that whatever it returns is self-consistent. These are the direct
    /// assertions on the classification itself.
    #[test]
    fn true_when_either_operand_is_nan() {
        assert!(OpKind::Min.fold_is_platform_specific(&[f32::NAN, 1.0]));
        assert!(OpKind::Max.fold_is_platform_specific(&[1.0, f32::NAN]));
    }

    #[test]
    fn true_for_equal_but_oppositely_signed_zeros() {
        // Equal but distinguishable is exactly ±0.0: x86 picks by operand
        // order, aarch64 by sign — `==` can't see it, only the bit pattern.
        assert!(OpKind::Min.fold_is_platform_specific(&[-0.0, 0.0]));
        assert!(OpKind::Max.fold_is_platform_specific(&[0.0, -0.0]));
    }

    #[test]
    fn false_for_equal_operands_with_identical_bits() {
        assert!(!OpKind::Min.fold_is_platform_specific(&[1.0, 1.0]));
        assert!(!OpKind::Max.fold_is_platform_specific(&[1.0, 1.0]));
    }

    #[test]
    fn false_for_ordinary_distinct_finite_operands() {
        assert!(!OpKind::Min.fold_is_platform_specific(&[2.0, 3.0]));
        assert!(!OpKind::Max.fold_is_platform_specific(&[2.0, 3.0]));
    }
}

#[cfg(test)]
mod round_platform_specific {
    use super::OpKind;

    #[test]
    fn round_platform_specific_true_at_an_exact_tie() {
        assert!(OpKind::Round.fold_is_platform_specific(&[2.5]));
    }

    #[test]
    fn round_platform_specific_true_in_the_negative_zero_sign_interval() {
        // `-0.5 < x <= -0.0`: both JIT tiers round to `-0.0` (sign
        // preserved) while the combinator tier's `(x + 0.5).floor()` gives
        // `+0.0`. `-0.5` itself is already covered by the tie clause above.
        assert!(OpKind::Round.fold_is_platform_specific(&[-0.3]));
    }

    #[test]
    fn round_platform_specific_false_for_a_negative_value_outside_the_interval() {
        assert!(!OpKind::Round.fold_is_platform_specific(&[-10.0]));
    }

    #[test]
    fn round_platform_specific_false_for_an_ordinary_non_tie_value() {
        assert!(!OpKind::Round.fold_is_platform_specific(&[1.7]));
    }
}

#[cfg(test)]
mod arity {
    use super::OpKind;

    #[test]
    fn arity_of_leaf_ops_is_zero() {
        for op in [OpKind::Var, OpKind::Const, OpKind::Tuple, OpKind::Buffer] {
            assert_eq!(op.arity(), 0, "{op:?} is a leaf and takes no operands");
        }
    }

    #[test]
    fn arity_of_unary_ops_is_one() {
        for op in [OpKind::Neg, OpKind::Sqrt, OpKind::TruncToInt, OpKind::Round] {
            assert_eq!(op.arity(), 1, "{op:?} is unary");
        }
    }

    #[test]
    fn arity_of_binary_ops_is_two() {
        for op in [OpKind::Add, OpKind::Shl, OpKind::Eq, OpKind::Dwrt] {
            assert_eq!(op.arity(), 2, "{op:?} is binary");
        }
    }

    #[test]
    fn arity_of_ternary_ops_is_three() {
        for op in [OpKind::MulAdd, OpKind::Select, OpKind::Gather] {
            assert_eq!(op.arity(), 3, "{op:?} is ternary");
        }
    }

    #[test]
    fn arity_of_reduce_is_four_for_its_combiner_var_extent_body_operands() {
        assert_eq!(OpKind::Reduce.arity(), 4);
    }
}

#[cfg(test)]
mod op_names {
    use super::OpKind;

    /// `known_method_names()` (and its round-trip test in `method_names`)
    /// deliberately excludes `EmitStyle::Special` ops (`Var`, `Const`,
    /// `Tuple`, `Dwrt`, `Buffer`, `Gather`, `RawGather`, `Reduce`) because
    /// they have no method spelling — but `from_name`/`name` must still
    /// round-trip for those, since the arena's textual dump/parse round-trip
    /// (and `Reduce`'s combiner-by-name encoding) depends on it.
    #[test]
    fn from_name_round_trips_every_op_including_special_ones() {
        for op in OpKind::all() {
            assert_eq!(
                OpKind::from_name(op.name()),
                Some(op),
                "{op:?}'s name() does not round-trip through from_name()"
            );
        }
    }

    #[test]
    fn from_name_accepts_the_powf_alias_for_pow() {
        assert_eq!(OpKind::from_name("powf"), Some(OpKind::Pow));
    }

    #[test]
    fn from_name_rejects_an_unknown_name() {
        assert_eq!(OpKind::from_name("this_is_not_an_op"), None);
    }
}

#[cfg(test)]
mod default_cost {
    use super::OpKind;

    #[test]
    fn default_cost_of_structural_leaves_is_zero() {
        for op in [
            OpKind::Var,
            OpKind::Const,
            OpKind::Tuple,
            OpKind::Buffer,
            OpKind::Reduce,
        ] {
            assert_eq!(op.default_cost(), 0, "{op:?} should cost nothing");
        }
    }

    #[test]
    fn default_cost_prices_dwrt_far_above_every_ordinary_op() {
        // `Dwrt` must be rewritten away before extraction; if it ever
        // survives, the extractor must never prefer it over a decomposed
        // form, which only holds if its price dwarfs everything else.
        assert!(OpKind::Dwrt.default_cost() > 1_000);
    }

    #[test]
    fn default_cost_of_ordinary_arithmetic_is_a_handful_of_cycles() {
        assert_eq!(OpKind::Add.default_cost(), 4);
        assert_eq!(OpKind::Mul.default_cost(), 5);
    }
}

#[cfg(test)]
mod algebraic_properties {
    use super::OpKind;

    #[test]
    fn is_commutative_is_true_for_add_mul_min_max_eq_ne() {
        for op in [
            OpKind::Add,
            OpKind::Mul,
            OpKind::Min,
            OpKind::Max,
            OpKind::Eq,
            OpKind::Ne,
        ] {
            assert!(op.is_commutative(), "{op:?} should be commutative");
        }
    }

    #[test]
    fn is_commutative_is_false_for_sub_div_and_shl() {
        for op in [OpKind::Sub, OpKind::Div, OpKind::Shl] {
            assert!(!op.is_commutative(), "{op:?} should not be commutative");
        }
    }

    #[test]
    fn is_associative_is_true_for_add_mul_min_max() {
        for op in [OpKind::Add, OpKind::Mul, OpKind::Min, OpKind::Max] {
            assert!(op.is_associative(), "{op:?} should be associative");
        }
    }

    #[test]
    fn is_associative_is_false_for_sub_div_and_a_non_arithmetic_op() {
        for op in [OpKind::Sub, OpKind::Div, OpKind::Eq] {
            assert!(!op.is_associative(), "{op:?} should not be associative");
        }
    }

    #[test]
    fn identity_of_additive_ops_is_zero() {
        assert_eq!(OpKind::Add.identity(), Some(0.0));
        assert_eq!(OpKind::Sub.identity(), Some(0.0));
    }

    #[test]
    fn identity_of_multiplicative_ops_is_one() {
        assert_eq!(OpKind::Mul.identity(), Some(1.0));
        assert_eq!(OpKind::Div.identity(), Some(1.0));
    }

    #[test]
    fn identity_is_none_for_an_op_with_no_identity_element() {
        assert_eq!(OpKind::Min.identity(), None);
        assert_eq!(OpKind::Eq.identity(), None);
    }

    #[test]
    fn annihilator_of_mul_is_zero() {
        assert_eq!(OpKind::Mul.annihilator(), Some(0.0));
    }

    #[test]
    fn annihilator_is_none_for_an_op_with_no_annihilating_element() {
        assert_eq!(OpKind::Add.annihilator(), None);
        assert_eq!(OpKind::Min.annihilator(), None);
    }

    #[test]
    fn is_idempotent_is_true_for_min_max_and_abs() {
        for op in [OpKind::Min, OpKind::Max, OpKind::Abs] {
            assert!(op.is_idempotent(), "{op:?} should be idempotent");
        }
    }

    #[test]
    fn is_idempotent_is_false_for_add() {
        assert!(!OpKind::Add.is_idempotent());
    }

    #[test]
    fn is_monoid_is_true_for_ops_with_a_monoid_identity() {
        assert!(OpKind::Add.is_monoid());
        assert!(OpKind::Mul.is_monoid());
    }

    #[test]
    fn is_monoid_is_true_for_min_max_and_the_mask_monoids() {
        // Reachable only as a `Reduce` combiner, so nothing in this crate
        // otherwise exercises `Min`/`BitOr`/`BitAnd` through
        // `monoid_identity` — unlike `Add`/`Mul`, which the dwrt/passes
        // tests reduce over already.
        for op in [OpKind::Min, OpKind::Max, OpKind::BitOr, OpKind::BitAnd] {
            assert!(
                op.is_monoid(),
                "{op:?} should be a valid reduction combiner"
            );
        }
    }

    #[test]
    fn monoid_identity_of_min_max_are_the_extended_reals_bounds() {
        assert_eq!(OpKind::Min.monoid_identity(), Some(f32::INFINITY));
        assert_eq!(OpKind::Max.monoid_identity(), Some(f32::NEG_INFINITY));
    }

    #[test]
    fn monoid_identity_of_bitor_is_the_all_clear_pattern() {
        assert_eq!(OpKind::BitOr.monoid_identity(), Some(0.0));
    }

    #[test]
    fn monoid_identity_of_bitand_is_the_all_set_pattern() {
        // The all-ones pattern reads as NaN, so compare bits, not `f32`
        // equality (NaN != NaN).
        assert_eq!(
            OpKind::BitAnd.monoid_identity().map(f32::to_bits),
            Some(u32::MAX)
        );
    }

    #[test]
    fn is_monoid_is_false_for_an_op_with_no_monoid_identity() {
        assert!(!OpKind::Eq.is_monoid());
        assert!(!OpKind::Sub.is_monoid());
    }
}

#[cfg(test)]
mod domain_classification {
    use super::OpKind;

    #[test]
    fn is_seed_op_excludes_structural_and_type_invalid_ops() {
        for op in [
            OpKind::Var,
            OpKind::Const,
            OpKind::Tuple,
            OpKind::MulAdd,
            OpKind::Lt,
            OpKind::Select,
            OpKind::Buffer,
            OpKind::Gather,
            OpKind::RawGather,
            OpKind::Reduce,
        ] {
            assert!(
                !op.is_seed_op(),
                "{op:?} should be excluded from seed expressions"
            );
        }
    }

    #[test]
    fn is_seed_op_includes_ordinary_arithmetic() {
        for op in [OpKind::Add, OpKind::Mul, OpKind::Sin] {
            assert!(op.is_seed_op(), "{op:?} should be eligible as a seed op");
        }
    }

    #[test]
    fn is_bitwise_domain_is_true_for_comparisons_and_integer_domain_ops() {
        for op in [
            OpKind::Lt,
            OpKind::Eq,
            OpKind::Select,
            OpKind::BitAnd,
            OpKind::Shl,
            OpKind::TruncToInt,
        ] {
            assert!(op.is_bitwise_domain(), "{op:?} should be bitwise-domain");
        }
    }

    #[test]
    fn is_bitwise_domain_is_false_for_ordinary_arithmetic() {
        for op in [OpKind::Add, OpKind::Mul, OpKind::Sin] {
            assert!(
                !op.is_bitwise_domain(),
                "{op:?} should not be bitwise-domain"
            );
        }
    }
}

#[cfg(test)]
mod eval_binary_arms {
    use super::OpKind;

    #[test]
    fn arithmetic_matches_direct_computation() {
        assert_eq!(OpKind::Add.eval_binary(2.0, 3.0), Some(5.0));
        assert_eq!(OpKind::Sub.eval_binary(5.0, 3.0), Some(2.0));
        assert_eq!(OpKind::Mul.eval_binary(2.0, 3.0), Some(6.0));
        assert_eq!(OpKind::Div.eval_binary(6.0, 3.0), Some(2.0));
    }

    #[test]
    fn min_max_on_equal_operands_return_the_second_operands_bit_pattern() {
        // x86 `minps`/`maxps` compute `(a OP b) ? a : b`, so when neither
        // operand is strictly less/greater the SECOND operand wins — a fact
        // that plain numeric equality can't see (`-0.0 == 0.0`), only bits.
        let neg_zero = -0.0f32;
        let pos_zero = 0.0f32;
        assert_eq!(
            OpKind::Min
                .eval_binary(neg_zero, pos_zero)
                .unwrap()
                .to_bits(),
            pos_zero.to_bits(),
            "Min(-0.0, 0.0) must return the second operand's bit pattern"
        );
        assert_eq!(
            OpKind::Max
                .eval_binary(pos_zero, neg_zero)
                .unwrap()
                .to_bits(),
            neg_zero.to_bits(),
            "Max(0.0, -0.0) must return the second operand's bit pattern"
        );
    }

    /// `mask(true)` is an all-ones bit pattern, which reads as NaN — so
    /// `assert_eq!` on the raw `f32` is always false even when the masks
    /// match. Compare bit patterns instead, as every consumer of a mask does.
    fn assert_mask_eq(got: Option<f32>, want_true: bool, msg: &str) {
        assert_eq!(
            got.map(f32::to_bits),
            Some(OpKind::mask(want_true).to_bits()),
            "{msg}"
        );
    }

    #[test]
    fn lt_le_are_strict_and_non_strict_at_the_boundary() {
        // At x == y, Lt is false and Le is true — exactly what distinguishes
        // `<` from `<=`.
        assert_mask_eq(OpKind::Lt.eval_binary(3.0, 3.0), false, "Lt(3, 3)");
        assert_mask_eq(OpKind::Le.eval_binary(3.0, 3.0), true, "Le(3, 3)");
    }

    #[test]
    fn gt_ge_are_true_when_either_operand_is_nan() {
        assert_mask_eq(OpKind::Gt.eval_binary(f32::NAN, 1.0), true, "Gt(NaN, 1.0)");
        assert_mask_eq(OpKind::Ge.eval_binary(1.0, f32::NAN), true, "Ge(1.0, NaN)");
    }

    #[test]
    fn gt_ge_agree_with_ordered_comparison_on_finite_operands() {
        assert_mask_eq(OpKind::Gt.eval_binary(3.0, 3.0), false, "Gt(3, 3)");
        assert_mask_eq(OpKind::Ge.eval_binary(3.0, 3.0), true, "Ge(3, 3)");
        assert_mask_eq(OpKind::Gt.eval_binary(1.0, 2.0), false, "Gt(1, 2)");
    }

    #[test]
    fn eq_ne_are_exact_and_disagree_only_at_equality() {
        assert_mask_eq(OpKind::Eq.eval_binary(3.0, 3.0), true, "Eq(3, 3)");
        assert_mask_eq(OpKind::Ne.eval_binary(3.0, 3.0), false, "Ne(3, 3)");
        assert_mask_eq(OpKind::Eq.eval_binary(3.0, 4.0), false, "Eq(3, 4)");
        assert_mask_eq(OpKind::Ne.eval_binary(3.0, 4.0), true, "Ne(3, 4)");
    }

    #[test]
    fn iadd_wraps_the_bit_pattern_as_a_twos_complement_integer() {
        let max = f32::from_bits(i32::MAX as u32);
        let one = f32::from_bits(1u32);
        assert_eq!(
            OpKind::IAdd.eval_binary(max, one),
            Some(f32::from_bits(i32::MIN as u32)),
            "IAdd must wrap on overflow, not saturate"
        );
    }

    #[test]
    fn shl_shifts_the_bit_pattern_toward_the_high_end() {
        let x = f32::from_bits(0b1);
        assert_eq!(OpKind::Shl.eval_binary(x, 1.0), Some(f32::from_bits(0b10)));
    }

    #[test]
    fn shr_shifts_the_bit_pattern_toward_the_low_end() {
        let x = f32::from_bits(0b10);
        assert_eq!(OpKind::Shr.eval_binary(x, 1.0), Some(f32::from_bits(0b1)));
    }

    #[test]
    fn bitand_bitor_operate_on_the_bit_pattern() {
        let a = f32::from_bits(0b1100);
        let b = f32::from_bits(0b1010);
        assert_eq!(
            OpKind::BitAnd.eval_binary(a, b),
            Some(f32::from_bits(0b1000))
        );
        assert_eq!(
            OpKind::BitOr.eval_binary(a, b),
            Some(f32::from_bits(0b1110))
        );
    }

    #[test]
    fn returns_none_for_a_non_binary_op() {
        assert_eq!(OpKind::Neg.eval_binary(1.0, 2.0), None);
    }
}

#[cfg(test)]
mod eval_ternary_arms {
    use super::OpKind;

    #[test]
    fn muladd_computes_the_fused_product_plus_addend() {
        assert_eq!(OpKind::MulAdd.eval_ternary(2.0, 3.0, 1.0), Some(7.0));
    }

    #[test]
    fn select_blends_bit_patterns_by_mask() {
        let mask_true = OpKind::mask(true);
        let mask_false = OpKind::mask(false);
        let a = 5.0f32;
        let b = 9.0f32;
        assert_eq!(OpKind::Select.eval_ternary(mask_true, a, b), Some(a));
        assert_eq!(OpKind::Select.eval_ternary(mask_false, a, b), Some(b));
    }

    #[test]
    fn returns_none_for_a_non_ternary_op() {
        assert_eq!(OpKind::Add.eval_ternary(1.0, 2.0, 3.0), None);
    }
}
