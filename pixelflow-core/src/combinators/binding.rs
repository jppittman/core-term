//! # Let Bindings via Domain Extension
//!
//! Provides `Let` and `Var` combinators for local variable bindings in kernel
//! expressions. Uses Peano-encoded indices for type-safe de Bruijn indexing.
//!
//! ## Architecture
//!
//! Let bindings work by **extending the domain**:
//!
//! - `Let<Val, Body>` evaluates `Val`, extends domain with the result, evaluates `Body`
//! - `Var<N>` reads from the domain stack using Peano index `N`
//!
//! ## Domain Flow
//!
//! ```text
//! Base domain: (x, y)
//!
//! After Let(val_expr, body):
//!   1. Evaluate val_expr on (x, y) → v
//!   2. Create extended domain: LetExtended(v, (x, y))
//!   3. Evaluate body on extended domain
//!
//! Inside body:
//!   - Var<0> reads v (head of domain)
//!   - X, Y, Z, W read spatial coords (via Spatial trait)
//! ```
//!
//! ## Example
//!
//! ```ignore
//! // let dist = sqrt(x² + y²); dist - 1.0
//! Let::new(
//!     (X * X + Y * Y).sqrt(),  // val: compute distance once
//!     Var::<N0> - 1.0f32,      // body: use it (Var<0> reads head)
//! )
//! ```
//!
//! ## Why Peano Numbers?
//!
//! Const generic recursion like `Get<{ N - 1 }>` causes the trait solver to
//! hang because it performs arithmetic during resolution. Peano encoding
//! (`Succ<Succ<Zero>>`) is pure structural recursion - no arithmetic needed.

use crate::Manifold;
use crate::domain::{Head, LetExtended, Tail};
use core::marker::PhantomData;

// ============================================================================
// Peano Numbers (Type-Level Naturals)
// ============================================================================

/// Type-level zero.
#[derive(Clone, Copy, Debug)]
pub struct Zero;

/// Type-level successor (N + 1).
#[derive(Clone, Copy, Debug)]
pub struct Succ<N>(PhantomData<N>);

// Convenience aliases
/// Index 0
pub type N0 = Zero;
/// Index 1
pub type N1 = Succ<N0>;
/// Index 2
pub type N2 = Succ<N1>;
/// Index 3
pub type N3 = Succ<N2>;
/// Index 4
pub type N4 = Succ<N3>;
/// Index 5
pub type N5 = Succ<N4>;
/// Index 6
pub type N6 = Succ<N5>;
/// Index 7
pub type N7 = Succ<N6>;

// ============================================================================
// Let Combinator
// ============================================================================

/// Bind a value and evaluate the body with the extended domain.
///
/// `Let<Val, Body>` evaluates `Val` on the input domain, extends the domain
/// with the result using `LetExtended`, then evaluates `Body` on the extended domain.
///
/// ## Domain Extension
///
/// If the input domain is `P`, and `Val: Manifold<P, Output = V>`, then:
/// - `Body` is evaluated on domain `LetExtended<V, P>`
/// - Inside `Body`, `Var<0>` reads the bound value `V`
/// - `X`, `Y`, `Z`, `W` still work (via `Spatial` trait delegation)
#[derive(Clone, Debug)]
pub struct Let<Val, Body> {
    /// The value expression to bind.
    pub val: Val,
    /// The body expression (evaluated with extended context).
    pub body: Body,
}

impl<Val, Body> Let<Val, Body> {
    /// Create a new let binding.
    pub fn new(val: Val, body: Body) -> Self {
        Self { val, body }
    }
}

impl<P, Val, Body> Manifold<P> for Let<Val, Body>
where
    P: Copy + Send + Sync,
    Val: Manifold<P>,
    Val::Output: Copy + Send + Sync,
    Body: Manifold<LetExtended<Val::Output, P>>,
{
    type Output = Body::Output;

    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        // 1. Evaluate the value being bound
        let val = self.val.eval(p);

        // 2. Create extended domain with bound value
        let extended = LetExtended(val, p);

        // 3. Evaluate body on extended domain
        self.body.eval(extended)
    }
}

// ============================================================================
// Var Combinator
// ============================================================================

/// Read a bound variable from the domain stack.
///
/// `Var<N>` retrieves the value at Peano index `N` from the domain stack.
/// Index 0 is the most recently bound value (head of domain).
#[derive(Clone, Copy, Debug)]
pub struct Var<N>(PhantomData<N>);

impl<N> Var<N> {
    /// Create a new variable reference.
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<N> Default for Var<N> {
    fn default() -> Self {
        Self::new()
    }
}

// Var<Zero> reads the head of the domain
impl<P> Manifold<P> for Var<Zero>
where
    P: Head + Send + Sync,
    P::Value: Copy + Send + Sync,
{
    type Output = P::Value;

    #[inline(always)]
    fn eval(&self, p: P) -> P::Value {
        p.head()
    }
}

// Var<Succ<N>> recurses down the domain stack
impl<N, P> Manifold<P> for Var<Succ<N>>
where
    N: Send + Sync,
    P: Tail + Send + Sync,
    P::Rest: Copy,
    Var<N>: Manifold<P::Rest>,
{
    type Output = <Var<N> as Manifold<P::Rest>>::Output;

    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        Var::<N>::new().eval(p.tail())
    }
}

// ============================================================================
// Stub Manifold<Field4> impl for ManifoldExt compatibility
// ============================================================================
//
// Var<N> should only be evaluated on domains with Head trait (i.e., LetExtended).
// However, to use ManifoldExt methods for AST construction (.sqrt(), .abs(), etc.),
// we need Manifold<Field4> impls. These panic if actually called, but that should
// never happen since Var<N> is always wrapped in Let bindings that extend the domain.

type Field4 = (crate::Field, crate::Field, crate::Field, crate::Field);

impl Manifold<Field4> for Var<Zero> {
    type Output = crate::Field;

    fn eval(&self, _p: Field4) -> Self::Output {
        unreachable!("Var<N> must be evaluated within a Let binding that extends the domain")
    }
}

impl<N: Send + Sync> Manifold<Field4> for Var<Succ<N>> {
    type Output = crate::Field;

    fn eval(&self, _p: Field4) -> Self::Output {
        unreachable!("Var<N> must be evaluated within a Let binding that extends the domain")
    }
}

// ============================================================================
// Operator Overloading for Var
// ============================================================================

impl<N, R> core::ops::Add<R> for Var<N> {
    type Output = crate::ops::Add<Var<N>, R>;
    fn add(self, rhs: R) -> Self::Output {
        crate::ops::Add(self, rhs)
    }
}

impl<N, R> core::ops::Sub<R> for Var<N> {
    type Output = crate::ops::Sub<Var<N>, R>;
    fn sub(self, rhs: R) -> Self::Output {
        crate::ops::Sub(self, rhs)
    }
}

impl<N, R> core::ops::Mul<R> for Var<N> {
    type Output = crate::ops::Mul<Var<N>, R>;
    fn mul(self, rhs: R) -> Self::Output {
        crate::ops::Mul(self, rhs)
    }
}

impl<N, R> core::ops::Div<R> for Var<N> {
    type Output = crate::ops::Div<Var<N>, R>;
    fn div(self, rhs: R) -> Self::Output {
        crate::ops::Div(self, rhs)
    }
}

// ============================================================================
// Legacy Compatibility: Graph trait and Root
// ============================================================================

/// The empty context (stack bottom) for legacy Graph-based code.
#[derive(Clone, Copy, Debug)]
pub struct Empty;

/// Retrieve a value from the context stack at type-level index `N`.
/// Legacy trait for backward compatibility.
pub trait Get<N>: Send + Sync {
    /// Get the value at index N.
    fn get(&self) -> crate::Field;
}

// Base case: index Zero gets the head of the stack
impl<Tail: Send + Sync> Get<Zero> for (crate::Field, Tail) {
    #[inline(always)]
    fn get(&self) -> crate::Field {
        self.0
    }
}

// Recursive case: index Succ<N> skips head, gets N from tail
impl<N, Tail> Get<Succ<N>> for (crate::Field, Tail)
where
    Tail: Get<N>,
{
    #[inline(always)]
    fn get(&self) -> crate::Field {
        self.1.get()
    }
}

/// A computation graph node that evaluates with a context stack.
/// Legacy trait for backward compatibility with existing code.
pub trait Graph<Ctx>: Send + Sync {
    /// Evaluate at coordinates with the given context.
    fn eval_at(
        &self,
        ctx: &Ctx,
        x: crate::Field,
        y: crate::Field,
        z: crate::Field,
        w: crate::Field,
    ) -> crate::Field;
}

/// Wrapper to lift a `Manifold` into the `Graph` world.
#[derive(Clone, Debug)]
pub struct Lift<M>(pub M);

impl<M, Ctx> Graph<Ctx> for Lift<M>
where
    M: Manifold<Field4, Output = crate::Field>,
    Ctx: Send + Sync,
{
    #[inline(always)]
    fn eval_at(
        &self,
        _ctx: &Ctx,
        x: crate::Field,
        y: crate::Field,
        z: crate::Field,
        w: crate::Field,
    ) -> crate::Field {
        self.0.eval((x, y, z, w))
    }
}

/// Graph-level addition (legacy).
#[derive(Clone, Debug)]
pub struct GAdd<L, R>(pub L, pub R);

impl<Ctx, L, R> Graph<Ctx> for GAdd<L, R>
where
    L: Graph<Ctx>,
    R: Graph<Ctx>,
    Ctx: Send + Sync,
{
    #[inline(always)]
    fn eval_at(
        &self,
        ctx: &Ctx,
        x: crate::Field,
        y: crate::Field,
        z: crate::Field,
        w: crate::Field,
    ) -> crate::Field {
        use crate::numeric::Numeric;
        self.0
            .eval_at(ctx, x, y, z, w)
            .raw_add(self.1.eval_at(ctx, x, y, z, w))
    }
}

/// Graph-level subtraction (legacy).
#[derive(Clone, Debug)]
pub struct GSub<L, R>(pub L, pub R);

impl<Ctx, L, R> Graph<Ctx> for GSub<L, R>
where
    L: Graph<Ctx>,
    R: Graph<Ctx>,
    Ctx: Send + Sync,
{
    #[inline(always)]
    fn eval_at(
        &self,
        ctx: &Ctx,
        x: crate::Field,
        y: crate::Field,
        z: crate::Field,
        w: crate::Field,
    ) -> crate::Field {
        use crate::numeric::Numeric;
        self.0
            .eval_at(ctx, x, y, z, w)
            .raw_sub(self.1.eval_at(ctx, x, y, z, w))
    }
}

/// Graph-level multiplication (legacy).
#[derive(Clone, Debug)]
pub struct GMul<L, R>(pub L, pub R);

impl<Ctx, L, R> Graph<Ctx> for GMul<L, R>
where
    L: Graph<Ctx>,
    R: Graph<Ctx>,
    Ctx: Send + Sync,
{
    #[inline(always)]
    fn eval_at(
        &self,
        ctx: &Ctx,
        x: crate::Field,
        y: crate::Field,
        z: crate::Field,
        w: crate::Field,
    ) -> crate::Field {
        use crate::numeric::Numeric;
        self.0
            .eval_at(ctx, x, y, z, w)
            .raw_mul(self.1.eval_at(ctx, x, y, z, w))
    }
}

/// Graph-level division (legacy).
#[derive(Clone, Debug)]
pub struct GDiv<L, R>(pub L, pub R);

impl<Ctx, L, R> Graph<Ctx> for GDiv<L, R>
where
    L: Graph<Ctx>,
    R: Graph<Ctx>,
    Ctx: Send + Sync,
{
    #[inline(always)]
    fn eval_at(
        &self,
        ctx: &Ctx,
        x: crate::Field,
        y: crate::Field,
        z: crate::Field,
        w: crate::Field,
    ) -> crate::Field {
        use crate::numeric::Numeric;
        self.0
            .eval_at(ctx, x, y, z, w)
            .raw_div(self.1.eval_at(ctx, x, y, z, w))
    }
}

/// Root node that converts a `Graph` into a `Manifold` (legacy).
#[derive(Clone, Debug)]
pub struct Root<G>(pub G);

impl<G> Manifold<Field4> for Root<G>
where
    G: Graph<Empty> + Send + Sync,
{
    type Output = crate::Field;

    #[inline(always)]
    fn eval(&self, p: Field4) -> crate::Field {
        self.0.eval_at(&Empty, p.0, p.1, p.2, p.3)
    }
}

// Legacy Graph impl for Let (for backward compatibility)
impl<Ctx, Val, Body> Graph<Ctx> for Let<Val, Body>
where
    Ctx: Send + Sync + Copy,
    Val: Graph<Ctx>,
    Body: Graph<(crate::Field, Ctx)>,
{
    #[inline(always)]
    fn eval_at(
        &self,
        ctx: &Ctx,
        x: crate::Field,
        y: crate::Field,
        z: crate::Field,
        w: crate::Field,
    ) -> crate::Field {
        let val = self.val.eval_at(ctx, x, y, z, w);
        let new_ctx = (val, *ctx);
        self.body.eval_at(&new_ctx, x, y, z, w)
    }
}

// Legacy Graph impl for Var
impl<N, Ctx> Graph<Ctx> for Var<N>
where
    N: Send + Sync,
    Ctx: Get<N>,
{
    #[inline(always)]
    fn eval_at(
        &self,
        ctx: &Ctx,
        _x: crate::Field,
        _y: crate::Field,
        _z: crate::Field,
        _w: crate::Field,
    ) -> crate::Field {
        ctx.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::LetExtended;
    use crate::{Field, X, Y};

    #[test]
    fn test_let_binding_new_style() {
        // let v = 10.0; v + 5.0
        let expr = Let::new(10.0f32, Var::<N0>::new() + 5.0f32);

        // Evaluate on 2D domain
        let domain = (Field::from(0.0), Field::from(0.0));
        let result = expr.eval(domain);

        let mut buf = [0.0f32; crate::PARALLELISM];
        result.store(&mut buf);
        assert_eq!(buf[0], 15.0); // 10 + 5
    }

    #[test]
    fn test_let_with_spatial() {
        // let dist = x; dist * 2
        let expr = Let::new(X, Var::<N0>::new() * 2.0f32);

        let domain = (Field::from(5.0), Field::from(0.0));
        let result = expr.eval(domain);

        let mut buf = [0.0f32; crate::PARALLELISM];
        result.store(&mut buf);
        assert_eq!(buf[0], 10.0); // 5 * 2
    }

    #[test]
    fn test_nested_let_new_style() {
        // let a = 3.0; let b = 4.0; a + b
        let expr = Let::new(
            3.0f32, // a = 3.0 (becomes Var<1> after second let)
            Let::new(
                4.0f32,                              // b = 4.0 (becomes Var<0>)
                Var::<N1>::new() + Var::<N0>::new(), // a + b
            ),
        );

        let domain = (Field::from(0.0), Field::from(0.0));
        let result = expr.eval(domain);

        let mut buf = [0.0f32; crate::PARALLELISM];
        result.store(&mut buf);
        assert_eq!(buf[0], 7.0); // 3 + 4
    }

    #[test]
    fn test_legacy_peano_get() {
        let ctx: (Field, (Field, Empty)) = (Field::from(10.0), (Field::from(20.0), Empty));

        // Index 0 should get 10.0 (head)
        let v0: Field = <(Field, (Field, Empty)) as Get<N0>>::get(&ctx);
        let mut buf = [0.0f32; crate::PARALLELISM];
        v0.store(&mut buf);
        assert_eq!(buf[0], 10.0);

        // Index 1 should get 20.0 (tail head)
        let v1: Field = <(Field, (Field, Empty)) as Get<N1>>::get(&ctx);
        v1.store(&mut buf);
        assert_eq!(buf[0], 20.0);
    }

    #[test]
    fn test_legacy_let_binding() {
        // let x = 5.0; x + x
        let graph = Let::new(Lift(5.0f32), GAdd(Var::<N0>::new(), Var::<N0>::new()));

        let zero = Field::from(0.0);
        let result = graph.eval_at(&Empty, zero, zero, zero, zero);

        let mut buf = [0.0f32; crate::PARALLELISM];
        result.store(&mut buf);
        assert_eq!(buf[0], 10.0); // 5 + 5
    }
}
