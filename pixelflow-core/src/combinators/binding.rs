//! # Let Bindings via Type-Level Stack Machine
//!
//! Provides `Let` and `Var` combinators for local variable bindings in kernel
//! expressions. Uses Peano-encoded indices to avoid compiler hangs from
//! recursive const generic trait resolution.
//!
//! ## Architecture
//!
//! The context is a nested tuple stack: `(v2, (v1, (v0, ())))`
//!
//! - `Let<Val, Body>` evaluates `Val`, pushes result onto stack, evaluates `Body`
//! - `Var<N>` reads the value at index `N` from the stack
//!
//! ## Example
//!
//! ```ignore
//! // let x = X * X; let y = Y * Y; x + y
//! Let(
//!     Mul(X, X),                    // x = X * X (index 0)
//!     Let(
//!         Mul(Y, Y),                // y = Y * Y (index 1, but 0 in inner scope)
//!         Add(Var::<N1>, Var::<N0>) // Var<1> = x, Var<0> = y
//!     )
//! )
//! ```
//!
//! ## Why Peano Numbers?
//!
//! Const generic recursion like `Get<{ N - 1 }>` causes the trait solver to
//! hang because it performs arithmetic during resolution. Peano encoding
//! (`Succ<Succ<Zero>>`) is pure structural recursion - no arithmetic needed.

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
// Stack Context Trait
// ============================================================================

/// Retrieve a value from the context stack at type-level index `N`.
pub trait Get<N>: Send + Sync {
    /// Get the value at index N.
    fn get(&self) -> crate::Field;
}

/// The empty context (stack bottom).
#[derive(Clone, Copy, Debug)]
pub struct Empty;

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

// ============================================================================
// Graph Trait (Context-Aware Evaluation)
// ============================================================================

/// A computation graph node that evaluates with a context stack.
///
/// This is the internal evaluation interface for nodes that need access
/// to bound variables. Standard `Manifold` types are automatically lifted
/// to `Graph` via a blanket impl.
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

// ============================================================================
// Let Combinator
// ============================================================================

/// Bind a value to the stack and evaluate the body.
///
/// `Let<Val, Body>` evaluates `Val`, pushes the result onto the context stack,
/// then evaluates `Body` with the extended context.
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
        // 1. Evaluate the value being bound
        let val = self.val.eval_at(ctx, x, y, z, w);

        // 2. Push onto stack (create extended context)
        let new_ctx = (val, *ctx);

        // 3. Evaluate body with extended context
        self.body.eval_at(&new_ctx, x, y, z, w)
    }
}

// ============================================================================
// Var Combinator
// ============================================================================

/// Read a bound variable from the context stack.
///
/// `Var<N>` retrieves the value at Peano index `N` from the stack.
/// Index 0 is the most recently bound value.
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

// ============================================================================
// Lift Manifold to Graph (Blanket Impl)
// ============================================================================

/// Wrapper to lift a `Manifold` into the `Graph` world.
///
/// Manifolds don't use the context, so we just ignore it and call `eval_raw`.
#[derive(Clone, Debug)]
pub struct Lift<M>(pub M);

impl<M, Ctx> Graph<Ctx> for Lift<M>
where
    M: crate::Manifold<crate::Field, Output = crate::Field>,
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
        self.0.eval_raw(x, y, z, w)
    }
}

// ============================================================================
// Graph Binary Operations
// ============================================================================

/// Graph-level addition.
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

/// Graph-level subtraction.
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

/// Graph-level multiplication.
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

/// Graph-level division.
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

// ============================================================================
// Root: Convert Graph to Manifold
// ============================================================================

/// Root node that converts a `Graph` into a `Manifold`.
///
/// Starts evaluation with an empty context, making the graph evaluable
/// as a standard manifold.
#[derive(Clone, Debug)]
pub struct Root<G>(pub G);

impl<G> crate::Manifold<crate::Field> for Root<G>
where
    G: Graph<Empty> + Send + Sync,
{
    type Output = crate::Field;

    #[inline(always)]
    fn eval_raw(
        &self,
        x: crate::Field,
        y: crate::Field,
        z: crate::Field,
        w: crate::Field,
    ) -> crate::Field {
        self.0.eval_at(&Empty, x, y, z, w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Field;

    #[test]
    fn test_peano_get() {
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
    fn test_let_binding() {
        // let x = 5.0; x + x
        // Using Lift to wrap constant, Var to read, GAdd to add
        let graph = Let::new(
            Lift(5.0f32), // val = 5.0
            GAdd(Var::<N0>::new(), Var::<N0>::new()), // x + x
        );

        let zero = Field::from(0.0);
        let result = graph.eval_at(&Empty, zero, zero, zero, zero);

        let mut buf = [0.0f32; crate::PARALLELISM];
        result.store(&mut buf);
        assert_eq!(buf[0], 10.0); // 5 + 5 = 10
    }

    #[test]
    fn test_nested_let() {
        // let a = 3.0; let b = 4.0; a + b
        let graph = Let::new(
            Lift(3.0f32), // a = 3.0 (index 0 in outer, index 1 in inner)
            Let::new(
                Lift(4.0f32),                             // b = 4.0 (index 0 in inner)
                GAdd(Var::<N1>::new(), Var::<N0>::new()), // a + b
            ),
        );

        let zero = Field::from(0.0);
        let result = graph.eval_at(&Empty, zero, zero, zero, zero);

        let mut buf = [0.0f32; crate::PARALLELISM];
        result.store(&mut buf);
        assert_eq!(buf[0], 7.0); // 3 + 4 = 7
    }
}
