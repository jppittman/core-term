//! The compiler's stages as values, and the one way to put them in a row.

use crate::error::CompileError;

/// One stage of the compiler: a function from what the stage before it made
/// to what the stage after it reads. It carries its own parameters (a shape,
/// a lane count, a backend) as fields, takes only its input, and calls no
/// other pass.
///
/// Stages compose with [`then`](Pass::then), Kleisli composition in
/// `Result<_, CompileError>`: `a.then(b).run(x)` is `a.run(x)` and, if that
/// succeeded, `b.run` of its output. So `then` is associative —
/// `a.then(b).then(c)` and `a.then(b.then(c))` run the same stages in the same
/// order and end in the same value or the same first error — and the first
/// `Err` ends the chain: no later stage runs.
///
/// A chain is a type, built where its stages' parameters are known: a backend
/// or a tier is chosen once, when it is built, and nothing dispatches on it
/// per use. There is no `dyn Pass`, registry or list of passes at run time.
pub(crate) trait Pass: Sized {
    /// What the stage reads.
    type Input;
    /// What the stage makes.
    type Output;

    /// Run the stage.
    ///
    /// # Errors
    ///
    /// Whatever the stage refuses.
    fn run(self, input: Self::Input) -> Result<Self::Output, CompileError>;

    /// This stage, then `next` on its output.
    fn then<Next: Pass<Input = Self::Output>>(self, next: Next) -> Then<Self, Next> {
        Then(self, next)
    }
}

/// Two stages in a row: [`Pass::then`]'s result, a stage itself.
pub(crate) struct Then<First, Second>(First, Second);

impl<First: Pass, Second: Pass<Input = First::Output>> Pass for Then<First, Second> {
    type Input = First::Input;
    type Output = Second::Output;

    fn run(self, input: Self::Input) -> Result<Self::Output, CompileError> {
        self.1.run(self.0.run(input)?)
    }
}
