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

#[cfg(test)]
mod tests {
    use super::*;

    /// A stage given as a function on `u64`. The law below is about how
    /// stages group, not about any stage, so one shape serves all three.
    struct Stage(fn(u64) -> Result<u64, CompileError>);

    impl Pass for Stage {
        type Input = u64;
        type Output = u64;

        fn run(self, input: u64) -> Result<u64, CompileError> {
            (self.0)(input)
        }
    }

    /// The bound `double` refuses above, so the chain has a refusal in its
    /// middle as well as at its ends.
    const BOUND: u64 = 50;

    fn increment() -> Stage {
        Stage(|x| Ok(x + 1))
    }

    fn double() -> Stage {
        Stage(|x| {
            if x > BOUND {
                Err(CompileError::Legalize("double: input over the bound"))
            } else {
                Ok(x * 2)
            }
        })
    }

    fn add_ten() -> Stage {
        Stage(|x| Ok(x + 10))
    }

    /// `increment`, then `double`, then `add_ten`, written out: the value
    /// the chain must equal wherever it does not refuse.
    fn chain_oracle(x: u64) -> Result<u64, CompileError> {
        let y = x + 1;
        if y > BOUND {
            Err(CompileError::Legalize("double: input over the bound"))
        } else {
            Ok(y * 2 + 10)
        }
    }

    /// `(a.then(b)).then(c)` and `a.then(b.then(c))` give the same value, and
    /// the value is the chain's: the stages in the order they were chained.
    #[test]
    fn then_is_associative_on_values() {
        for x in [0, 1, 5, 48, 49, 50, 1000] {
            let left = increment().then(double()).then(add_ten()).run(x);
            let right = increment().then(double().then(add_ten())).run(x);
            assert_eq!(left, right, "groupings disagree at {x}");
            assert_eq!(left, chain_oracle(x), "chain at {x} is not its stages");
        }
    }

    /// The first `Err` ends the chain, and it is the same error in either
    /// grouping: a later stage's refusal never replaces an earlier one.
    #[test]
    fn then_is_associative_on_the_first_error() {
        let refuse_first = || Stage(|_| Err(CompileError::BudgetExceeded("first")));
        let refuse_later = || Stage(|_| Err(CompileError::Mmap));

        let left = refuse_first()
            .then(refuse_later())
            .then(refuse_later())
            .run(0);
        let right = refuse_first()
            .then(refuse_later().then(refuse_later()))
            .run(0);
        assert_eq!(left, right);
        assert_eq!(left, Err(CompileError::BudgetExceeded("first")));

        let left = increment().then(double()).then(refuse_later()).run(100);
        let right = increment().then(double().then(refuse_later())).run(100);
        assert_eq!(left, right);
        assert_eq!(
            left,
            Err(CompileError::Legalize("double: input over the bound"))
        );
    }
}
