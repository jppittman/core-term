//! The whole pipeline's contract, `front_end` (`lib.rs`): what every stage
//! together accepts and refuses, under both macros' optimizers.

/// Every method name the front end advertises must survive both macros.
///
/// `Method::resolve` settles a method's name and arity once, in the parser,
/// and every later stage matches the resulting value exhaustively. What
/// still has to be shown is that every `Method` the parser resolves survives
/// `sema` typing, lowering and emission, which no test of a single stage can
/// see: the stage under test is the one that is right. Nor can sampling: a
/// method is exercised only if some test happens to write a kernel calling it.
/// So run the real pipeline, both macros' versions of it, over the whole
/// advertised surface. Adding an op to `OpKind::is_dsl_method` or a
/// `LibraryMethod` without a path through every stage fails here, for whoever
/// adds it.
mod every_advertised_method_compiles {
    use crate::ast::Method;
    use crate::{Identity, Optimize, front_end, macro_tier};
    use proc_macro2::Span;
    use quote::quote;
    use syn::Ident;

    /// Which macro's pipeline to run. `kernel!` saturates the e-graph between
    /// sema and lowering; `kernel_raw!` goes straight across. The difference
    /// between them is where the `hypot`/`fract` asymmetry lived, so both are
    /// swept.
    #[derive(Clone, Copy, Debug)]
    enum Macro {
        Kernel,
        KernelRaw,
    }

    /// Expand a call of `method` through `which` macro's own pipeline — the
    /// [`front_end`] `kernel!` and `kernel_raw!` run — and report whether it
    /// yields code.
    ///
    /// With no knowledge of how the method types: the language has two value
    /// types, so the receiver is tried as a number, `X`, and then as a mask,
    /// `X.lt(X)`, every argument `X`, and the method passes iff one of the two
    /// compiles. A method neither admits reports the number's refusal, the
    /// plainer attempt.
    fn expand(which: Macro, method: &str, arg_count: usize) -> Result<(), String> {
        let name = Ident::new(method, Span::call_site());
        let args = vec![quote!(X); arg_count];
        let on = |receiver: proc_macro2::TokenStream| {
            let body = quote! { || #receiver.#name(#(#args),*) };
            let mut kernel_optimizer;
            let mut raw_optimizer;
            let optimizer: &mut dyn Optimize = match which {
                Macro::Kernel => {
                    kernel_optimizer = macro_tier();
                    &mut kernel_optimizer
                }
                Macro::KernelRaw => {
                    raw_optimizer = Identity;
                    &mut raw_optimizer
                }
            };
            front_end(body, optimizer)
                .map(|_| ())
                .map_err(|e| e.to_string())
        };
        let as_a_number = on(quote!(X));
        if as_a_number.is_ok() {
            return Ok(());
        }
        on(quote!(X.lt(X))).or(as_a_number)
    }

    #[test]
    fn through_the_kernel_macros_pipeline() {
        for (method, arg_count) in Method::advertised() {
            assert_eq!(
                expand(Macro::Kernel, method, arg_count),
                Ok(()),
                "`kernel!(|| X.{method}(..))` is advertised but does not compile"
            );
        }
    }

    #[test]
    fn through_the_kernel_raw_macros_pipeline() {
        for (method, arg_count) in Method::advertised() {
            assert_eq!(
                expand(Macro::KernelRaw, method, arg_count),
                Ok(()),
                "`kernel_raw!(|| X.{method}(..))` is advertised but does not compile"
            );
        }
    }

    /// The converse guard: a name nothing advertises must still be refused,
    /// so the sweeps above cannot be satisfied by accepting everything.
    #[test]
    fn a_name_the_front_end_does_not_advertise_is_still_refused() {
        assert!(expand(Macro::Kernel, "not_a_real_method", 0).is_err());
        // A real op at the wrong arity is just as unadvertised.
        assert!(expand(Macro::Kernel, "sqrt", 2).is_err());
    }

    /// A recognized name at the wrong arity must say so. Refusing it is not
    /// enough: `.sqrt(1.0)` is a hard error, and the refusal must
    /// report the arity rather than fall into the typo-suggestion search, which
    /// would answer `unknown method 'sqrt'; did you mean 'sqrt'?`.
    #[test]
    fn a_known_name_at_the_wrong_arity_reports_the_arity() {
        let err = expand(Macro::Kernel, "sqrt", 2).expect_err("wrong arity must fail");
        assert!(
            err.contains("takes 0 arguments"),
            "expected an arity diagnostic, got: {err}"
        );
        assert!(
            !err.contains("did you mean"),
            "a recognized name must not be sent to the typo search: {err}"
        );
    }
}
