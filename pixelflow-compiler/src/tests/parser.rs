//! The parser's contract (`parser/mod.rs`): the `KernelDef` a token stream
//! spells, or the refusal at the token that spells nothing.

use crate::ast::{BinaryOp, ClosureEntry, Expr, KernelDef, UnaryOp};
use crate::parser::parse;
use proc_macro2::TokenStream;
use quote::quote;

/// The closure sugar's one entry.
fn entry(def: &KernelDef) -> &ClosureEntry {
    let KernelDef::Closure(entry) = def else {
        panic!("the closure form parses to a closure entry");
    };
    entry
}

/// The body `|| X #op X` parses to.
fn binary(op: &TokenStream) -> Expr {
    let def = parse(quote!(|| X #op X)).unwrap_or_else(|e| panic!("`{op}`: {e}"));
    entry(&def).body.clone()
}

#[test]
fn every_supported_binary_operator_parses_to_its_own_variant() {
    let cases: &[(TokenStream, BinaryOp)] = &[
        (quote!(+), BinaryOp::Add),
        (quote!(-), BinaryOp::Sub),
        (quote!(*), BinaryOp::Mul),
        (quote!(/), BinaryOp::Div),
        (quote!(<), BinaryOp::Lt),
        (quote!(<=), BinaryOp::Le),
        (quote!(>), BinaryOp::Gt),
        (quote!(>=), BinaryOp::Ge),
        (quote!(==), BinaryOp::Eq),
        (quote!(!=), BinaryOp::Ne),
        (quote!(&), BinaryOp::BitAnd),
        (quote!(|), BinaryOp::BitOr),
    ];
    for (op, expected) in cases {
        let Expr::Binary(binary) = binary(op) else {
            panic!("`X {op} X` is a binary expression");
        };
        assert_eq!(binary.op, *expected, "{op}");
    }
}

/// `%` has no IR op and `+=` assigns: neither is a kernel operator, and each
/// is refused at the operator, naming it.
#[test]
fn an_unsupported_binary_operator_is_refused_at_the_operator() {
    let cases: &[(TokenStream, &str)] = &[
        (quote!(+=), "`+=` assigns"),
        (quote!(%), "`%` has no IR op"),
        (quote!(^), "unsupported binary operator `^`"),
        (quote!(&&), "unsupported binary operator `&&`"),
    ];
    for (op, claim) in cases {
        let refusal = parse(quote!(|| X #op X))
            .err()
            .unwrap_or_else(|| panic!("`X {op} X` is refused"))
            .to_string();
        assert!(refusal.contains(claim), "`{op}`: {refusal}");
    }
}

/// `!` has no IR op, so it is not a kernel operator either.
#[test]
fn negation_is_the_one_unary_operator() {
    let def = parse(quote!(|| -X)).unwrap();
    let Expr::Unary(unary) = &entry(&def).body else {
        panic!("`-X` is a unary expression");
    };
    assert_eq!(unary.op, UnaryOp::Neg);
    let refusal = parse(quote!(|| !X)).unwrap_err().to_string();
    assert!(refusal.contains("`!` has no IR op"), "{refusal}");
}

/// An entry's `Args` record is its name in UpperCamelCase, then `Args`;
/// a raw identifier is named by what it spells.
#[test]
fn an_entrys_args_record_is_its_name_in_upper_camel_case() {
    for (entry, args) in [
        ("circle", "CircleArgs"),
        ("shifted_radius2", "ShiftedRadius2Args"),
        ("a__b_", "ABArgs"),
        ("r#type", "TypeArgs"),
        ("_1", "_1Args"),
        ("__3", "_3Args"),
    ] {
        let input: TokenStream = format!("pub fn {entry}() -> f32 {{ X }}")
            .parse()
            .expect("an items block");
        let def = parse(input).unwrap_or_else(|e| panic!("{entry}: {e}"));
        let [item] = def.fns() else {
            panic!("{entry}: one fn item");
        };
        assert_eq!(item.args_record().to_string(), args, "{entry}");
    }
}
