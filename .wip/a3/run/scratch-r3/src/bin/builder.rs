//! The builder path (open_fold / close_fold) with a value bound that is the
//! enclosing fold's index: J5's placeholder variance and the re-check at close.
use pixelflow_ir::arena::{ExprArena, ExprNode, FoldTerms};
use pixelflow_ir::fold::{Bound, Monoid};
use pixelflow_ir::variance::compute_arena_variance;
use pixelflow_ir::OpKind;

fn main() {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let outer = a.open_fold(0).expect("depth 0");
    let i = outer.index();
    let inner = a.open_fold(1).expect("depth 1");
    let j = inner.index();
    let body = a.push_binary(OpKind::Mul, x, j);
    let terms = FoldTerms { monoid: Monoid::SUM, stride: 1, lo: Bound::Lit(0), hi: Bound::Value(i) };
    let inner_fold = a.close_fold(inner, body, terms).expect("a binder");
    let body = a.push_binary(OpKind::Add, inner_fold, i);
    let terms = FoldTerms { monoid: Monoid::SUM, stride: 1, lo: Bound::Lit(0), hi: Bound::Lit(4) };
    let root = a.close_fold(outer, body, terms).expect("a binder");
    println!("{}", a.display(root));
    println!("free_index {:?}", a.free_index(root));
    println!("variance {:?}", compute_arena_variance(&a)[root.0 as usize]);
    if let ExprNode::Reduce { fold, body, .. } = a.node(root) {
        println!("outer binder slot {}", fold.binder().slot());
        let ExprNode::Binary(_, f, _) = a.node(body) else { panic!() };
        if let ExprNode::Reduce { fold, hi, .. } = a.node(f) {
            println!("inner binder slot {}, hi {:?} = {}", fold.binder().slot(), hi, hi.value().map(|v| a.display(v).to_string()).unwrap_or_default());
        }
    }

    // A bound that reads its own fold's placeholder through the builder: refused.
    let r = std::panic::catch_unwind(|| {
        let mut a = ExprArena::new();
        let open = a.open_fold(0).unwrap();
        let i = open.index();
        let terms = FoldTerms { monoid: Monoid::SUM, stride: 1, lo: Bound::Lit(0), hi: Bound::Value(i) };
        let _ = a.close_fold(open, i, terms);
    });
    println!("own-placeholder bound refused: {}", r.is_err());
}
