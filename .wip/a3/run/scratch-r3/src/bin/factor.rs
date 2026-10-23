//! Why does FactorFold not factor `Y` out of `Σ_{j<i} Y·(X + j)`?
use pixelflow_ir::arena::ExprArena;
use pixelflow_ir::fold::{Binder, Bound, Fold, Monoid};
use pixelflow_ir::OpKind;
use pixelflow_search::egraph::{
    CostModel, ENode, RuleSet, Saturation, SaturationConfig, Vocabulary, extract, fold_rules,
    insert, FactorFold, Rewrite,
};

fn b(s: u8) -> Binder {
    Binder::from_slot(s).unwrap()
}

fn main() {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let j = a.push_var(b(0).var());
    let i = a.push_var(b(1).var());
    let xj = a.push_binary(OpKind::Add, x, j);
    let body = a.push_binary(OpKind::Mul, y, xj);
    let inner = a.push_reduce(Fold::new(Monoid::SUM, b(0)), Bound::Lit(0), Bound::Value(i), body);
    let outer = a.push_reduce(Fold::new(Monoid::SUM, b(1)), Bound::Lit(0), Bound::Lit(3), inner);

    // The inner fold alone.
    for (name, root) in [("inner alone", inner), ("outer", outer)] {
        let mut eg = Saturation::over(RuleSet::new(fold_rules()));
        let class = insert(&a, root, &mut eg, Vocabulary::Runtime).unwrap();
        // Ask FactorFold directly of every fold node in the graph.
        for c in eg.class_ids().collect::<Vec<_>>() {
            for n in eg.nodes(c).copied().collect::<Vec<_>>() {
                if let ENode::Reduce { .. } = n {
                    let r = FactorFold.apply(&eg, c, &n);
                    println!("{name}: class {:?} fold {:?} -> FactorFold {}", c, n.as_reduce(), if r.is_some() {"fires"} else {"declines"});
                }
            }
        }
        let stats = SaturationConfig::compatibility(10).run(&mut eg);
        println!("{name}: stats {stats:?}");
        let class = eg.find(class);
        let (out, r, cost) = extract(&eg, class, &CostModel::latency_prior());
        println!("{name}: cost {cost} {}", out.display(r));
    }
}
