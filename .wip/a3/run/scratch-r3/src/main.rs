//! Round-3 soundness probe for A3's value-bound paths (scratch, not in the repo).
//!
//! Differential: a term with value-bounded folds is evaluated by a scalar
//! reference evaluator written here (A4's guard semantics: an out-of-domain
//! value bound makes a real fold NaN), inserted, saturated, extracted, and the
//! extracted arena evaluated the same way. A disagreement, a panic, or an open
//! extracted term is a finding.

use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode, UniformDecl, UniformIdentity};
use pixelflow_ir::fold::{Binder, Bound, Fold, Monoid};
use pixelflow_ir::{Lit, OpKind};
use pixelflow_search::egraph::{
    CostModel, RuleSet, Saturation, SaturationConfig, Vocabulary, extract, fold_rules, insert,
};

const EXACT: f32 = 16_777_216.0;

fn b(slot: u8) -> Binder {
    Binder::from_slot(slot).expect("slot")
}

fn real(lit: Lit) -> f32 {
    match lit {
        Lit::Real(v) => v,
        Lit::Mask(m) => f32::from_bits(if m { u32::MAX } else { 0 }),
    }
}

fn eval(a: &ExprArena, id: ExprId, x: f32, y: f32, env: &mut [f32; 128]) -> f32 {
    match a.node(id) {
        ExprNode::Var(0) => x,
        ExprNode::Var(1) => y,
        ExprNode::Var(v) => env[v as usize],
        ExprNode::Const(lit) => real(lit),
        ExprNode::Uniform(u) => a.uniform_decl(u).default,
        ExprNode::Unary(op, c) => {
            let v = eval(a, c, x, y, env);
            match op {
                OpKind::Neg => -v,
                OpKind::Abs => v.abs(),
                OpKind::Sqrt => v.sqrt(),
                OpKind::Rsqrt => 1.0 / v.sqrt(),
                OpKind::Recip => 1.0 / v,
                OpKind::Floor => v.floor(),
                OpKind::Sin => v.sin(),
                OpKind::Cos => v.cos(),
                other => panic!("unary {other:?}"),
            }
        }
        ExprNode::Binary(op, l, r) => {
            let (l, r) = (eval(a, l, x, y, env), eval(a, r, x, y, env));
            match op {
                OpKind::Add => l + r,
                OpKind::Sub => l - r,
                OpKind::Mul => l * r,
                OpKind::Div => l / r,
                OpKind::Min => l.min(r),
                OpKind::Max => l.max(r),
                other => panic!("binary {other:?}"),
            }
        }
        ExprNode::Ternary(op, p, q, s) => {
            let (p, q, s) = (
                eval(a, p, x, y, env),
                eval(a, q, x, y, env),
                eval(a, s, x, y, env),
            );
            match op {
                OpKind::MulAdd => p * q + s,
                other => panic!("ternary {other:?}"),
            }
        }
        ExprNode::Nary(op, ..) => {
            let kids: Vec<ExprId> = a.children(id).collect();
            let vals: Vec<f32> = kids.iter().map(|&k| eval(a, k, x, y, env)).collect();
            match op {
                OpKind::Add => vals.iter().sum(),
                OpKind::Mul => vals.iter().product(),
                OpKind::MulAdd => vals[0] * vals[1] + vals[2],
                other => panic!("nary {other:?}"),
            }
        }
        ExprNode::Reduce { fold, lo, hi, body } => {
            let mut out_of_domain = false;
            let mut bound = |bd: Bound<ExprId>, env: &mut [f32; 128]| match bd {
                Bound::Lit(n) => n as f32,
                Bound::Value(v) => {
                    let v = eval(a, v, x, y, env);
                    if !(0.0 <= v && v <= EXACT && v.floor() == v) {
                        out_of_domain = true;
                    }
                    v
                }
            };
            let lo = bound(lo, env);
            let hi = bound(hi, env);
            if out_of_domain {
                return f32::NAN;
            }
            let stride = a.known_range(id).map_or(1, |r| r.stride()) as f32;
            let monoid = fold.monoid();
            let op = |acc: f32, t: f32| match monoid {
                m if m == Monoid::SUM => acc + t,
                m if m == Monoid::PRODUCT => acc * t,
                m if m == Monoid::MIN => acc.min(t),
                m if m == Monoid::MAX => acc.max(t),
                m => panic!("monoid {m:?}"),
            };
            let slot = fold.binder().var() as usize;
            let saved = env[slot];
            let mut acc = real(monoid.identity());
            let mut k = lo;
            while k < hi {
                env[slot] = k;
                let t = eval(a, body, x, y, env);
                acc = op(acc, t);
                k += stride;
            }
            env[slot] = saved;
            acc
        }
        ExprNode::Ref(_) | ExprNode::Param(_) | ExprNode::Buffer(_) | ExprNode::Write { .. } => {
            panic!("unexpected node {}", a.display(id))
        }
    }
}

fn close(p: f32, q: f32) -> bool {
    if p.is_nan() || q.is_nan() {
        return p.is_nan() && q.is_nan();
    }
    if p.is_infinite() || q.is_infinite() {
        return p == q;
    }
    (p - q).abs() <= 1e-3 * (1.0 + p.abs().max(q.abs()))
}

struct Case {
    name: &'static str,
    arena: ExprArena,
    root: ExprId,
}

/// Run `case` under `rules`, and report.
fn probe(case: &Case, rules: fn() -> RuleSet, iterations: usize, label: &str) -> bool {
    let mut eg = Saturation::over(rules());
    let class = match insert(&case.arena, case.root, &mut eg, Vocabulary::Runtime) {
        Ok(c) => c,
        Err(e) => {
            println!("[{label}] {}: insert declined {e:?}", case.name);
            return false;
        }
    };
    SaturationConfig::compatibility(iterations).run(&mut eg);
    let class = eg.find(class);
    let (out, out_root, cost) = extract(&eg, class, &CostModel::latency_prior());
    let mut ok = true;
    if let Some(free) = out.free_index(out_root) {
        println!("[{label}] {}: OPEN extracted term, reads Var({free})", case.name);
        ok = false;
    }
    for &(x, y) in &[(0.7f32, 1.3f32), (-2.25, 0.5), (3.0, -1.0)] {
        let mut env = [f32::NAN; 128];
        let want = eval(&case.arena, case.root, x, y, &mut env);
        let mut env = [f32::NAN; 128];
        let got = eval(&out, out_root, x, y, &mut env);
        if !close(want, got) {
            println!(
                "[{label}] {}: MISMATCH at ({x}, {y}): source {want}, extracted {got}\n  source:    {}\n  extracted: {}",
                case.name,
                case.arena.display(case.root),
                out.display(out_root)
            );
            ok = false;
        }
    }
    if ok {
        println!("[{label}] {}: ok (cost {cost}) {}", case.name, out.display(out_root));
    }
    ok
}

fn uniform(a: &mut ExprArena, default: f32) -> ExprId {
    let decl = a.declare_uniform(UniformDecl {
        id: UniformIdentity::mint(),
        default,
    });
    a.push_uniform(decl)
}

/// `⊕o_{i<n} ⊕i_{j ∈ [lo(i), hi(i))} body(x, y, i, j)`, built by hand.
fn nested(
    name: &'static str,
    outer: (Monoid, u64),
    inner: Monoid,
    bounds: impl Fn(&mut ExprArena, ExprId) -> (Bound<ExprId>, Bound<ExprId>),
    body: impl Fn(&mut ExprArena, ExprId, ExprId, ExprId, ExprId) -> ExprId,
) -> Case {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let j = a.push_var(b(0).var());
    let i = a.push_var(b(1).var());
    let t = body(&mut a, x, y, i, j);
    let (lo, hi) = bounds(&mut a, i);
    let inner = a.push_reduce(Fold::new(inner, b(0)), lo, hi, t);
    let root = a.push_reduce(
        Fold::new(outer.0, b(1)),
        Bound::Lit(0),
        Bound::Lit(outer.1),
        inner,
    );
    Case {
        name,
        arena: a,
        root,
    }
}

fn cases() -> Vec<Case> {
    let mut v = Vec::new();
    for n in [3u64, 4, 5, 8] {
        v.push(nested(
            "tri Σ_i Σ_{j<i} X·j",
            (Monoid::SUM, n),
            Monoid::SUM,
            |_, i| (Bound::Lit(0), Bound::Value(i)),
            |a, x, _, _, j| a.push_binary(OpKind::Mul, x, j),
        ));
        v.push(nested(
            "upper Σ_i Σ_{j∈[i,n)} X·j",
            (Monoid::SUM, n),
            Monoid::SUM,
            move |_, i| (Bound::Value(i), Bound::Lit(n)),
            |a, x, _, _, j| a.push_binary(OpKind::Mul, x, j),
        ));
        v.push(nested(
            "body reads outer Σ_i Σ_{j<i} (X·j + i)",
            (Monoid::SUM, n),
            Monoid::SUM,
            |_, i| (Bound::Lit(0), Bound::Value(i)),
            |a, x, _, i, j| {
                let xj = a.push_binary(OpKind::Mul, x, j);
                a.push_binary(OpKind::Add, xj, i)
            },
        ));
        v.push(nested(
            "reversed after peel Σ_i Σ_{j∈[2,i)} X·j",
            (Monoid::SUM, n),
            Monoid::SUM,
            |_, i| (Bound::Lit(2), Bound::Value(i)),
            |a, x, _, _, j| a.push_binary(OpKind::Mul, x, j),
        ));
        v.push(nested(
            "max of sums max_i Σ_{j<i} (X − j)",
            (Monoid::MAX, n),
            Monoid::SUM,
            |_, i| (Bound::Lit(0), Bound::Value(i)),
            |a, x, _, _, j| a.push_binary(OpKind::Sub, x, j),
        ));
        v.push(nested(
            "min inner Σ_i min_{j<i} (X + j)",
            (Monoid::SUM, n),
            Monoid::MIN,
            |_, i| (Bound::Lit(0), Bound::Value(i)),
            |a, x, _, _, j| a.push_binary(OpKind::Add, x, j),
        ));
        v.push(nested(
            "factorable Σ_i Σ_{j<i} (Y·(X + j))",
            (Monoid::SUM, n),
            Monoid::SUM,
            |_, i| (Bound::Lit(0), Bound::Value(i)),
            |a, x, y, _, j| {
                let xj = a.push_binary(OpKind::Add, x, j);
                a.push_binary(OpKind::Mul, y, xj)
            },
        ));
        v.push(nested(
            "arith bound Σ_i Σ_{j<i+u} X·j",
            (Monoid::SUM, n),
            Monoid::SUM,
            |a, i| {
                let u = uniform(a, 2.0);
                let s = a.push_binary(OpKind::Add, i, u);
                (Bound::Lit(0), Bound::Value(s))
            },
            |a, x, _, _, j| a.push_binary(OpKind::Mul, x, j),
        ));
        v.push(nested(
            "both ends Σ_i Σ_{j∈[i, i+u)} X·j",
            (Monoid::SUM, n),
            Monoid::SUM,
            |a, i| {
                let u = uniform(a, 3.0);
                let s = a.push_binary(OpKind::Add, i, u);
                (Bound::Value(i), Bound::Value(s))
            },
            |a, x, _, _, j| a.push_binary(OpKind::Mul, x, j),
        ));
        v.push(nested(
            "uniform inner Σ_i Σ_{j<u} (Y·(X·j + i))",
            (Monoid::SUM, n),
            Monoid::SUM,
            |a, _| {
                let u = uniform(a, 5.0);
                (Bound::Lit(0), Bound::Value(u))
            },
            |a, x, y, i, j| {
                let xj = a.push_binary(OpKind::Mul, x, j);
                let s = a.push_binary(OpKind::Add, xj, i);
                a.push_binary(OpKind::Mul, y, s)
            },
        ));
        v.push(nested(
            "out-of-domain uniform Σ_i Σ_{j<u} X·j, u = 2.5",
            (Monoid::SUM, n),
            Monoid::SUM,
            |a, _| {
                let u = uniform(a, 2.5);
                (Bound::Lit(0), Bound::Value(u))
            },
            |a, x, _, _, j| a.push_binary(OpKind::Mul, x, j),
        ));
        v.push(nested(
            "fraction bound Σ_i Σ_{j<3/2} X·j",
            (Monoid::SUM, n),
            Monoid::SUM,
            |a, _| {
                let three = a.push_const(3.0);
                let two = a.push_const(2.0);
                let q = a.push_binary(OpKind::Div, three, two);
                (Bound::Lit(0), Bound::Value(q))
            },
            |a, x, _, _, j| a.push_binary(OpKind::Mul, x, j),
        ));
        v.push(nested(
            "foldable bound Σ_i Σ_{j<6/2} X·j",
            (Monoid::SUM, n),
            Monoid::SUM,
            |a, _| {
                let six = a.push_const(6.0);
                let two = a.push_const(2.0);
                let q = a.push_binary(OpKind::Div, six, two);
                (Bound::Lit(0), Bound::Value(q))
            },
            |a, x, _, _, j| a.push_binary(OpKind::Mul, x, j),
        ));
        v.push(nested(
            "doubled bound Σ_i Σ_{j<2i} (X − j)",
            (Monoid::SUM, n),
            Monoid::SUM,
            |a, i| {
                let two = a.push_const(2.0);
                let q = a.push_binary(OpKind::Mul, i, two);
                (Bound::Lit(0), Bound::Value(q))
            },
            |a, x, _, _, j| a.push_binary(OpKind::Sub, x, j),
        ));
        v.push(nested(
            "infinite bound Σ_i Σ_{j<u/0} X·j",
            (Monoid::SUM, n),
            Monoid::SUM,
            |a, _| {
                let u = uniform(a, 2.0);
                let zero = a.push_const(0.0);
                let q = a.push_binary(OpKind::Div, u, zero);
                (Bound::Lit(0), Bound::Value(q))
            },
            |a, x, _, _, j| a.push_binary(OpKind::Mul, x, j),
        ));
        v.push(nested(
            "min factor Σ_i min_{j<i} (Y + X·j)",
            (Monoid::SUM, n),
            Monoid::MIN,
            |_, i| (Bound::Lit(0), Bound::Value(i)),
            |a, x, y, _, j| {
                let xj = a.push_binary(OpKind::Mul, x, j);
                a.push_binary(OpKind::Add, y, xj)
            },
        ));
        v.push(nested(
            "product outer Π_i Σ_{j<i} (X + j)",
            (Monoid::PRODUCT, n),
            Monoid::SUM,
            |_, i| (Bound::Lit(0), Bound::Value(i)),
            |a, x, _, _, j| a.push_binary(OpKind::Add, x, j),
        ));
        v.push(nested(
            "i minus i bound Σ_i Σ_{j<i−i+2} X·j",
            (Monoid::SUM, n),
            Monoid::SUM,
            |a, i| {
                let d = a.push_binary(OpKind::Sub, i, i);
                let two = a.push_const(2.0);
                let q = a.push_binary(OpKind::Add, d, two);
                (Bound::Lit(0), Bound::Value(q))
            },
            |a, x, _, _, j| a.push_binary(OpKind::Mul, x, j),
        ));
    }
    // Three levels: Σ_{i<4} Σ_{j<i} Σ_{k<j} (X + k).
    {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let k = a.push_var(b(0).var());
        let j = a.push_var(b(1).var());
        let i = a.push_var(b(2).var());
        let body = a.push_binary(OpKind::Add, x, k);
        let kk = a.push_reduce(Fold::new(Monoid::SUM, b(0)), Bound::Lit(0), Bound::Value(j), body);
        let jj = a.push_reduce(Fold::new(Monoid::SUM, b(1)), Bound::Lit(0), Bound::Value(i), kk);
        let root = a.push_reduce(Fold::new(Monoid::SUM, b(2)), Bound::Lit(0), Bound::Lit(5), jj);
        v.push(Case {
            name: "three levels Σ_{i<5} Σ_{j<i} Σ_{k<j} (X + k)",
            arena: a,
            root,
        });
    }
    // Shadowing: Σ_{i<4} (max_{i<u} (i·X) + i), the inner rebinding slot 1.
    {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let i = a.push_var(b(1).var());
        let u = uniform(&mut a, 3.0);
        let ix = a.push_binary(OpKind::Mul, i, x);
        let max = a.push_reduce(Fold::new(Monoid::MAX, b(1)), Bound::Lit(0), Bound::Value(u), ix);
        let body = a.push_binary(OpKind::Add, max, i);
        let root = a.push_reduce(Fold::new(Monoid::SUM, b(1)), Bound::Lit(0), Bound::Lit(4), body);
        v.push(Case {
            name: "shadowing Σ_{i<4} (max_{i<u} i·X + i)",
            arena: a,
            root,
        });
    }
    v
}

fn main() {
    let mut failures = 0;
    for case in cases() {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            probe(&case, || RuleSet::new(fold_rules()), 40, "fold")
        }));
        match r {
            Ok(true) => {}
            Ok(false) => failures += 1,
            Err(_) => {
                println!("[fold] {}: PANIC", case.name);
                failures += 1;
            }
        }
    }
    for case in cases() {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            probe(&case, RuleSet::runtime, 6, "runtime")
        }));
        match r {
            Ok(true) => {}
            Ok(false) => failures += 1,
            Err(_) => {
                println!("[runtime] {}: PANIC", case.name);
                failures += 1;
            }
        }
    }
    println!("failures: {failures}");
}
