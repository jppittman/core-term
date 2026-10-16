> **Archived 2026-10-16 — status `Landed`.** The e-graph's store as an arena (node ids are tags, membership an intrusive list, per-id columns, a per-kind class index, a cache value with no `add`), byte-identical by gate: all 13 CLs landed in five PRs (#1416, #1418, #1420, #1422, #1423). §7 and `docs/research-problems/2026-10-15-the-egraph.md` hold what remains; §8's decisions are JP's. Kept for rationale; not authoritative about the tree.

# The e-graph is an arena

**Status:** `Landed` (JP, 2026-10-15: "write the arena plan. Execute it."). All 13 CLs of §6 are in, as five PRs:
group A (CLs 0, 3a, 2) #1416, group B (CLs 5a–d, 6) #1418, group C (CLs 1, 3b) #1420, group D (CLs 3c, 4) #1422, and
group E (CL 7) #1423. What is left is §7, which is research, and §8, which is JP's.
**Verified against:** `43f6923a` (CL 7 and its review fixes, the final source commit); written at `2a99d933`; CLs 0, 3a and 2 landed as `4af25113`; CLs 5a–d and 6 on `43019669`; CLs 1 and 3b on `98fb75a7`; CLs 3c and 4 on `4183ecf6`; CL 7 first on `ae195ab2`. Changes
representation only: every CL below left `byte_probe` and the 29 font
programs' extracted kernels byte-identical (§3.2, gates 1 and 1b) and was
measured, not asserted. What would change the optimizer (congruence
completeness, deduplication, match order, the budget) is out of scope and is
stated as research in `docs/research-problems/2026-10-15-the-egraph.md`.

JP, 2026-10-15: "Can this thing be an arena like the other dag? What's
actually going on where this is slow?"

It can, because the stored part already is one with the arena hidden: every
node is added once, takes the next id and names only classes older than
itself (law 1), which is `ExprArena`'s shape. It is not the textbook
e-graph: it is hash-consed at insertion only, not closed under congruence,
and its quotient by `≡` has cycles. `pixelflow-search` stores it the classic
way, a `Vec<EClass>` whose every class owns a heap `Vec<ENode>` and a
parallel `Vec<ENodeId>`, beside five more per-id vectors whose equal length
is a comment (`graph.rs:52-124`). On the cold compile of ASCII's 29 font
programs (`/home/user/scratch/egraph/prof2/profile.md`) saturation is 51% of
instructions, malloc/free 17-18% of all of them, and the rule scan visits
15.2 M (rule, class) pairs to find 502 k matches.

## 1. Denotation

The state is `(A, ≡, R, <, M, W, F)`:

- **`A`**, every e-node ever added, in id order. A node applies an operation
  to *classes* (`ENode::Op { op, children }`, a fold over a body class, a
  shift of an operand class, or a leaf).
- **`≡`**, an equivalence on ids, kept as a union-find; `find(t)` is the
  smallest id in `t`'s block.
- **`R ⊆ A`**, the retired nodes (law 7).
- **`<`**, a total order on each class's members (law 8), not id order.
- **`M`**, the memo: canonical node forms to ids (law 2).
- **`W`**, the worklist: a stack of slots and their `queued` bits. It
  decides which congruences ever fire (law 5).
- **`F`**, the facts: `const_fact`, `var_fact` and `last_changed` per
  representative, the stamp counter, `refused_const_unions`.

Derived: `members(c) = ({t : find(t) = c} \ R, <)`;
`live = {c = find(c) : members(c) ≠ ∅}`; `node(c)`, the node whose id is
`c`. **`node(c)` need not be a member of `c`, nor its first member.**
"Representative" means `find(t)` only. The node `PeelFold`, `HalveFold`
(`fold_rules.rs:46-55`, `274-276`) and `ChainRule` (`derivative.rs:67`)
substitute through is the **first member**, which law 8 decides.

**The tag is the id.** A node's id, the id of the class it founds and its
tag `ENodeId` are one integer at birth, without exception: `add`
(`graph.rs:346-376`) alone mints an id or pushes a class, `Clone` copies both
counters, and `GrowthPredictor`'s virtual ids (`graph.rs:1242`) name no node.
Only a `debug_assert_eq!` under telemetry checks it (`runner.rs:1196-1204`).

**One id domain.** `add` returns every index as an `EClassId`
(`graph.rs:357`), so the domain is `EClassId`'s `u32`, for its measured
reason (`node.rs:11-48`: 24- to 40-byte `ENode`, +18% peak RSS). `ENodeId`
stays `u64` and ranges over it; node-to-class conversion is total inside the
store.

Laws:

1. **Operands are older.** Every stored operand id is smaller than its
   node's id: `add` canonicalizes, so `find(child) ≤ child < len`
   (`graph.rs:346-358`), and `rebuild` only lowers stored ids. `A` is
   acyclic. Cycles exist only in the quotient by `≡` (`x·1 ≡ x`), which is
   why `fold_rules.rs:195-217` keeps `on_stack` and extraction has
   `choices_have_cycle_from` and `choices_have_cycle_through`.
2. **Hash-consing at insertion only.** `add(n)` returns `find(M[canon(n)])`
   on a hit, else mints exactly one id. `M` maps every canonical form
   computed on a miss, by `add` *or a rebuild item* (`graph.rs:571`), to the
   class it was in then. Entries are only added, never re-keyed or pruned:
   `M` is history, not a function of the partition. Nothing compares a node
   with its class's other members, so duplicates are kept.
3. **Union only adds equalities.** Nothing splits a class; the survivor is
   the smaller id.
4. **A refused union adds no equality.** Classes with unequal constant facts
   never merge (`+0.0`/`-0.0` and identical NaN bits are equal). A refusal
   records `(bits, bits)` once in the public `refused_const_unions` (pinned
   by `fold_exactness.rs`) and returns `min(a, b)` as if it had merged
   (`graph.rs:456`); callers tell it apart only by `union_counted`'s 0.
5. **Congruence, as far as the worklist reaches.** If `n ≡ n'` operand-wise
   and both were canonicalized after the union that made them so, their
   classes merge. No use index, so a union does not revisit its parents:
   closure is incomplete (issue #1106), visibly so only with `M` and `W`
   as state.
6. **Stored names are state.** `rebuild` replaces each stored operand with
   its representative, and the stored name is observable: `Commutative`
   compares operands raw (`algebra.rs:507`; a stale `a != b` can mint a
   class and move a cap stop) and `witness::induce` keys a `Reduce` raw on
   purpose (`witness.rs:346-351`). So `A` is append-only in length, not in
   content, and byte identity covers stored names. Reading them through
   `find` is research (§7).
7. **Retired nodes.** A node whose canonical form collides with a class it
   cannot merge with leaves its class's members (`graph.rs:567-569`); a
   founder can be retired too. `≡` means "same value" only between members:
   for a retired `t`, `find(t)` is the class that dropped it, and its value
   is in `find(M[canon(t)])`. A retired node answers nothing: not to rules,
   extraction, `node_for_tag` or `children_of`.
8. **Order.** `add` gives `[t]`; `union(p, c)` gives `p ++ c`. A rebuild
   item detaches `find(slot)`'s members, its mid-item unions append as
   `union` does, and at the end the kept nodes, in walk order, are appended
   to `find(slot)`'s members (`graph.rs:513-594`): `[absorbed] ++ [kept]`
   when the class survives, the survivor's prior members first when it
   loses. Nothing pins this today: the tests at `graph.rs:1953` and `2065`
   assert presence with `.any`, and `settlement_order_is_pinned`
   (`extract.rs:2882`) digests `choices` on an SDF graph without fold rules.
   CL 0 adds the pin.

Membership (`find(t) = c` for every member `t` of `c`) is held by §3.3's
test and by the store's write-back, which takes a `Root` (§2.4).

## 2. The representation

### 2.1 The node arena (CL 3b)

`EGraph` holds `nodes: Vec<ENode>`; `add` pushes at `len` and returns it as
an `EClassId`. The tags vectors and `next_enode_id` are deleted: `tags(c)`
yields member ids, `node(t)` is an index.

- `node_for_tag(id, t)` keeps both arguments: `Some(node(t))` iff
  `find(t) == find(id)` and `t ∉ R`, the "`None` is not in this class now"
  that `apply_single_rule` relies on (`runner.rs:586`).
- `children_of(t)` (`derivation_ancestors*`) is `node(t).children()`, and
  empty for `t ∈ R` as today (`graph.rs:664-669`, `708-713`), with no scan
  of every class. `is_member(t)` serves
  `guide_scope_saturation_delta.rs:373-386`.
- `node_count()` (`graph.rs:772`) keeps its meaning, minted minus retired
  with duplicates counted, as `len − |R|` from a counter.
- Every other reader of the counter reads `len`: `GrowthPredictor`
  (`graph.rs:1226`, `1242`), the journal bracket
  `minted_from..next_enode_id` (`runner.rs:1143`, `1157`) and the telemetry
  delta (`runner.rs:1193-1195`).
- `ENodeId`'s doc says "process-wide" (`provenance.rs:88`, `graph.rs:60`),
  false today; it becomes "the node's index in this graph".

The memo keeps a *copy* of each key. It cannot become ids hashed by the
arena's contents, which `rebuild` rewrites (law 6) while a key keeps the form
it was computed in (law 2): the first thing a design modelled on the
immutable `ExprArena` would get wrong.

### 2.2 Membership is an intrusive list (CL 4)

Order is semantics: 40 production rules return on the first matching node,
substitution uses the first member, `Insertion` keeps the first admissible
candidate (`extract.rs:1737-1744`), and which applications land before the
cap depends on it. CL 3b keeps a per-class `Vec<ENodeId>`; CL 4 replaces it
with a singly linked list through the arena that reproduces law 8:

- per node `next`, per representative `head` and `tail`; "no node" is
  `u64::MAX`, private to the store, so a link is 8 bytes rather than an
  `Option<u64>`'s 16;
- `add`: `[t]`. `union`: splice the child's list after the parent's tail,
  O(1). An empty child splices nothing; an empty parent takes the child's
  head. Today's `extend` does both implicitly (`graph.rs:461-464`), and both
  are ordinary: the class under rebuild is empty when its item's first
  mid-item union makes it the parent or the child. Retirement cannot empty a
  class: a refusal needs a constant fact on both sides, a fact is seeded only
  by a `Const` node, and that node's own form hits its own memo entry, so it is
  never retired.

`nodes(c)` and `tags(c)` are opaque iterators from CL 1. All 53 production
rules that read a child class already read it by an ordered scan with no
index, slice or length (inventory A §4).

This is the plan's performance bet: a class walk becomes one dependent load
per node instead of one per class (live classes average about 3.4 nodes at
the cap). CL 3b takes a smaller dose of the same risk, since every member
read becomes an indirect load. Both are decided by gate 5's cache metric,
not by Ir, which cannot see a load-latency chain.

### 2.3 Rebuild in place

S1 (commit `1be37dc5`) already rebuilds in place: −2.98% Ir, `byte_probe`
identical. It lands first, as CL 0.

In CL 3b an item takes `id = find(slot)`'s member vector out of the class,
as `mem::take` does today, so a mid-item union `id` survives appends to the
class and not to the vector being walked (compacting it attached would give
`[kept] ++ [unprocessed] ++ [absorbed]`). It walks the vector, canonicalizing
`nodes[t]` in the arena and probing, unioning and retiring as today,
compacts the kept ids in place with a write cursor, and writes back to
`find(id)`: a move when that class is empty, an extend otherwise. In CL 4 the
item detaches `id`'s list, walks it reading `next` before relinking, gathers
kept nodes in a local list (two ids) and splices it after `find(id)`'s tail.
Neither allocates per item.

CL 3b corrects four comments that claim more than rebuild does:
`graph.rs:496` and `ClassContentKey`'s "distinct nodes" (`candidate.rs:111`)
(rebuild deduplicates nothing); `Canonical`'s totality
(`extract.rs:1749-1756`), which rests on the same premise, so with duplicates
its result depends on insertion order; and `predicted_growth`'s "upper bound"
(`graph.rs:850-858`, `runner.rs:960-964`), which is exact per application,
not a bound across a batch, since an earlier commit's union can turn a later
predicted hit into a miss.

### 2.4 Per-id columns (CL 3a)

Every per-id column grows in `add` and nowhere else. They go behind one
private type with one `push`, so equal length is a property of the type, and
the type keeps their three meanings apart:

- **per node**: the node, `parent`, `next`, the retired bit;
- **per representative**: `head`, `tail`, `const_fact`, `var_fact`,
  `last_changed`, CL 2's kind mask and per-kind class bitsets. They are
  indexed by a private `Root` that only the store's internal `root(t)`
  returns (the public `find` keeps returning `EClassId`), so reading one at a
  non-representative, where it is silently stale, does not type-check;
- **per worklist slot**: `queued`, cleared by the popped slot, not its root
  (`graph.rs:511-514`).

Facts merge in `union` as today. From CL 3b the member store is the only
thing that fills or empties a class (`add`, union's append, rebuild's detach
and write-back), so it maintains `LiveClasses` itself, and `track_liveness`'s
call sites and both drift checks (`graph.rs:597-602`, `runner.rs:892`) go:
drift is no longer representable.

### 2.5 The per-kind class index (CL 2)

The scan's class filter (`runner.rs:913-916`) asks of every live class, for
every rule, whether any member has a root shape the rule admits: 15.2 M
(rule, class) visits making 36.1 M `admits` calls, 88% of the visits rejected,
0.72 G Ir. It becomes a lookup:

- **One kind function**, `RootShapes::shape_of` (`rewrite.rs:288-307`), for
  `admits`, the masks and the bitsets. `ENode::op()` is `None` for every leaf
  and `Reduce`, where the fold rules root. A `Ref` has no shape, sets no bit.
- **A bit per `OpKind`** needs a numbering, and `OpKind::index()` is
  `pub(crate)` on purpose (`docs/archive/designs/opkind-numbering-is-private.md`).
  So an `OpSet` (`u64`) goes beside `OpMap` in `pixelflow-ir/src/kind.rs`
  with a const assertion that the kinds fit, and the public
  `RootShapes::Kinds` carries one (§5).
- Per class, a mask of the kinds its members have had; per kind, a bitset of
  classes. `add` sets both; `union` ORs the child's mask into the parent's
  and sets the parent's bit for those kinds. Bits are never cleared:
  `by_kind[k] ⊇ {c : some member of c has kind k}`, `debug_assert!`ed once
  per full `rebuild()`, not per rule. A stale bit costs only the per-node
  `admits`, which stays and decides.
- `sweep_rule` walks `live ∧ ⋁_{k ∈ rule} by_kind[k]` ascending, a word at
  a time. No production rule returns `RootShapes::Any`.
- The minted-cap and allowance checks (`runner.rs:899`, `907`) move before
  the loop, in that order, **only when `live` is non-empty**: an empty graph
  never reaches them today and returns `Completed`, advancing `swept`
  (`runner.rs:747-770`). Their inputs change only inside the node loop,
  where an identical check follows each change (`runner.rs:966-979`). The
  deadline check stays per visited class, so a caller with a deadline can
  stop elsewhere; production passes none (`saturate_budgeted`,
  `runner.rs:675-677`).

### 2.6 Extraction reads node ids (CL 3c)

`AndOrGraph::of` walks every live class's members in order once per
saturated graph, and a candidate is computed from its slot
(`and_or.rs:91-108`). CL 3c stores one column, `node_of_slot`, and `Pick`
holds the chosen node's id instead of its ordinal: one choice, one name.
`choices` come from the ordinal the settlement already writes
(`extract.rs:1919`); `Selection::of_map` (`extract.rs:808-858`) derives the
id by `tags(class).nth(idx)`, so `costed`'s `debug_assert_eq!`
(`extract.rs:1549-1554`) compares like with like. The four production reads
of `nodes(class)[idx]` (`extract.rs:782, 879, 1870, 1914`) become `node(id)`.

`Selection::node`'s arity check (`extract.rs:879-893`) guards "another graph,
or rebuilt since"; node ids remove the second case only, and the public
`Optimized::to_arena` documents that panic (`optimizer.rs:290-308`). So it
becomes the O(1), stronger `find(id) == pick.class && id ∉ R`, kept in
production. (A brand tying `Selection` to its graph would make it a type;
not this plan.) CL 3c is needed only by CL 4, since `node_at`'s positional read
over CL 3b's vectors is O(1) (`nodes(c).nth(k)` is a walk of `k + 1` members,
because `nodes` is an iterator adapter), and lands just before it.

Public choice APIs keep positions: `ExtractedDAG.choices`,
`Optimized.choices`, `Extraction::choice` and the labeler's
`&[Option<usize>]` are ordinals in an unchanged member order, still pinned
by `settlement_order_is_pinned`. Seam, research and bin sites that index by
position (about a dozen structures) use `nth`. Changing that is §8.

### 2.7 `ENode` and `ExprNode`

They stay two types because they mean different things: `ENode`'s `Const`
is bits (it must be `Eq`/`Hash`), its `Buffer`/`Uniform` hold declarations by
value because classes outlive any one arena, its shift count is node data;
`ExprNode` has store nodes the e-graph refuses and n-ary children in a side
table. One node functor over its child id in `pixelflow-ir` would make
`insert`, `AndOrGraph::of` and `to_arena` maps rather than translations; that
is its own plan. `dag::Builder` is the same arena with immutable nodes and a
`Vec` per memo key, so it is not reused.

### 2.8 Memory

Signed peak-RSS estimates against a 133 MB peak, measured per CL
(`ru_maxrss`, median of 3), never asserted. Per-id columns double and the
caps land just past powers of two (capacity/size 1.44), so today's 48-byte
`EClass` header and the new columns roughly cancel. The saving is the
per-live-class heap: about 160 B of nodes and tags (an emptied `Vec` regrows
to at least 4 slots) × 134 k live classes ≈ 21 MB.

| CL | estimate |
|---|---|
| 0 (S1) | measured −0.3 MB |
| 3a, 1, 5a-d, 6 | ~0 |
| 2 | +9 to +10 MB while the graph is cached (8-byte mask, 53 bits per id, × 1.44) |
| 3b | about −15 MB (node and tag vectors; a member-id vector remains); measured −10.5 MB |
| 3c | up to +3.6 MB (`node_of_slot`); measured +3.9 MB |
| 4 | about −5 MB (the member-id vector, ~40 B per live class); measured −6.7 MB |
| 7 | the memo and saturation-only columns; the memo's entry count at freeze is measured first; measured −34.0 MB (123,660 → 89,696 KiB, −27.5%) |

The shape cache kept every saturated graph for the process, and nothing
reads the memo after saturation: its readers are `add`, rebuild and
`GrowthPredictor`. CL 7 measured first, as this
row said it would. At the moment a graph entered the cache, the 29 programs'
memos held 458,667 entries for 449,637 ids (one per `add` miss, and about 2%
more from rebuild items that missed: law 2), 26.22 MB at capacity; the memo,
the worklist, `queued`, `last_changed`, the kind mask and the per-kind
bitsets together held 38.03 MB at capacity, and the worklist was empty at all
29 freezes. A prototype that emptied them at that point measured −33.5 and
−34.7 MB of peak RSS on two binaries; the CL measured −34.0 MB (the runs span
88,476 to 90,024 KiB; its own earlier measurement was 89,396). The cache now
holds a `FrozenGraph` (`runtime.rs:577`, from `EGraph::freeze` at `726`): the
node arena, union-find, member lists, live set, facts and units, which is
everything extraction reads. It has no method that takes `&mut self`, so
nothing can add to a cached graph.

## 3. Byte identity

### 3.1 What is preserved, and how

| must hold exactly (inventory B §6.1) | how |
|---|---|
| ids dense, ascending, one per `add`'s memo miss; `num_classes()` = those misses | arena `len` |
| `add`: canonicalize, look up; a miss seeds facts, stamps, sets the live bit, pushes no worklist entry; a hit has no side effect; refusals before mutation | unchanged code over new storage |
| union: smaller id survives; parent then child; fact merges; stamp; one worklist push of the parent; journal event only on a real merge | append or splice (§2.2) |
| rebuild: LIFO; `queued` cleared by the popped slot; stale pops count against the budget; mid-item unions immediate; law 8's order; refused node retired; duplicates kept | §2.3 |
| laws 2, 4, 5, 6: memo only grows; refusal as stated; no use index; stored names rewritten | unchanged code, nothing added |
| scan: live classes ascending, members in order, commits in queue order | §2.5 visits the same classes in the same order |
| extraction: candidates in member order; ties keep the first | the snapshot numbers the same walk |

### 3.2 Gates

Every CL passes all of these before it is submitted.

- **Gate 1.** `cargo run --release -q -p pixelflow-codegen --example byte_probe`
  equals `/home/user/scratch/egraph/speed/byteprobe-base.txt` (sha256
  `fab7d3b7…3181885`, 107 lines): synthetic kernels, no font program.
- **Gate 1b.** `zz_prof_all_cold` (`/home/user/scratch/egraph/cl7/probes-append.rs`)
  prints, per program, `KernelKey::of(...).bits()` of what
  `optimize_runtime_arena` returns and an `fnv1a64` over its node sequence
  (the key does not see the arena's push order), equal to a base recorded
  at `2a99d933`. It is the only gate that sees extraction on the font path:
  gate 2 cannot, and the suite's font tests are ceilings, not pins.
- **Gate 2.** The per-program telemetry tuple (applications, unions,
  classes at stop, live, iterations, stop reason) equals a base re-taken at
  `2a99d933` with `saturation-telemetry` and kept in the ledger (profile §6
  was taken at `7cd96a32`).
- **Gate 3.** During development, a lock-step shadow: old and new store fed
  the same `add`/`union`/`rebuild` stream and compared after every call on
  each member as an (id, `ENode` value) pair in order, `find`, facts,
  `last_changed`, the stamp, `queued`, the live set, the worklist, the memo
  and `refused_const_unions` in order, plus membership; deleted before
  submit (`docs/process/2026-10-13-tests-check-the-as-if-contract.md`).
- **Gate 4.** Each crate the CL touches, tested per crate (`cargo test -p
  <crate>`, with `--all-features` where a feature gates changed code; never
  `--workspace` locally), and `cargo clippy --workspace --all-targets
  --all-features -- -D warnings` as CI runs it: `provenance-journal` and
  `saturation-telemetry` compile code CL 1 and CL 3b change. Its assertions
  change only as §3.3 says.
- **Gate 5, measured, never asserted**, in
  `/home/user/scratch/egraph/speed/ledger.md`: callgrind Ir of the 29-program
  cold compile (`measure.sh`), saturation and extraction Ir, peak RSS,
  native wall time (never decisive; the VM drifts ~30%). Identical binaries
  measured 7,520,771,444 and 7,517,175,359 Ir, so |ΔIr| under 10 M is zero
  and each CL is measured against its own re-measured start. **CL 3b and
  CL 4 are decided by a metric fixed now**: callgrind `--cache-sim=yes`,
  `Ir + 10·(L1 misses) + 100·(LL misses)` inclusive of `Saturation::saturate`
  and `extract_dag_over`, which must fall by at least 1% or the previous
  layout stands.

One-time probes on CL 3b, for the research problem: classes holding two
equal canonical members after a full rebuild, and retired nodes.

### 3.3 Tests

CL 0 pins law 8 through the public `nodes()`: `union` gives parent then
child; a rebuild item whose class survives a mid-item union gives
`[absorbed] ++ [kept]`; one whose class loses gives the survivor's prior
members, then any it absorbs later in the item, then `[kept]`. That is
`nodes()`'s documented contract, not a layout pin. CL 4's review adds the fourth
case, which a reading of §2.2 as "an empty child splices nothing" does not
mention: an item that survives one mid-item union and then loses a later one hands
the survivor everything it absorbed while it stood, so the order is the survivor's
prior members, then the item's absorbed ones, then whatever the survivor absorbs
after the loss, then `[kept]`
(`a_rebuild_item_that_survives_then_loses_gives_prior_then_absorbed_then_kept`).

Layout tests become the law they stood for. The rebuild regressions
(`graph.rs:1953`, `2065`) become: every id ever added is a member of
`find(id)` or retired, and listed by exactly one class. (A second `rebuild`
changing nothing was in this sentence and is not: after a rebuild that ran the
worklist dry it pops nothing, so it cannot fail. The fixpoint worth stating is
the one congruence has not reached, law 5.)
`eg.classes.len()` (`graph.rs:2712`, `2720`) becomes `num_classes()`. Slice
patterns (`insert.rs:363, 417`, `extraction_gap.rs:184`,
`tests/root_shapes.rs:240`) and `[0]` (`template.rs` tests) read the
iterator; `first()` in the `fold_rules` tests becomes `next()`, "the first
member". The offline probes in `runtime.rs:1215+` read `nodes(i)`.

## 4. Consequences, followed to closure

Deleted: `EClass`, its `tags`, the lockstep `debug_assert`s (`graph.rs:523`,
`2122`) and the comment asking for them; `next_enode_id` and the telemetry
assertion that it equals `classes.len()` (`runner.rs:1196-1204`), with
`growth.rs`'s argument for it; the per-class node and tag vectors (CL 3b) and
member vectors (CL 4), with `union`'s reallocations; `node_for_tag`'s O(class)
and `children_of`'s O(all classes) searches; `track_liveness`'s call sites
and the drift checks; the "orphan" bug class (`graph.rs:2065`), since every
write-back targets a `Root`; production positional reads in extraction; the
class filter's per-class `admits` scan and two budget checks per class.

Ordinary: the journal's `origins: HashMap<ENodeId, Origin>` is keyed by a
dense index and may become a `Vec`.

Not added: a parents (use) index. It is not one more column: uses belong to
classes, so a union must merge them, and uses naming a retired id would be
lost through member lists. It would also close the congruence gap and change
which classes merge, so it is research (§7).

## 5. Public changes

`EGraph::nodes` and `EGraph::tags` return iterators, not slices (CL 1;
`pixelflow-pipeline` bins read them). `RootShapes::Kinds` carries an `OpSet`
(CL 2). Nothing else public changes type; §8 lists what would.

## 6. The CL sequence

Independent, low-risk wins first, then the interface, then the
representation, so each representation CL starts from a clean measurement.
CL 3a precedes CL 2 so CL 2's columns go into the type. CL 1 moves every
read of `classes` and `next_enode_id` outside `graph.rs` behind accessors
(`runner.rs:533`, `535`, `913`, `931`, seven `classes.len()`, `1143`,
`1157`, `1193-1202`, the `runtime.rs` probes), so CL 3b and CL 4 touch
`graph.rs`, the snapshot and §2.1's sites. Lines are estimates.

| CL | what | files | lines −/+ | expected Ir |
|---|---|---|---|---|
| 0 | S1 (`1be37dc5`); the order pin (§3.3); gate 1b's base | `graph.rs` | −20 / +100 | measured −0.22 G |
| 3a | Per-id columns behind one type; `Root` (§2.4) | `graph.rs`, `graph/runner.rs` | −60 / +100 | ~0 |
| 2 | `OpSet`; per-kind class index; hoisted checks (§2.5) | `pixelflow-ir/src/kind.rs`, `rewrite.rs`, `graph/runner.rs`, `graph.rs`, `live.rs` | −40 / +170 | −1.1 to −1.2 G |
| 5a | Split `union_below` into `cost_below` and `members_below` (the price path pushes members nothing reads) | `extract.rs` | −20 / +40 | 5a-c: −0.3 to −0.5 G |
| 5b | A one-child row reads its child's settled total; one-child price calls are counted first | `extract.rs` | −10 / +25 | |
| 5c | Carry `(own, var)` in `Settlement.best` instead of recomputing at settle | `extract.rs` | −30 / +25 | |
| 5d | Price a class from its largest child's settled total; it reorders sparse reach members, so its description argues byte identity and gate 1b decides | `extract.rs` | −30 / +60 | −0.4 to −0.7 G |
| 6 | Saturation scratch: the predictor's `pending` map, `sweep_rule`'s `updates`, `ConstantFold`'s argument vector stop allocating per call | `graph.rs`, `graph/runner.rs`, `algebra.rs` | −40 / +50 | −0.2 G |
| 1 | Class reads are opaque `impl Iterator + Clone` (no `len`, `rev` or `as_slice` to depend on); field reads behind accessors | `math/*`, `egraph/{fold_rules,derivative,template,candidate,witness,labeler,extract,and_or,insert,graph}.rs`, `graph/runner.rs`, `extraction_gap.rs`, `tests/root_shapes.rs`, `runtime.rs`, 6 pipeline bins | −110 / +100 | ~0 |
| 3b | The node arena; tags are ids; rebuild over member ids; liveness in the store; four comment fixes (§2.1, 2.3, 2.4) | `graph.rs`, `graph/runner.rs`, `provenance.rs`, `growth.rs`, `candidate.rs`, `extract.rs`, `labeler.rs`, `runtime.rs` probes | −350 / +250 | gate 5 decides |
| 3c | `node_of_slot`; `Pick` holds a node id; the membership check (§2.6) | `extract.rs`, `and_or.rs` | −40 / +80 | ~0 |
| 4 | Membership is an intrusive list; union is a splice (§2.2) | `graph.rs` | −80 / +120 | gate 5 decides |
| 7 | Freezing drops saturation-only state (memo, worklist, `queued`, `last_changed`, kind index); the frozen value has no `add` | `graph.rs`, `graph/columns.rs`, `extract.rs`, `and_or.rs`, `candidate.rs`, `witness.rs`, `optimizer.rs`, `runtime.rs`, `egraph/mod.rs`; small edits in four more | −20 / +80 estimated, −251 / +463 over 14 files as committed (extraction's crate-private signatures follow the frozen type: `/home/user/scratch/egraph/arena/spec-E.md` 3.10) | memory (§2.8): −34.0 MB measured, Ir zero |

Each CL is byte-identical by §3.2 and carries its ledger row; the
expected-Ir column is the reviews' estimate
(`/home/user/scratch/egraph/speed/review-*.md`) and decides nothing. CL 2's
identity rests on two conditions stated in its description: the hoisted
checks run only on a non-empty `live`, and production passes no deadline.
CL 7's rested on nothing reading the dropped state after `into_graph()`, which
now holds by type: the cached value is a `Frozen<FrozenGraph>`, which has
neither those fields nor a method that reaches them, so it holds for every
input and not only the 29 programs. Before the change a probe that emptied
the memo, the worklist and the four columns at the freeze point left gates
1b, 2 and 2x identical on the 29 programs. The one control-flow change is
`extract_over`'s reranker branch, now an assertion: a reranker scores an
`Extraction`, which lends an `&EGraph`, and the runtime tier sets none.

**Landed.** On the cold compile of the 29 font programs, callgrind Ir, with
gates 1, 1b, 2 and (from group B) 2x identical at every step:

| group | CLs | Ir | saturate | extraction | cache-sim sum | RSS |
|---|---|--:|--:|--:|--:|--:|
| A (#1416) | 0, 3a, 2 | 7,520 M → 6,122 M (−18.6%) | 3,817 M → 2,432 M | ~1,648 M | 6,361 M → 4,672 M | 133.3 → 137.5 MB |
| B (#1418) | 5a–d, 6 | 6,096 M → 5,383 M (−11.7%) | 2,408 M → 2,261 M | 1,647 M → 1,074 M (−34.8%) | 4,646 M → 3,889 M | 137.7 → 137.6 MB |
| C (#1420) | 1, 3b | 5,383 M → 5,210 M (−3.2%) | 2,261 M → 2,093 M | 1,074 M → 1,082 M | 3,903 M → 3,727 M (−4.5%, one host: see below) | 137.6 → 126.7 MB |
| D (#1422) | 3c, 4 | 5,210 M → 5,103 M (−2.1%) | 2,093 M → 1,950 M | 1,082 M → 1,083 M | 3,727 M → 3,528 M (−5.4%) | 126.7 → 123.7 MB |
| E (#1423) | 7 | 5,103 M → 5,097 M (−0.1%, zero) | 1,950 M → 1,952 M | 1,083 M → 1,082 M | 3,528 M → 3,512 M (−0.4%) | 123.7 → 89.7 MB (−27.5%) |
| **plan total** | all 13 | 7,520 M → 5,097 M (−32.2%) | 3,817 M → 1,952 M (−48.9%) | 1,650 M → 1,082 M (−34.4%, not like-for-like: see below) | 6,361 M → 3,889 M on one geometry (−38.9%), 3,903 M → 3,512 M on the other (−10.0%) | 133.3 → 89.7 MB (−32.7%) |

Against the estimates: 5a alone took −470 M of extraction, more than the
−0.3 to −0.5 G expected of 5a–c; 5b measured zero (one-child rows are 2.9%
of price calls) and 5d's general path absorbed it; 5d took −68 M, a tenth of
its estimate, because the largest child's set is dense in only 5.6% of wide
rows and a sparse one must still be stamped (60% of what pricing now walks;
answering membership in a sparse set without stamping is a representation
change, not in this plan). CL 6 took −142 M of the −0.2 G expected.

Group C's rows are against group B's end (`f30ef3b`, the final-B2 measurement),
except the cache-sim sum. The machine that measured CL 3b simulates a different
cache (D1 32 KiB 8-way and LL 34 MiB 17-way, where groups A and B and CL 1 ran
on D1 48 KiB 12-way and LL 264 MiB 33-way), so `f30ef3b` was measured again on
it: 5,383,588,991 Ir, which equals final-B2's within 1 M, and a cache-sim sum of
3,902,980,082 where group B's end read 3,889 M on the other geometry. CL 1
took −24 M Ir (−0.45%) where about zero was expected.

**CL 3b's decision** (the cache-sim sum falls by at least 1%, or the previous
layout stands): it passes on both readings. Against CL 1's row it is
3,865,007,883 → 3,727,192,993, −3.57%. Against a base on the same geometry, the
first half of 3b (`feb82335`, CL 1's production code) re-measured at
3,880,467,773, it is −3.95%, with Ir −147.9 M (−2.76%), saturate −148.7 M
(−6.64%), extraction +10.2 M (+0.95%: one more dependent load per member),
L1 misses +1.3% and LL misses −13.4%. A counterfactual tree that deletes only
the always-on drift check, which the member store makes unrepresentable (§2.4),
measures −94.3 M (−2.43% of the base) by itself, so the arena and the member
store are the other −58.9 M (−1.52%). Peak RSS fell 10.5 MB where §2.8
estimated about 15 MB.

Group D's rows are against final-C, the end of group C. Both ran on a machine that
simulates the same cache (I1 and D1 32 KiB 8-way, LL 34 MiB 17-way: checked before the
first edit and in each row's recorded geometry), so final-C's cache-sim sum is
comparable and nothing was re-taken. CL 3c took Ir +2.8 M (zero, under the 10 M floor),
extraction −7.9 M (an index into `node_of_slot` where `node_at` walked to a root and
loaded from a slice), the sum −14.3 M (−0.38%) and peak RSS +3.9 MB (`node_of_slot` is
8 bytes per slot and lives with the snapshot; §2.8 estimated up to +3.6 MB). It is not
decided by the cache rule; its sum is CL 4's base. Group D as a whole: Ir −107.3 M
(−2.06%), saturate −142.5 M (−6.81%), extraction +0.8 M (zero), the sum −199.6 M
(−5.36%), peak RSS −3.0 MB; against the plan's base row Ir is −32.1% and saturate
−48.9%.

**CL 4's decision** (the cache-sim sum falls by at least 1%, or the previous layout
stands): it passes. Against CL 3c's row, on the same geometry, the sum is
3,712,851,651 → 3,527,575,057, −4.99% (the rule needed 3,675,723,134 or less), with Ir
−110.1 M (−2.11%), saturate −143.1 M (−6.84%), extraction +8.7 M (+0.81%), L1 misses
−9.9% and LL misses −7.5%; peak RSS fell 6.7 MB where §2.8 estimated about 5 MB. The Ir
carries it, not the load chain §2.2 bet on: the sum's change is −134.4 M of Ir (73%),
−40.9 M of L1 misses (22%) and −9.9 M of LL misses (5%), and the Ir is the allocator and
the splice. `union` no longer extends a vector per merge and `add` no longer allocates
one for a one-node class (`realloc` −92.5 M Ir, `free` −21.4 M, `EGraph::union`
−133.4 M, `EGraph::add` −54.3 M, `rebuild_budgeted` −33.5 M), while the rule scan, which
walks a list where it walked a slice, costs +67.0 M Ir and saves 2.7 M of L1 misses
(+42.1 M on the metric). The walk is a cost the allocator's saving repays more than three
times over, not the gain.

During development a lock-step shadow of the CL 3c store was compared against the new
one after every `add`, `union` and `rebuild_budgeted` call and deleted before submit;
two deliberate mutations of the new store (`absorb` putting the child's members first,
`write_back` overwriting where it should splice) both failed it. The fourth case of §3.3,
an item that survives one union and then loses a later one, was pinned only by that
shadow until CL 4's review added a test; a mutant of `absorb` that skips a
child whose list was set mid-item fails it and passes the three tests before it.

**Group E** (CL 7) is against final-D, the end of group D. Both ran on a machine that
simulates the same cache (I1 and D1 32 KiB 8-way, LL 34 MiB 17-way, in each row's
recorded geometry), so final-D's sum is comparable and nothing was re-taken. CL 7
changes no instruction on the font path: Ir −5.9 M (zero, under the 10 M floor),
saturate +1.6 M, extraction −1.5 M, the sum −15.7 M (−0.44%, L1 and LL misses
unchanged to the digit between CL 7's own measurement and the final one), and peak
RSS −34.0 MB, which is the figure of merit (§2.8). Against the plan's base row Ir is
−32.2%, saturate −48.9% and peak RSS −32.7%. The extraction figures differ in scope:
the base's is `extract_dag_over` inclusive, the last rows' `Optimizer::extract_over`
inclusive (the caller: the same work plus its own frame), so only the total compares
across the CL 3a boundary. The cache-sim sum has no single figure from the base to the
end, because the machine changed once: the base geometry (D1 48 KiB 12-way, LL 264 MiB
33-way) ran groups A and B and CL 1, where the sum fell 6,361 M → 3,889 M (final-B), and
every later row ran on D1 32 KiB 8-way and LL 34 MiB 17-way, where group B's end
re-measured at 3,903 M and group E ends at 3,512 M. Rows are comparable only within one
geometry.

Next: nothing in this plan; its CLs are done. §7 and the research problem
(`docs/research-problems/2026-10-15-the-egraph.md`) hold what remains, and §8 is JP's.

## 7. What this does not address

Each changes what saturation or extraction does, hence the emitted kernels;
none is a representation change. They are open problems in
`docs/research-problems/2026-10-15-the-egraph.md`.

- **The budget.** 28 of 29 programs stop on the class cap (8× the inserted
  classes) mid-way through the second sweep; nothing reaches a fixed point.
  A faster store spends the same budget sooner.
- **Units.** Compiling each glyph program unit by unit (JP's decision 3 on
  compile time) changes what is inserted and so every count.
- **Matching.** Pattern-compiled rules, relational e-matching and semi-naive
  evaluation find the same matches in a different order, and order decides
  what lands before the cap and which node a tie keeps.
- **Congruence, duplicates, stored names.** A parents index (issue #1106),
  deduplicating rebuild, re-keying the memo, union by size, reading stored
  operands through `find`: each is a different optimizer, with budgets and
  pins re-derived in its own measured, re-baselined change.
- **Extraction's objective.** The tree/DAG gap and the schedule cost
  (`docs/compiler/2026-09-01-schedule-cost-model-denotation.md`).
- **The host kernel build** (19% of the cold run, 36% of it in the
  allocator), which is outside the e-graph.

## 8. Decisions for JP

1. **Positions in public APIs.** Keep `choices: Vec<Option<usize>>` (this
   plan), or make a choice a node id: the labeler, `witness.rs` and nine
   pipeline bins change type and `settlement_order_is_pinned` is
   re-recorded.
2. **(class, tag) pairs.** A tag determines its class, so the class beside
   it is redundant and can disagree: `node_for_tag(id, t)`,
   `RewriteTarget.class_id` (`runner.rs:33-48`), `derivation_ancestors*`'s
   pairs, the bins' `class_of_tag` maps. This plan keeps them and checks
   agreement (§2.1); dropping the class is a public change.
3. **Reserving the store to the class cap** at saturation start (about
   80 M Ir). 28 of 29 programs stop at the cap, so on this workload it
   removes the 1.44× doubling slack; only a graph that quiesces early keeps
   unused capacity.
4. **`EGraph: Clone`** is derived and public (`graph.rs:38`). Only tests and
   an offline probe use it; dropping it removes a cost the arena would
   otherwise have to keep cheap.
5. **The facts are saturation-only too.** `const_fact` and `var_fact` are read
   only by `add` (the seed), `union` (the refusal and the merge) and
   `variance()` (the fold rules' side conditions), so nothing reads them after
   saturation, and the cached `FrozenGraph` still holds them. Moving both to
   `SaturationColumns` is a two-field change and measured another −7.8 MB of
   peak RSS (89,132 → 81,312 KiB on CL 7's prototype, gate 1b identical with
   them emptied). The plan named the set as the memo, worklist, `queued`,
   `last_changed` and the kind index, so CL 7 kept them.
