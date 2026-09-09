# `ExprArena` on `Dag`: staging the port

**Status:** Proposed
**Date:** 2026-09-09
**Follows** the `dag` module (`pixelflow-ir/src/dag.rs`), which shipped
deliberately unused by `ExprArena`.

---

## 0. What this revisits

`dag.rs` landed with a stated reason for not porting `ExprArena` onto it:

> `ExprArena` is mutated and re-rooted throughout a kernel's whole
> compilation — `substitute_params`, `splice`, `substitute_vars_with` all take
> `&mut self`, push more nodes into the *same* growing arena, and hand back a
> new root to keep working against, interleaved with reads.

**That reason does not survive contact with the call sites.** It described the
*signatures* rather than the *usage*, and the usage is already pure. This
document records what the blockers actually are, which is a different and
smaller set, and stages the port around them.

---

## 1. The denotation

> `ExprArena` is a `Dag` of expression data, plus two identity tables, plus a
> choice of who may name a node.

Three parts, and only the third is contested:

| Part | `ExprArena` today | `Dag` |
|---|---|---|
| graph | `nodes: Vec<ExprNode>` + `nary_children: Vec<ExprId>` | `Dag<T>` |
| identity tables | `buffers`, `uniforms` (merged by `BufferIdentity`/`UniformIdentity`) | absent — belongs in the wrapper |
| who may name a node | `ExprId(pub u32)`, forgeable anywhere | `Node<'a, T>`, unforgeable, borrowed |

The first two are mechanical. The third is the whole question, and §5 splits
the plan on it.

---

## 2. The lifecycle objection was wrong

Every mutating method is already shaped as `old graph → new graph`:

| Method | Actual shape | Evidence |
|---|---|---|
| `relink` | **already pure** — `&self → (ExprArena, ExprId)` | `arena.rs:1222`, fresh `out` at `:1256` |
| `substitute_params` | one call site, old root dropped immediately | `pixelflow-compiler/src/emit.rs:97`, then `Kernel::from_parts` |
| `splice` | donor is `&ExprArena`; only the host grows | `kernel.rs:334,342,596,698` |
| `substitute_vars_with` | old root shadowed at every site | `kernel.rs:642,700`; `template.rs:308,443`; `oracle.rs:253` |

And `Kernel` is `Arc<KernelData>` (`kernel.rs:251-263`) — immutable once
wrapped — so **every combinator already clones the arena first**
(`map:326`, `combine:333`, `combine3:341`, `over:639`, `at:697`). "One arena,
dozens of sequential mutations" does not happen. Each op is: fresh arena,
one or two splices, done. That is a `Builder` verbatim.

The one deliberate accretion is `Kernel::sum` (`kernel.rs:589-600`), N splices
into one arena to avoid the O(n²) re-clone a naive fold would cost on a
glyph's thousands of leaves (doc at `:575-579`). It is **write-only until
done** — which is exactly what `Builder` supports.

Two shapes the port must keep:

1. **Two live roots in one graph.** `oracle.rs:253-254` and `template.rs:443-444,470-471`
   substitute `lhs` and `rhs` against one arena and keep both. This is
   already supported: `Builder::finish(&[Id])` takes a *slice* of entries and
   `Rooted::entries()` vends them back. Nothing to build.
2. **`Kernel::sum`'s N-splice fold**, above.

### 2.1 The garbage is a cost, not a contract

Nothing depends on unreachable nodes surviving. Four all-node scans exist, and
every one is an *identity fast-path* that garbage makes strictly worse:
`passes.rs:213,512`, `oracle_lowering.rs:53`, and `pixelflow-compiler/src/lib.rs:188-191`
(where `DwrtFree` declines to optimize if *any* `Dwrt` exists anywhere,
reachable or not). Their stated purpose is stability — *"keeps lowering a true
no-op"* (`passes.rs:209-212`) — not correctness. Dropping garbage makes all
four **more precise**. `retired_axis` (`arena.rs:383`) is already
reachability-scoped by design.

So a build-once-freeze port is not paying a tax here. It is collecting one.

---

## 3. The blockers that are real

### 3.1 `ExprNode::Nary(OpKind, u32, u16)` publishes raw offsets

67 sites match `ExprNode::Nary`; **40 bind the offsets**, 27 discard them.
The 40 split:

| Location | Count |
|---|---|
| `pixelflow-ir/src/arena.rs` | 13 |
| `pixelflow-search/src/nnue/mod.rs` | 6 |
| `pixelflow-pipeline/src/training/corpus.rs` | 5 |
| `pixelflow-ir/src/{variance,passes}.rs` | 8 |
| `pixelflow-compiler/src/emit.rs` | 2 |
| everything else (7 files) | 6 |

`ExprNode` is pinned `<= 16 bytes` (`arena.rs:213`), so `Nary` cannot carry
children inline — some indirection must remain. What can go is its
*publication*.

### 3.2 Three structs own an arena and index it in the same value

A lifetime-tied `Node<'a, T>` cannot be stored beside the arena it borrows:

- `nnue/mod.rs:254-263` — `BwdTrainingPairArena { arena, optimized: ExprId, unoptimized: ExprId }`
- `nnue/factored.rs:195-199` — `ArenaRuleTemplate { arena, lhs: Option<ExprId>, rhs: Option<ExprId> }`
- `egraph/template.rs:145-152` — `TemplateRewrite { arena: Arc<ExprArena>, lhs, rhs }`

**`Rooted<T>` is already the answer to this shape.** It stores entry
positions privately as `Vec<u32>` and vends `Node<'_, T>` handles on demand —
owned identity inside, borrowed identity at the API. All three become
`Rooted<ExprData>` with two entries. The self-reference dissolves; it does not
need an `Arc` or an index side-channel.

### 3.3 `runtime.rs:235-330` `canonical_key` is the hardest single site

Walks `nodes_raw()`, synthesizes `ExprId(idx as u32)` (`:260`), matches all
nine variants, reads the slab by raw offset (`:320`), and leans on both
`id.0` as a dense array index and the append-only "children precede parents"
invariant (`:228`). It is the only `nary_children_raw` caller in
pixelflow-search.

### 3.4 The corpus on-disk format embeds the encoding — and its guard is prose

`pixelflow-pipeline/src/training/corpus.rs` persists `PXCR` files whose `Nary`
record *is* `(op, start: u32, len: u16)` (write `~:380`, read `:523-528`), fed
straight back through `ExprArena::from_raw` (`:484`).

`corpus_identity()` (`:168`) folds `CorpusFormat::SCHEMA_IDENTITY` — a content
hash of the `SCHEMA` prose (`:91`) — with the live `OpKind` table. The design
is deliberate and already documented at `:80-84`: *"This text IS the version:
change what a field means here and `SCHEMA_IDENTITY` (its content hash)
changes with it, so a corpus written under the old meaning cannot silently be
read under the new one."* There is even a regression test for the hole where
prose alone was insufficient (`:844-851`, PR #1019).

**So the guard works — but only if the prose is edited.** The encoding is not
auto-derived from the code, so changing the on-disk `Nary` record without
touching that string leaves stale corpora passing the identity check. No
corpora are committed (`data/` is gitignored), but ~12 binaries read local
ones, regenerated via `regen_command()` (`:115`).

This is the one place in the port where the failure mode is quiet, and the
mitigation is a one-line discipline the file already asks for.

---

## 4. What is already insulated

Worth stating, because it is most of the surface:

- **arena → e-graph is fully generic.** `egraph/insert.rs:60` is
  `insert<I: Ir>(term, root, egraph, vocab)`, using only `project`/`Shape`;
  no `ExprId` in the file. `reachable_count<I: Ir>` (`:142`) likewise.
- **The `Ir` trait already is the node-by-node API this port wants**
  (`term.rs`, `term_arena.rs`). The three boundaries in §3 do not lack an
  abstraction; they *bypass* the one that exists.
- **The JIT cache key does not depend on offsets.** Child ids are
  dense-remapped before hashing (`jit_cache.rs:~240`); `start` is consumed only
  to find the slice and never enters the key. What it *does* require is the
  topological guarantee — which `Dag` gives by construction, since a child
  must exist before its parent can name it.
- **NNUE inference is already decoupled** — `nnue/guide/**` has zero
  `ExprNode` references; `bilinear.rs:165-175` walks via `kind()`/`children()`.
- `Ord` on `Ir::Ref` buys exactly two things, both memo keys in `insert.rs`
  (`:73` `BTreeMap`, `:143` `BTreeSet`). Nothing sorts a `Ref`.

### 4.1 A bug the port deletes for free

`nnue/mod.rs:996` — `remap_node`'s `Nary` arm rebuilds remapped children into
`_children` (`:977`, underscore-prefixed, unused) and then returns
`ExprNode::Nary(*op, *start, *len)` **with the original, unremapped offsets**,
under a comment conceding "Nary is extremely rare." A `Shape`-based rewrite
cannot express that mistake, because `Children::Many` hands over a slice
instead of a range to forward blindly.

---

## 5. The fork: does `ExprId` survive?

| | Keep `ExprId` (public index) | Retire it for `Node`/`Rooted` |
|---|---|---|
| refs in blast radius | ~800 (`ExprNode` only) | ~2150 (`ExprId` + `ExprNode`) |
| files | ~40 | ~110 |
| gets the unified representation | yes | yes |
| gets "consume it without knowing about the arena" | **no** | yes |
| §3.2 self-referential structs | untouched | become `Rooted` with 2 entries |

Per-crate reference counts, for sizing:

| Crate | `ExprId` | `ExprNode` |
|---|---|---|
| pixelflow-ir | 304 | 343 |
| pixelflow-search | 390 | 222 |
| pixelflow-pipeline (512 lib / 156 research bins) | 529 | 143 |
| pixelflow-codegen | 87 | 39 |
| pixelflow-compiler | 22 | 20 |
| pixelflow-graphics | 15 | 29 |
| pixelflow-core | 2 | 0 |

**Recommendation: stage it so the fork is deferred, not decided now.** Stages
A–C below unify the representation and are worth landing on their own terms;
Stage D is the fork, and by the time it is reachable most of its cost has
already been paid by A.

### 5.1 Consing is a third question, already deferred on the record

`Kernel::sum` carries a standing note (`kernel.rs:581-587`):

> DEFERRED (shared-store direction): the deeper fix is one hash-consed arena
> that all `Kernel`s index by `ExprId`, so composition interns instead of
> splicing (copies vanish, structural sharing is automatic). Not taken yet:
> it changes the `Kernel` representation and wants the same store P7–P9's
> discrete domains/typed fields will live in — land it there, deliberately,
> rather than as a silent representation swap.

Two things follow. First, this plan must **not** smuggle consing in — the note
asks for exactly the opposite, which is why Stage C says `push_unique`.

Second, **the benchmark in `benches/dag_vs_arena.rs` does not settle the
question, and should not be cited as if it did.** It measured `intern` against
a *bare push* (48–115×). The note's proposal is `intern` against a **`splice`,
which copies the donor's whole reachable subgraph**. Those are different
baselines, and on `Kernel::sum`'s thousands-of-leaves fold the copying one is
the expensive side. Whether consing wins there is unmeasured — and measuring
it wants real glyph kernels, not the synthetic shapes that bench uses.

---

## 6. Stages

### Stage A — route the bypassers through `Ir` *(no representation change)*

Convert the §3 offset-consumers to op+children access. Independently landable,
no behavior change, and it deletes most of the 40 offset-binding sites:

- `emit.rs:118-202` — emit a *sequence of builder calls* instead of a flat
  `vec![ExprNode::…]` + `from_raw`. The node vec is already in
  child-before-parent order and every child reference is already an index into
  it, so `ExprId(#child)` becomes `#child_ident` and the `Nary` arm becomes
  `push_nary(op, &[…])`. Keep the `Const` bit-pattern emission (`quote`'s
  `f32_suffixed` asserts `is_finite`, and non-finite constants are ordinary
  here) and the `Buffer`/`Uniform` expansion-time panics.
- `runtime.rs:235-330` — `kind()`/`children()`; keep the dense remap.
- `jit_cache.rs` `Nary` arm — `children()`; key bytes unchanged (verify by
  asserting an unchanged key on a fixture arena).
- `nnue/mod.rs` `remap_node`/`junkify_arena_pass` — `Shape`-based; fixes §4.1.
- `corpus.rs` — builder-call walk on read, `children()` on write. **Bump
  `CorpusFormat::SCHEMA` (`:101`) in the same commit** and confirm
  `corpus_identity()` changes, so stale files are rejected loudly.

### Stage B — make the offsets private

With Stage A landed, few consumers remain. Replace `Nary`'s `(u32, u16)` with
an opaque `Copy` range newtype (private fields, still 16-byte-clean), or drop
`nodes_raw()`/`nary_children_raw()` from the public API outright. This is the
commit after which the representation is swappable.

### Stage C — swap the storage

`ExprArena { dag: Dag<ExprData>, buffers, uniforms }`, with `ExprId` kept as
the public handle and translated at the boundary, and `ExprNode` reconstructed
on demand by `node()`. Two consequences to decide explicitly:

- **`ExprData::Const` must key on bit pattern** (`u32`), not `f32` — `f32` is
  neither `Eq` nor `Ord`, so `Builder`'s `Key` bound refuses it. This matches
  what the codebase already wants (`subtree_eq` compares constants bit-exactly,
  `arena.rs:1386`) and what both the cache key and the corpus format already
  do.
- **Use `push_unique` here; do not let this stage decide consing.** The
  representation swap should be semantics-preserving, and consing is a
  separate question the codebase has already thought about — see §5.1.

### Stage D — the fork (optional)

Retire `ExprId` from the public API; move consumers to `Node<'_, T>`; convert
the §3.2 structs to `Rooted` with two entries. This is the stage that buys
"consume the DAG without knowing about the arena" for `ExprArena` itself.

---

## 7. Verification

Per stage, not at the end:

- **A**: full workspace test suite; JIT cache key byte-identical on a fixture
  arena before/after; `corpus_identity()` demonstrably *changes* and a stale
  corpus is rejected with a regeneration message rather than misdecoded.
- **B**: compile-only — the point is that nothing outside `arena.rs` can name
  an offset.
- **C**: `pixelflow-ir` unit + integration tests; `cargo test --workspace`;
  `cargo bench -p pixelflow-ir --bench dag_vs_arena` to confirm no
  construction-path regression against the numbers recorded there;
  `xtask isa-matrix --smoke`.
- **D**: workspace tests; the three §3.2 structs are the acceptance criterion —
  if they express cleanly as `Rooted`, the model held.

Stage A is worth doing whether or not C and D ever happen: it removes a latent
bug (§4.1), makes four fast-paths more precise (§2.1), and routes five
bypassers through the abstraction that already exists.
