# Design Doc: The Op Numbering Is Private, and `marshal` Is How You Get Bytes

## Metadata
- **Author**: jppittman (with Claude)
- **Status**: Accepted — describes `pixelflow-ir/src/kind.rs` as built
- **Created**: 2026-08-16
- **Related**: `docs/STYLE.md`, `pixelflow-pipeline/src/training/corpus.rs`,
  `pixelflow-search/src/nnue/factored.rs`

---

## 0. Why this document exists

`OpKind` assigns every operation a number. That number is load-bearing in four
places and *interesting* in none of them, and every attempt to make it safe by
making it more visible has made it worse. This records where the numbering is
allowed to be seen, why the answer is "almost nowhere", and — because four
separate designs were tried and three were wrong — what each one got wrong, so
the next person does not re-derive them in order.

The current rule is one sentence:

> **The numbering is `pixelflow-ir`'s alone. Code that needs an op in bytes
> uses `OpKind::marshal`, and code that persists those bytes owes them a
> format version.**

---

## 1. The shape

| Item | Visibility | Job |
|---|---|---|
| `OpKind` variants, `name`, `arity`, … | `pub` | what an op *is* |
| `OpKind::all()` | `pub` | enumerate every op |
| `OpMap<T>`, `OpMap::LEN` | `pub` | a total per-op table |
| `OpKind::marshal` / `unmarshal`, `OpCode` | `pub` | an op as bytes, opaquely |
| `COUNT`, `ALL`, `index()`, `from_index()` | `pub(crate)` | the numbering itself |

`index()` is a subscript and nothing else: it exists so `OpMap` has somewhere
to put things. `OpCode` promises a round trip and a current width, and promises
nothing about which byte any op encodes to or that it is stable across
releases.

## 2. Why `marshal` exists at all

Not for persistence. It has four callers and only one writes a file:

| Caller | Bytes outlive the process? |
|---|---|
| `pixelflow-pipeline/src/training/corpus.rs` | **yes** — a real corpus |
| `pixelflow-codegen/src/jit_cache.rs` | no — `OnceLock<HashMap>` |
| `pixelflow-search/src/runtime.rs` | no — `OnceLock<HashMap>` |
| `pixelflow-pipeline/src/training/unified_backward.rs` (deleted 2026-09-01 with the extraction-head program) | no — in-memory edge record |

Three of four need an injective byte per op within one process and have no
opinion about which byte. `marshal` is therefore an **encapsulation boundary**,
not a serialization framework: its job is that `index()` can stay private.

## 3. What the leak looked like

`index()` returned `usize` — a subscript type — and each of those four callers
immediately wrote `as u8` into its own byte buffer. The signature described the
wrong job, so everyone who wanted "an op as bytes" reached for the numbering.

The consequence worth remembering is not the ugliness. It is that
`corpus.rs` ended up documenting `pixelflow-ir`'s discriminants as part of
*its own* file format, and bumping its `VERSION` for a renumbering that
happened in another crate. A private detail of one crate had become a
compatibility constraint of another, and nothing in the type system said so.

## 4. Four designs, three wrong

### 4.1 Dense discriminants, checked by a test (rejected)

Close the gaps, keep `index()` as `self as usize`, keep the bounds-check +
`transmute` in `from_index`, and add a test asserting the numbering is dense.

**Why it fails:** the test walks `0..COUNT` from the integer side, so it
structurally cannot see a variant added *past* `COUNT`. Measured: adding a
variant and leaving `COUNT` alone compiles clean, passes every test, and yields
an op that `all()` silently omits and `OpMap` panics on. A check that cannot
see the failure is not a check.

### 4.2 Four hand-written declarations, proved equal in a `const` block (rejected)

Write the discriminant, the `ALL` roster, and both directions of the match by
hand; prove at compile time that all four agree.

**Why it fails:** the same blind spot, one level up. The proof iterates the
roster, so a variant *missing from the roster* is exactly the one it never
visits. A proof can only check the variants something hands it.

**What it did buy, and what replaced it:** the `const` block survives, with one
job left — `op_table!` takes the *numbers* on faith, so a gap in the table
(`Const = 5`) is caught there and nowhere else. Duplicates need no help: two
entries sharing a number is `E0081` on the discriminants.

### 4.3 A derived encoding fingerprint (rejected)

Hash the op table into `OpKind::ENCODING_ID`, write it into both persisted
headers, refuse a mismatch. Renumbering then invalidates exactly the files
written under the old numbering, automatically, with no version to remember.

**Why it fails:** not incorrectness — proportion. It guarded a single format
that already carried a `VERSION`, at the cost of a const-eval FNV hash, eight
bytes in two headers, a coupling test, and a subtlety of its own (hashing the
table's literals rather than `marshal`'s output would leave a rewritten
`marshal`/`unmarshal` pair round-tripping happily with an unchanged id). It
also forced a header growth, which then required the version bumps it was
meant to replace — and getting *those* wrong silently corrupted an older
reader, which is the failure it existed to prevent.

The honest summary: it was machinery built to avoid typing a version bump, and
it cost more than the bump.

### 4.4 A version field (current)

`marshal`/`unmarshal` for encapsulation; `VERSION` and the weight magic for
persistence; nothing derived.

The obligation is written at each version constant, including the case that is
easy to miss: **changing the op encoding changes nothing in the consuming
file**, so it still parses and every op byte quietly names a different
operation. A stale corpus is cheap to replace and expensive to misread.

## 5. Consequences to keep in mind

- **Renumbering is allowed** and requires bumping `corpus::VERSION` (and, until
  its 2026-09-01 deletion, the `ExprNnue` weights magic — no weight file survives
  in the tree now). Nothing detects it for you; that is the accepted trade.
- **The numbering must not be pinned by a test.** A test asserting `Add == 2`
  recreates §3 by making an internal detail into a promise consumers can build
  on.
- `marshal` narrows to one byte, which is sound only while the op set fits in
  one. That is a `const` assert next to `marshal`, not a comment.
- Adding an op is a line in `op_table!` plus the arms the compiler demands.
  Roster, count and both directions follow from the table; if you find yourself
  updating any of them by hand, something has been un-generated.
