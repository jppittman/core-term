# The Reduce-site checklist (rulings B2, C10)

Every line of the workspace's Rust sources that spells `Reduce {` is a **site**. Each site is
classified in `classes.json`, and `checklist.py` fails when the two disagree. It exists so that
no code that walks a fold is left unread when a fold's bounds can be values (A3), and so that A4
finds the serializers it owes.

## Run it

```
python3 -I .wip/a3/gen/checklist.py          # from anywhere in the repository
python3 -I .wip/a3/gen/checklist.py --list   # also one line per site: class, file:line, enclosing item
```

Stdlib only; `-I` so nothing is imported from the current directory. It runs the spec's grep,

```
grep -rn 'Reduce {' --include=*.rs pixelflow-*/src pixelflow-*/examples pixelflow-*/tests pixelflow-pipeline/src/bin
```

from the repository root, drops the duplicate `(file, line)` pairs (the four
`pixelflow-pipeline/src/bin` files are matched by both `pixelflow-*/src` and the last
directory: 155 lines, 151 sites at `3bfdaded`), and prints the class counts, then

- `UNCLASSIFIED n`: hits no row covers (each with a ready-to-paste row stub);
- `STALE n`: rows no hit covers.

Exit status: 0 when both are 0, 1 otherwise, 2 when `classes.json` is malformed (unknown class,
missing field, empty reason, duplicate key, bad count). At `3bfdaded` it reports 0 and 0 with
the counts already-correct 39, changed-by-A3 19, left 13, tests 60, n/a 5, A4 15, A5 0.

`--root DIR` scans another tree (for trying the script on a copy); `--classes FILE` reads
another rows file.

## What a row is

`classes.json` is a list of rows:

| field | meaning |
|---|---|
| `file` | path from the repository root |
| `fn` | the enclosing item (below) |
| `text` | the hit line, trimmed of leading and trailing whitespace |
| `count` | how many hits share this exact `(file, fn, text)` (usually 1) |
| `class` | one of the classes below |
| `reason` | why, one line; required |
| `lines` | the line numbers when the row was written: for the reader only, never read by the script |

The key is `(file, fn, text)`. A line number is not in it, so adding or deleting code above a
site does not unclassify it. A site whose line is edited, or whose function is renamed or
moved, **is** a new key: the old row goes `STALE` and the new text goes `UNCLASSIFIED`, which
is the point: someone re-reads the site.

**The enclosing item** is found by matching braces over the file with comments, strings and char
literals masked out: the innermost `fn` whose body contains the line, by bare name (`transfer`,
not `Foo::transfer`); else the innermost `struct`/`enum`/`union`/`trait`/`mod`/`impl` body
(`enum ExprNode`, `impl Foo for Bar`); else `<top>` (a doc comment above a top-level item, such
as `Fold`'s denotation). Two sites with the same text in two methods of the same name in one file
share a key and a count; give them one class or split the text.

**Classes** (spec section 5):

- `already-correct`: takes every child, or answers rightly for a value bound without change.
- `changed-by-A3`: A3 edits the arm or its doc.
- `left`: it refuses or never meets a value bound, and stays so.
- `tests`: test code (a `#[cfg(test)]` module, or `tests/`).
- `n/a`: a definition or a doc.
- `A4`: the schedule, the key and the serializers, which refuse a value bound by name until A4.
- `A5`: none at `3bfdaded`; the compiler's `lower.rs` hits are all in its test module.

## Add, change or remove a row

1. Run the script. Its `UNCLASSIFIED` lines each carry a stub, for example
   `{"file": "pixelflow-ir/src/example.rs", "fn": "walk", "text": "ExprNode::Reduce { body, .. } => {", "count": 1, "class": "?", "reason": "?", "lines": [123]}`.
2. Paste it into `classes.json` (keep the list sorted by file, then first line, so a diff reads),
   and set `class` and `reason`. Read the site to choose the class; do not copy a neighbour's.
3. A site you rewrote: its old row now shows `STALE`. Edit that row in place (its `text` and
   `fn` are the new ones; update `class` if the answer changed, for instance `changed-by-A3`
   becomes `already-correct` once the arm is done) rather than adding a second row, so the
   history of the site stays in one place. A row whose site was deleted is removed.
4. A new test that matches is a `tests` row; an identical line repeated `n` times in one
   function is one row with `count: n`.
5. Re-run: it must end `UNCLASSIFIED 0` and `STALE 0`.

`classes.json` was seeded once from spec section 5's table by `seed_classes.py`, which joins each
table row's `file:line` to the hit at that line (all 151 matched, no table row without a hit, no
hit without a row) and refuses to overwrite an existing file. It is not meant to be re-run: after
the seed the file is edited by hand, as above.

## Rows A3 rewrites (known at base)

The spec's list (section 5) plus `arena.rs` `lowest_free_binder`, `extract.rs` `below` and
`insert.rs` `insert`: `variance.rs` `compute_arena_variance` (the two arms move into `transfer`),
`node.rs` `variance`, `arena.rs` `free_index`, `lowest_free_binder` and `substitute_vars_with`,
`extract.rs` `below` and `price`, `insert.rs` `insert` (twice) and `inserted_class_count`,
`cost.rs` `node_op_cost`, `fold_rules.rs` `rebuild` (twice), `node.rs` `enum ENode`, `reduce`
and `known_range`, `lower.rs` `arena_to_schedule` (the comment), `training/corpus.rs`
`write_node` (the message). They are keyed by their base text now; the implementer of each
updates the row in the same change. New production hits the spec predicts (`EGraph::add`'s
backstop in `graph.rs`, `KnownBound::apply` in `fold_rules.rs`, `literal_twin` in `extract.rs`,
the `free_index` fold arm with new text) and every new matching test get rows in the change that
adds them.

## Files

- `checklist.py`: the check.
- `classes.json`: the rows.
- `seed_classes.py`: the one-off seed from `.wip/a3/spec.md`.
- `rustdoc_norm.py`: the rustdoc normalizer and comparison (see `../run/rustdoc-base.md`).
