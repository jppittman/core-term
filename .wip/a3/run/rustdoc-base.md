# A3 rustdoc base (ruling C4)

Recorded at code base `3bfdaded` (HEAD carries only `.wip/` changes on top of it), host
toolchain `cargo 1.97.0` (`rust-toolchain.toml`: stable), default features.

## The gate

- `pixelflow-codegen`: `RUSTDOCFLAGS="-D warnings" cargo doc -p pixelflow-codegen --no-deps --document-private-items`
  **passes at base**, zero warnings (raw output: `rustdoc-base.pixelflow-codegen.raw.txt`).
- `pixelflow-ir`: 38 rustdoc warnings at base, 35 distinct (file, message) keys.
- `pixelflow-search`: 28 rustdoc warnings at base, 27 distinct keys.

For `pixelflow-ir` and `pixelflow-search` the head's warning set must be a subset of the base's:
no (file, message) key that is absent from the base file, and none that occurs more often than
the base file says. The gate is "no new warning", not "no warning"; a warning that goes away is
fine and is printed as `GONE`.

## The normalization (exact command)

A warning is a `warning: <message>` block of the output. Its key is the message plus the file
of the block's first `--> path:line:col` line, with the line and column dropped; a block with
no `-->` line (rustdoc prints none for some unresolved links, 16 of the 38 in `pixelflow-ir`, none in `pixelflow-search`)
has file `<no span>`. The `warning: <crate> (lib doc) generated N warnings` summary is not a
warning. Nothing else of a block is part of the key, so a warning that only moves within its file
does not change it.

The base files were produced by

```
cargo doc -p pixelflow-ir     --no-deps --document-private-items 2>&1 | python3 -I .wip/a3/gen/rustdoc_norm.py > .wip/a3/run/rustdoc-base.pixelflow-ir.txt
cargo doc -p pixelflow-search --no-deps --document-private-items 2>&1 | python3 -I .wip/a3/gen/rustdoc_norm.py > .wip/a3/run/rustdoc-base.pixelflow-search.txt
```

(from the repository root). Each line of a base file is `<count> TAB <file> TAB <message>`,
sorted by file then message. The counts sum to the crate's `generated N warnings` line (38 and
28).

## The comparison at head

```
cargo doc -p pixelflow-ir     --no-deps --document-private-items 2>&1 | python3 -I .wip/a3/gen/rustdoc_norm.py --compare .wip/a3/run/rustdoc-base.pixelflow-ir.txt
cargo doc -p pixelflow-search --no-deps --document-private-items 2>&1 | python3 -I .wip/a3/gen/rustdoc_norm.py --compare .wip/a3/run/rustdoc-base.pixelflow-search.txt
```

Each prints `warnings: base N, head M`, then `NEW k` with one line per new key, then `GONE j`.
It exits 1 when `k > 0`. Run at base it prints `NEW 0` and `GONE 0` for both crates (checked).
The pipeline's exit status is `PIPESTATUS[1]` in bash; the first stage is cargo's.

Do not compare line numbers or `git diff` the files: a doc edit that shifts a warning by a line
is not a new warning. A doc A3 rewrites that keeps its broken link keeps its key; one that
adds an unresolved or private link adds one (use explicit paths, `crate::fold::BoundRefusal::...`,
and no link from a `pub` item's doc to a private item: C5).

## Files

- `rustdoc-base.pixelflow-ir.txt`, `rustdoc-base.pixelflow-search.txt`: the normalized base sets.
- `rustdoc-base.pixelflow-ir.raw.txt`, `rustdoc-base.pixelflow-search.raw.txt`,
  `rustdoc-base.pixelflow-codegen.raw.txt`: cargo's full output at base, for reading a warning's
  source line.
- `../gen/rustdoc_norm.py`: the normalizer and the comparison.
