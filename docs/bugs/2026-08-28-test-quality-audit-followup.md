# Test quality control follow-up — 2026-08-28

Scope: scheduled continuation of
`docs/bugs/2026-08-26-test-quality-audit-followup.md`. That audit's backlog
item 1 was `pixelflow-search/src/egraph/cost.rs`, described as still open
per the 2026-08-08 audit (a partial mutants run that timed out on its own
slow `--lib` baseline before finishing).

## Backlog item 1 was stale: already closed by #1027

This pass started by mutation-testing `cost.rs` from scratch, using a
test-name filter (`cargo mutants ... -- --lib -- cost shallow_should`) to
work around the same slow-baseline problem the 2026-08-08 audit hit. That
produced a large set of new tests closing real gaps — `cost`/`set_cost`,
`node_op_cost`'s `Dwrt` guard, both `CostFunction` trait methods,
`save_toml`/`load_toml`/`load_or_default`/`from_map`/`to_map` — none of
which had direct tests under the branch's base commit.

Before pushing, `mcp__github__pull_request_read` reported the opened PR's
`mergeable_state` as `dirty`. Investigation found the branch had been built
from a **stale local `origin/main`**: an earlier `git fetch origin main
claude/zen-babbage-6a9p2k` failed outright (the second ref didn't exist),
which silently left the locally-cached `origin/main` 15 commits behind
without erroring loudly enough to notice, and `git checkout -B ... origin/main`
used that stale ref as the new branch's base.

One of those 15 missed commits was `80a2470` ("test: close cost.rs/x86.rs
mutation gaps... (2026-08-22 audit) (#1027)") — **the exact same backlog
item, already closed**, with materially better tests than the ones just
written here: `cost_model_accessors` and `persistence` modules covering
everything this pass found, plus cases this pass didn't think to check
(`load_toml_leaves_unmentioned_ops_at_zero_rather_than_the_latency_prior`,
malformed-line/malformed-value rejection). Its commit message specifically
calls out a flaw this pass's own `load_or_default` env-var test reproduced
blind: `unsafe { std::env::set_var(...) }` guarded only by "no other test
in this crate touches this variable" doesn't hold under a parallel test
harness, because `load_or_default` itself reads `HOME` too and nothing
stops an unrelated test elsewhere in the binary from calling
`load_or_default` while the var is set. #1027's version sidesteps this
entirely by exercising the override in a **spawned child process**, which
needs no `unsafe` at all.

The 2026-08-26 audit's backlog listed this item as still open two days
after #1027 (2026-08-27, per its commit date) closed it — the audits run
sequentially against whatever the doc says, not against a fresh mutants
run of the whole backlog each time, so a closed item that isn't struck from
the list stays "open" until someone notices. Recorded here so the next
pass doesn't repeat this: **item 1 (`pixelflow-search/src/egraph/cost.rs`)
is done, drop it from the backlog.**

Action taken: discarded this pass's redundant `cost.rs` changes entirely
(the branch was reset onto the real, current `main`, which already carries
#1027's version) rather than merge a worse duplicate over better work.

## Style fix that survived: `graph.rs`

`80a2470` did not touch `pixelflow-search/src/egraph/graph.rs`, which has
two tests exercising `CostModel::depth_cost`/`CostModel::shallow` — the
same public API #1027 covers from `cost.rs`'s own test file — with
bare-noun-phrase names and WHAT-comments restating what the asserts already
show:

- `depth_penalty_calculation` → `depth_cost_should_apply_linear_penalty_only_above_threshold`
- `shallow_cost_model` → `shallow_should_set_aggressive_depth_threshold_and_penalty`

Dropped the `// Test the hinge penalty function` / `// Shallow model should
have aggressive depth penalty` / `// Below threshold: no penalty` / `//
Above threshold: linear penalty` / `// Penalty kicks in after 16` comments
along with the renames — STYLE.md's comments section: prefer a name that
makes the comment redundant. This is a real, still-open fix (verified
against current `main`, not the stale base). No behavior change.

## Verified (against current `main`, commit `b4cc51f`)

- `cargo test -p pixelflow-search --lib -- cost shallow_should`: 25
  passed, 1 ignored (the persistence module's child-process helper test,
  by design), 0 failed.
- `cargo test -p pixelflow-search --lib`: 183 passed, 0 failed, 2 ignored.
- `cargo clippy -p pixelflow-search --lib --tests -- -D warnings`: clean.
- `cargo fmt -p pixelflow-search -- --check`: clean.

## Recommended next steps

Backlog carried forward from 2026-08-26, with item 1 struck (see above):

1. ~~`pixelflow-search/src/egraph/cost.rs`~~ — done, closed by #1027
   (2026-08-22 audit). Struck from the backlog.
2. `pixelflow-codegen/src/emit/*` (~11,700 lines across 8 files —
   `mod.rs` alone is ~228KB) — flagged since 2026-08-08 as never
   mutation-tested under its post-crate-split location. Still true, and
   too large for one pass; will need splitting across multiple audits,
   file by file, the way `spatial_bsp.rs` and `cost.rs` were handled.
3. `pixelflow-core/src/backend/x86.rs`'s `F32x8`/`F32x16`/`U32x8`/
   `U32x16`/`Mask8`/`Mask16` (AVX2/AVX-512) impls, and `arm.rs`'s NEON
   impls — never tested at the unit level at all under a build that
   actually activates those ISA levels (`xtask isa-matrix`).
4. `actor-scheduler/src/lib.rs`'s `backoff_unit_tests` — recommended for
   removal from the backlog by the 2026-08-26 audit (both halves of the
   original finding already resolved). No new information this pass;
   recommend actually dropping it now.

## Process note for future passes

Before starting substantive work, verify the branch's base actually
matches `origin/<default-branch>`'s current HEAD (`git fetch origin
<default-branch>` on its own, then `git merge-base --is-ancestor
origin/<default-branch> HEAD`) rather than trusting a combined
multi-ref fetch to have succeeded. A failed fetch of an unrelated ref can
silently leave the wrong branch as the checkout base, and `git log
--oneline -1 origin/<default-branch>` alone will not reveal this if the
stale cached value happens to print something plausible.
