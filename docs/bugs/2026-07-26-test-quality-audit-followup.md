# Test quality control follow-up — 2026-07-26

Scope: scheduled continuation of `docs/bugs/2026-07-20-test-quality-audit.md`.
Re-checked that pass's "Recommended next steps" against the current tree,
re-ran `cargo-mutants` (installed fresh, `cargo install cargo-mutants
--locked`, v27.1.0 — not present in this environment) on the same
pixelflow-core algebra subset, and finished the previously-incomplete
core-term static audit.

## Findings: most of the prior pass's recommendations were already done

Items 1–4 of the 2026-07-20 doc's "Recommended next steps" turned out to
already be fixed in the tree (this is a shallow clone with truncated git
history, so the exact landing commit isn't visible, but the code and test
content match the recommendations exactly — e.g. `actor-scheduler`'s
`backoff_unit_tests` module already has the "Kills: ..." mutant-targeted
tests the prior pass called for, and `ops/trig.rs` / `ops/compare.rs` already
have correctness test modules for range reduction and the soft-comparison
combinators). Mutation testing on the same pixelflow-core algebra subset as
2026-07-20 (trig.rs + compare.rs, 129 mutants) confirms the improvement
empirically: **281 missed → 3 missed** on this file pair specifically
(the 2026-07-20 number was for the larger 375-mutant/5-file scope; this pass
only re-scanned trig.rs + compare.rs).

`actor-scheduler/src/kubelet.rs` (item 2, `ManagedPod`/`poll_interval`
private-field tests) is now moot: the module was removed entirely in a
refactor to a Mealy-transducer actor model
(`docs/designs/actor-scheduler-mealy-transducer.md`).

## Fixed this pass (branch `claude/nifty-maxwell-sa39pa`)

1. **`pixelflow-core/src/ops/compare.rs`** — of the 3 mutants still missed on
   trig.rs+compare.rs, one was real: `smoothstep_sigmoid`'s `diff / k`
   survived being mutated to `diff * k`. Every existing `SoftGt`/`SoftLt`
   test either saturates (large `|diff|` clamps `t` to 0/1 regardless of
   whether `k` divides or multiplies) or sits exactly at the boundary
   (`diff = 0`, where `/` and `*` agree trivially). Added
   `soft_gt_mid_transition_divides_diff_by_sharpness`: `diff = 1.0,
   sharpness = 2.0` gives an unclamped `t = 0.75` under the correct code and
   a clamped `t = 1.0` under the mutant — checked against the exact
   Hermite-smoothstep value (0.84375). Re-ran cargo-mutants after the fix:
   **2 missed / 119 caught / 8 unviable** — the mutant is now killed.
   The 2 remaining misses are `cheby_atan2` (trig.rs:168, 176): `atan_val *
   sign_y` / `PI * sign_y` mutated to `/`. `sign_y` is always exactly `+-1.0`
   (from a `select`), and division by `+-1.0` is exact in IEEE-754 with no
   rounding difference from multiplication — these are genuine equivalent
   mutants, not a coverage gap, matching the precedent set for `mask.rs`'s
   `all_false` in the 2026-07-20 pass.
2. **`core-term/src/term/emulator/input_handler.rs`** —
   `paste_text_action_bracketed_on` called the `pub(super)`
   `handle_set_mode` directly instead of going through `interpret_input`,
   exactly as its own `TODO` said to fix. Switched to
   `CsiCommand::SetModePrivate(DecModeConstant::BracketedPaste)` via
   `interpret_input`, the same pattern already used for this in
   `term/tests.rs:1601`.
3. **`core-term/src/io/event_monitor_actor/{writer,mod}.rs`** — writer.rs's
   `#[cfg(test)] mod tests` constructed the private `PtyWriter` via struct
   literal and called `handle_data`/`handle_control`/`handle_management`
   directly, bypassing the actor message-passing surface entirely (a
   `pub(super)` type is still "not the public API" in the sense the actor
   model cares about — CLAUDE.md's Control/Management/Data lane contract is
   only exercised via `handle.send()`). Two of the three tests
   (`resize_before_bind_is_coalesced`, `bound_writer_accepts_writes`) were
   redundant with existing `mod.rs` troupe-level coverage
   (`resize_before_bind_is_applied`, `troupe_round_trips_writes_through_child`).
   The third (`write_before_bind_is_flushed_at_bind`) had no public-surface
   equivalent — moved it to `mod.rs` as a new test using
   `writer.send(Message::Data(...))` before `troupe.spawn()`, then deleted
   writer.rs's entire private-field test module.
4. **`core-term/src/term/screen.rs`** — `create_test_screen_with_scrollback`
   sets the private `scrollback_limit` field directly; left the behavior
   as-is (fixing it requires a real API decision: `Screen::new` sources the
   limit from the global `CONFIG` static with no parameterized public
   constructor, and CLAUDE.md restricts changing internal API visibility
   without explicit permission — not something to do unreviewed). Did trim
   the surrounding comment block, which was ~10 lines of rambling,
   self-doubting "For now, let's assume..." narration about a since-resolved
   design question — a direct violation of docs/STYLE.md's own rules against
   historical/uncertain commentary.

All changed/added tests verified passing (`cargo test -p core-term --lib`:
436/436; `cargo test -p pixelflow-core --lib ops::compare::`: 10/10;
`cargo test -p actor-scheduler --lib`: 126/126) and `cargo check --workspace`
clean.

## Static audit: finished the incomplete core-term sweep from 2026-07-20

The prior pass's core-term audit didn't get to 5 files. Status:

- `term/emulator/mouse.rs` — compliant, tests only call
  `encode_mouse_event()`.
- `term/emulator/key_translator.rs` — compliant, tests only call
  `translate_key_input()`.
- `term/emulator/input_handler.rs` — **fixed above.**
- `term/layout.rs` — compliant; `Layout`'s fields are all `pub`, so
  struct-literal construction in tests is legitimate public-API use, not a
  private-field violation.
- `term/unicode.rs` — compliant (minor, not fixed): one test also reads
  `GLOBAL_LOCALE_INITIALIZER.get().is_some()`, a private static's
  populated-state rather than a struct field. Low severity, left as-is.

Also re-checked the three items from the 2026-07-20 doc previously left "not
fixed" for `writer.rs`/`terminal_app.rs`/`screen.rs` — see above.

## Remaining known items (judgment calls, not done here)

Both still require a human design decision, same as 2026-07-20 concluded:

1. **`pixelflow-graphics/src/spatial_bsp.rs`** — 19 tests still index the
   private `bsp.interiors[...]` array directly (no accessor on the array
   field itself, though `InteriorNode`'s own fields are now `pub`). Fixing
   needs either a test-only introspection API, a property-test rewrite over
   `eval()`, or an explicit documented rule-break.
2. **`core-term/src/terminal_app.rs`** — `create_test_app()` calls the
   private `TerminalApp::new_registered` instead of the public
   `spawn_terminal_app()`. This looks like an intentional seam rather than
   an oversight: `spawn_terminal_app()` does real thread/window/PTY I/O,
   which a fast deterministic unit test of e.g. resize-handling logic
   shouldn't have to pay for. Worth a human call on whether to add a
   lighter-weight public test constructor or accept this as documented.
