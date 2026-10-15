# desloppify/src/review

**Is.** The review pipeline of a rule-driven code reviewer: a function from
(rules, files) to a `Report` in five stages, plan, decide, explain, diagnose,
lead review (desloppify/README.md, "How a review runs"). Every model reply but
the last is a value of a closed schema or a failure, never prose to parse.

## Owns
- The plan: which rules ask about which unit of which file, and which share a
  call (same unit and context; a crate-wide rule gets one call per crate).
- The cascade: System One first, a model at the rule's decide level below
  `min_confidence`, one level up per `unsure` to the explain level, an
  explanation only for a unit decided as a violation.
- What each call is shown and its reply's schema (`reply.rs`).
- The diagnosis ladder and its thresholds (`diagnose.rs`): identify (Strong),
  describe without the code, translate onto the outline, root (Frontier).

## Never
- No backend is named. A model is reached only through `Ask` at a
  `ModelLevel`, System One only through `Decide`; no `Provider`, CLI, key or
  URL appears in this directory.
- No rule's content is known: a rule is its id, question, outcomes, skills
  and context, and no id or outcome name is matched here.
- No call but the lead review (`synthesize`) is made without a `schema`, and
  no finding is kept under an outcome that is not its rule's violation.
- The describe step never sees the code: its prompt is the thing's name and
  context.
- A failed call never stops the others and is never silent: it is one entry
  in `Report::failures`, and the lead review is told how many failed.
- The module root is context only: a root file is shown alone.

## Contract
`mod.rs` is the contract and declares the five stage files private. Entries:
`plan`, `review`, `diagnose`, `synthesize`; the data they pass: `Call`,
`Reviewers`, `Finding`, `Report`, `Diagnosis`, `Root`, `Diagnoses`. The
stage files are `pub(super)`; `Source` is `pub(crate)`. `plan::module_root`
is the context the rule field `module_root` names: the outline of the nearest
`mod.rs`, `lib.rs`, `main.rs`, or the 2018-layout `<dir>.rs` beside the directory;
`plan::module_intent` is what `module_intent` adds, the nearest `intent.md`
up the tree within the crate.

## Enforced by
- `desloppify/tests/review.rs`, through the production API with a scripted
  `Ask` and `Decide`: `a_module_root_rule_shows_the_root_outline_as_context_and_a_root_file_alone`,
  `a_finding_under_an_outcome_that_is_not_a_violation_is_a_failure`,
  `converging_symptoms_are_named_described_without_the_code_and_diagnosed`.
  CI runs it with the workspace (`cargo nextest run --workspace`,
  `.github/workflows/rust.yaml`).
- Types: `Ask` and `Decide` are the only way out; `Surroundings` is a closed
  enum. Rules `mod-rs-is-a-contract`, `interface-lives-in-mod-rs` apply here.

## Behind
- `module_intent` finds the nearest `intent.md` by walking up to the first
  `Cargo.toml`; a file outside any crate walks to the filesystem root. No
  rule has asked for an intent above the crate, so nothing pins the stop.
- Nothing in CI runs desloppify over the tree; only its own tests run.

## Decided in
`desloppify/README.md` ("How a review runs", "Layout"); §4 of
`docs/process/2026-10-15-intent-lives-with-the-module.md`.
