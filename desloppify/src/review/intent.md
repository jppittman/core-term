# desloppify/src/review

**Is.** The review pipeline of a rule-driven code reviewer: a function from
(rules, files) to a `Report` in five stages, plan, decide, explain, identify,
synthesize (desloppify/README.md, "How a review runs"). Every model reply is a
value of a closed schema or a failure, never prose to parse. Small models make
many calls; the one frontier call, the synthesis, sees everything.

## Owns
- The plan: which rules ask about which unit of which file, and which share a
  call (same unit and context; a crate-wide rule gets one call per crate).
- The cascade: System One first, a model at the rule's decide level below
  `min_confidence`, one level up per `unsure` to the explain level, an
  explanation only for a unit decided as a violation.
- What each call is shown and its reply's schema (`reply.rs`).
- What a module is on disk (`module.rs`): root files, the module a path
  belongs to, a file's root outline and `intent.md`, an outline to a budget.
- Naming (`identify.rs`) and the synthesis (`synthesize.rs`): small calls
  name and describe the things a module's findings are about; one Frontier
  call diagnoses them, finds the roots and writes the review.
- Concurrency and order: every unit, rule and module is asked at once; the
  findings are in one total order, so the same answers give the same prompt.

## Never
- No backend is named. A model is reached only through `Ask` at a
  `ModelLevel`, System One only through `Decide`; no `Provider`, CLI, key
  or URL appears here.
- No rule's content is known: a rule is its id, question, outcomes, skills
  and context; no id or outcome name is matched.
- No call is made without a `schema`, and no finding is kept under an outcome
  that is not its rule's violation.
- The describe step never sees the code: its prompt is the name and context.
- A failed call never stops the others and is never silent: it is one entry
  in `Report::failures`, and the lead review is told how many failed.
- The module root is context only: a root file is shown alone.

## Contract
`mod.rs` is the contract and declares the stage files private. Entries:
`plan`, `review`, `identify`, `synthesize`; the data they pass: `Call`,
`Reviewers`, `Finding`, `Report`, `Thing`, `Identified`, `Diagnosis`, `Root`,
`Synthesis`. Stage files are `pub(super)`; `Source` is `pub(crate)`.
`module::root_outline` is the context `module_root` names: the outline of the
`mod.rs`, `lib.rs` or `main.rs` in the file's own directory, or the
2018-layout `<dir>.rs`; `module::intent` is what `module_intent` adds, the
nearest `intent.md` within the crate.

## Enforced by
- `desloppify/tests/transcript.rs`: a scripted review held to `tests/golden/`;
  a change to any prompt, schema, level or reported datum is a golden diff.
- `desloppify/tests/review.rs`, through the production API with a scripted
  `Ask` and `Decide`: `a_directory_without_a_root_file_shows_its_files_no_module_root`,
  `a_module_root_rule_shows_the_root_outline_as_context_and_a_root_file_alone`,
  `the_namer_sees_the_root_outline_first_then_the_files_with_the_most_symptoms`,
  `a_finding_under_an_outcome_that_is_not_a_violation_is_a_failure`,
  `converging_symptoms_are_named_by_small_models_and_described_without_the_code`,
  `the_synthesis_sees_the_things_and_their_outlines_and_its_numbers_become_indexes`,
  `every_rule_about_a_unit_is_asked_at_once`, `every_converging_module_is_named_at_once`
  (the last two fail if a stage serialises its asks). CI runs all
  (`.github/workflows/rust.yaml`).
- Types: `Ask` and `Decide` are the only way out; `Surroundings` is a closed
  enum. Rules `mod-rs-is-a-contract`, `interface-lives-in-mod-rs` apply here.

## Behind
- `module_intent` walks up to the first `Cargo.toml`, a file outside any
  crate to the filesystem root; no rule asks for more, so nothing pins the stop.
- Two walks, not one: `module::root_outline` looks only in the file's own
  directory, `module::containing` (where its findings cluster) climbs to the
  nearest directory with a root file; they differ for a rootless directory
  and a 2018 module. `module::shape` drops only neighbouring duplicates, so a
  root with symptoms but not the most is outlined twice, and journals key on
  that prompt. All pinned by the golden; each fix is its own change.
- A crate-wide rule never shares a call and gets no root or intent whatever
  its `context` (rule/intent.md, Behind).

## Decided in
`desloppify/README.md` ("How a review runs", "Layout"); §4 of
`docs/process/2026-10-15-intent-lives-with-the-module.md`.
