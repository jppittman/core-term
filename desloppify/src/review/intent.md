# desloppify/src/review

**Is.** The review pipeline of a rule-driven code reviewer: a function from
(rules, files) to a `Report` in five stages, plan, decide, explain, identify,
synthesize. Every model reply is a value of a closed schema or a failure;
small models make many calls, and the one frontier call sees everything.

## Owns
- The plan: which rules ask about which unit of which file, and which share a
  call (same unit and context; a crate-wide rule gets one call per crate).
- The cascade: System One first, a model at the rule's decide level below the
  session's minimum confidence, one level up per `unsure` to the explain
  level, an explanation only for a unit decided as a violation.
- What each call is shown and its reply's schema (`reply.rs`): every call is
  made from a `Step`, a level, a system prompt and a schema.
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
`mod.rs` is the contract and declares the stage files private. `plan` makes
`Call`s, each holding its rules; `Session` (who a review asks, how sure
System One must be) has the steps `review`, `identify`, `synthesize`; the data
they pass: `Report` (findings fixed when `review` returns, the rules that
asked), `Finding`, `Thing`, `Diagnosis`, `Root`, `Synthesis`. Stage files are
`pub(super)`; `Source` is `pub(crate)`. `module.rs` answers what a rule's
`module_root` and `module_intent` context name.

## Enforced by
- `desloppify/tests/transcript.rs`: a scripted review held to `tests/golden/`;
  a change to any prompt, schema, level or reported datum is a golden diff.
- `desloppify/tests/review.rs`, through the production API with a scripted
  `Ask` and `Decide`: `a_directory_without_a_root_file_shows_its_files_no_module_root`,
  `a_module_root_rule_shows_the_root_outline_as_context_and_a_root_file_alone`,
  `the_namer_sees_the_root_outline_first_then_the_files_with_the_most_symptoms`,
  `a_plan_is_reviewed_under_the_rules_it_was_planned_with`,
  `a_session_settles_at_or_above_its_minimum_confidence_and_asks_the_model_below`,
  `a_finding_under_an_outcome_that_is_not_a_violation_is_a_failure`,
  `converging_symptoms_are_named_by_small_models_and_described_without_the_code`,
  `the_synthesis_sees_the_things_and_their_outlines_and_its_numbers_become_indexes`,
  `a_naming_outside_its_schema_still_counts_its_tokens`, and three that fail if
  a stage serialises its asks: `every_rule_about_a_unit_is_asked_at_once`,
  `every_unit_is_reviewed_at_once`, `every_converging_module_is_named_at_once`.
  CI runs all (`.github/workflows/rust.yaml`).
- Types: `Ask` and `Decide` are the only way out; `Surroundings` is a closed
  enum. Rules `mod-rs-is-a-contract`, `interface-lives-in-mod-rs` apply here.

## Behind
- `module_intent` walks up to the first `Cargo.toml`, a file outside any
  crate to the filesystem root; no rule asks for more, so nothing pins the stop.
- Two walks, not one: `module::root_outline` looks only in the file's own
  directory, `module::containing` (where findings cluster) climbs to the
  nearest root; they differ for a rootless directory and a 2018 module.
  `module::shape` drops only neighbouring duplicates, so a root with symptoms
  but not the most is outlined twice. Pinned by the golden; each fix is its own change.
- A crate-wide rule never shares a call and gets no root or intent whatever
  its `context` (rule/intent.md, Behind).
- `Session::ask` and `::decide` are `pub` only so tests can read back the
  scripted agents they pass in (`Reviewers` had them); `main` never reads them.

## Decided in
`desloppify/README.md` ("How a review runs", "Layout"); §4 of
`docs/process/2026-10-15-intent-lives-with-the-module.md`.
