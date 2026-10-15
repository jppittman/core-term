# desloppify

**Is.** Rule-driven code review: a function from (rules, files) to a `Report`
(desloppify/README.md, not DENOTATION: this is not a pixelflow crate). A rule is
a typed decision; tree-sitter cuts the units, a cheap decider answers, a model
explains only the violations, and findings that converge on a module are
diagnosed as symptoms of a missing model.

## Owns
- The shipped rules (`rules/*.json`, 27) and the skills they pull in
  (`skills/<name>/SKILL.md`): what this repository asks a reviewer.
- The pipeline's shape: plan, decide, explain, diagnose, lead review;
  `main.rs` only wires it from flags.
- The exit status: failure on any finding or undecided unit.
- Its own layout: every module is a directory whose `mod.rs` is the contract,
  every other file is `pub(super)`, all tests are in `tests/` and drive the
  API `main` uses.

## Never
- tree-sitter never judges. It is imported only by `language/`, `snippet/`
  and `rule/`, and finds units; every verdict is a decider's or a model's.
- No model id is spelled outside `model/` (`Provider::model`): no rule, skill
  or backend names one. Jev's `jev-latest` is System One's, in `decide/`.
- A module that is handed source never reads a file: `language/` and
  `snippet/` import no `std::fs`; only `skills`, `rule`, `review/plan.rs`,
  `review/diagnose.rs`, `agent/journal.rs` and `main.rs` touch the disk.
- No rule's id or outcome is matched in `src/`; judgment lives in the JSON.
- No `#[test]` or `#[cfg(test)]` module in `src/`; tests live in `tests/`.

## Contract
`src/lib.rs` declares nine modules: `agent`, `decide`, `model`, `rate_limit`,
`review`, `rule`, `skills` public; `language` and `snippet` private. Each has
its own `intent.md`. The CLI is `src/main.rs` (`Args`): paths to review;
`--backend` (`anthropic`, `gemini`, `claude-code`, `dry-run`), `--system-one`
(`none`, `jev`, `dry-run`), `--limiter`, `--min-confidence`, `--jobs`,
`--findings-only`, `--out`, `--rules`, `--skills`; it calls `skills::load`,
`rule::load_dir`, `review::{plan, review, diagnose, synthesize}`.

## Enforced by
- `desloppify/tests/shipped_rules.rs`: every rule in `rules/` loads, and each
  shows a model the unit it claims, read from the prompts a review sends.
- `desloppify/tests/rule.rs`: the loader refuses each kind of bad rule.
- `tests/review.rs`, `agent.rs`, `decide.rs`, `rate_limit.rs`: the rest, via
  the production API. `RuleFile` carries `deny_unknown_fields` (a type).
- `.github/workflows/rust.yaml` runs these with the workspace. Nothing checks
  the import and `fs` lines under Never; `grep` does today.

## Behind
- 2026-10-15: no `module_intent` context. A rule's `"context"` is `module_root`
  only (two rules use it); none is shown the nearest `intent.md` (plan §4.1).
- `skills/pixelflow-architecture` (46 lines) and `skills/style-guide` (44)
  are hand-kept distillations of CLAUDE.md (584) and STYLE.md (277), not
  links; only `skills/ontology` is a symlink (plan §5, not done).
- Nothing in CI runs desloppify over the tree; only its own tests run. The
  intent check, `scripts/check-intent.sh`, is called by no workflow yet (§4.2).

## Decided in
`desloppify/README.md`; `docs/process/2026-10-15-intent-lives-with-the-module.md`
(§4, §5). No other plan.
