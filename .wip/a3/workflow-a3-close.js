export const meta = {
  name: 'a3-close',
  description: 'A3 closing pass: fix two blocking round-3 defects and the plainly correct nits, gate, review the fix, loop',
  phases: [
    { title: 'Fix', detail: 'unit closedness, L7 lawful member, one literal twin, doc nits' },
    { title: 'Gates', detail: 'bytes, keys, saturation, Ir, tests, clippy 1.99, docs, greps' },
    { title: 'Review', detail: 'soundness of the fixes; nothing moves on the literal path' },
  ],
}
const A = '/home/user/pixelflow/.wip/a3'
const G = '/home/user/pixelflow/.wip/gates'
const BR = 'claude/integral-area-work-ojiej6'
const RULES = `Ground rules (hard):
- Repository /home/user/pixelflow, branch ${BR}. The code base is main 3bfdaded; the A3 diff is \`git diff 3bfdaded -- . ':!.wip'\` (HEAD also carries chore(wip) commits; .wip/ holds working notes). Never build from a git worktree. Never stash/reset/rebase/amend; never checkout a tracked file (the gate scripts restore their own probe file).
- Commit or push only where your task says so: \`git add -A -- . ':!pixelflow-graphics/src/fonts/loop_blinn/program/tests.rs'\` only when that file is otherwise unmodified; never add a Cargo.toml/crate under .wip (scratch crates live in /home/user/scratch); \`git lfs ls-files | grep -c .\` must stay 68; subject exactly as given, ending with your harness's attribution trailer lines; push \`git push -q origin ${BR}\` (retry 4 times 2/4/8/16 s on a network error only).
- Never delete anything under /tmp/claude-0, /home/user/scratch or /home/user/pixelflow/.wip.
- Never run \`cargo test --workspace\`; test crate by crate. \`df -h /\` before builds; under 2 GB free, stop.
- No model names in any file; no tracked file outside .wip cites a .wip path or a .wip ruling label.
- Clippy is CI's: \`cargo +1.99.0 clippy ...\` (installed beside the default 1.97). Builds, tests and measure.sh use the default toolchain.`

const DESIGN = `The design: ${A}/rulings.md (rounds 1-3, authoritative), ${A}/spec.md, docs/language/2026-10-20-a-bound-is-a-value.md. Constraint: byte-identical on today's kernels (every bound a literal): byte_probe both tiers, the 29 font programs' nodes/keys/digests, saturation telemetry including extraction fields; a literal fold does no new work.`

const FIXES = `## Round 4 rulings (authoritative; record them at the end of ${A}/rulings.md as "Round 4")
**F1 (blocking) — a leaked placeholder must not pass the unit check.** Since J5 a placeholder Var (64..128) has variance CONST, so \`Program::of\`'s Law U check in pixelflow-search/src/runtime.rs (the \`variance.without(COORDS).is_const()\` assert) admits a unit holding a leaked fold index that main refused (repro: \`let mut leaked = None; Kernel::sum_over(2, |i| { leaked = Some(i.clone()); i.clone() });\`, \`KernelStore::intern\` of the leaked kernel, \`push_ref(key) + X\` into \`optimize_runtime_arena(.., LatticeShape::POINT)\` — main panics, head returns \`add(Var(64), Var(0))\`; notes ${A}/run/review-r3-literal-notes.txt). Fix: \`Program::of\` also refuses a unit whose body has \`free_index(body_root).is_some()\` — the test \`Kernel::by_ref\` makes — with a message naming the escaped index. Then find every other check that judges closedness or placement by variance and could see a placeholder (codegen \`arena_to_schedule\` / placement in pixelflow-codegen/src/program/, \`lower_dwrt\` in passes.rs, the runtime tier, anything else \`grep -rn 'compute_arena_variance\\|of_var' --include=*.rs\` finds): each must refuse a placeholder loudly (by free_index or an existing refusal you cite with file:line), never silently treat it as constant. Fix \`of_var\`'s doc ("nothing that places a node sees a placeholder" is false: a placeholder can escape its builder; closedness checks use free_index). Pin: a should_panic test in pixelflow-search with the repro program.
**F2 (blocking) — L7 never builds an unlawful bound.** \`PlanBuilder::substitute\` (fold_rules.rs) walks a class through \`representative\` (its first member). For a class reached as a nested fold's VALUE BOUND, the first member can read a coordinate the class does not vary with (the class fact is the meet over members), and the rebuilt bound then reads X: \`EGraph::add\`'s L3 backstop panics with BoundVaries inside saturation (repro ${A}/run/scratch-r4-run1.txt and its source ${A}/run/scratch-r4/representative.rs: \`Σ_{i<8} ( Σ_{j<i} X·j + ((i + X) + 1) − X )\` with the inner bound \`Var(i)\`, \`Vocabulary::Runtime\`, \`RuleSet::runtime()\`, \`SaturationConfig::compatibility(12)\`). Fix: when substitution reaches a class as a nested fold's value bound, walk the first member whose own variance (\`ENode::variance\` over its children's class facts, after the substitution's own effect is accounted for) admits that end under the fold's L3 (\`Fold::admits_bound\`); if none does, the plan declines (None) — declining costs completeness, never soundness. Bodies keep \`representative\`. Literal folds are untouched (no value-bound child). Pins: the repro saturates without panicking under the runtime set and every extracted inner bound reads only the outer index; the two minimal hand-made-union cases (HalveFold alone at n=4, PeelFold alone at n=1) decline instead of panicking. Fix HalveFold's doc ("EGraph::add accepts it…") to say why it now holds. Also state in spec §13 and as one sentence in the plan's A4 bullet the remaining extraction-side hole: extraction picks one member per class, so a bound class whose chosen member reads a coordinate the class does not vary with panics at materialization with BoundVaries (loud); A4 decides whether a bound's pick is constrained — unless your F2 pin shows extraction already never picks such a member, in which case say why.
**F3 — one literal twin.** \`literal_twin\` (extract.rs) and \`KnownBound::apply\` (fold_rules.rs) build the same thing twice. Keep one function (e.g. \`ENode::literal_twin(&self, literal: impl Fn(EClassId) -> Option<Lit>) -> Option<ENode>\` in node.rs, \`pub(crate)\` or narrower) and call it from both; the docs stop asserting the two agree, because they are one.
**F4 — doc nits, plainly correct:** KnownBound's rustdoc stops claiming "a single label" differs (also \`Saturation::total_evals\` rises on every kernel with a fold); anchor the plan's law labels in pixelflow-search docs where first used ("L3 of docs/language/2026-10-20-a-bound-is-a-value.md"), since L1-L5 there already name the optimizer laws; \`FoldHeader::new\`'s summary line says what it checks (not "the one place a fold is checked against its bounds"); \`ENode::reduce\`'s Panics keeps only "As [\`FoldHeader::new\`]"; \`is_the_fixture\`'s doc names \`copy\`; variance.rs's stale comment above \`transfer\`'s Var arm; \`close_over\`'s doc paragraph order (the "Constness is discovered first" paragraph must not break "that rule"'s antecedent).
**F5 — .wip only:** ${A}/run/public-surface.md: drop the must_use row (R1 removed them), add \`egraph::extract\`/\`extract_dag\` (now price a value-bounded fold by its literal twin, else ESTIMATED_TRIPS) and the private \`FoldHeader::with_a_value\`, and whatever F1-F3 add.`

const GATES = `## Gates (scripts in ${G}; README there; each exits 0 only when identical to base)
1. \`bash ${G}/byteprobe.sh TAG\`; 2. \`bash ${G}/gate1b.sh TAG\`; 3. \`bash ${G}/gate2.sh TAG && bash ${G}/gate2x.sh TAG\`; then \`git status --short pixelflow-graphics\` empty.
4. \`bash ${G}/measure.sh TAG\`: total Ir within +0.5% of base (out/measure.TAG.txt vs_base).
5. cargo test per crate: pixelflow-ir; pixelflow-search (default, --all-features, --features saturation-telemetry); pixelflow-codegen; pixelflow-compiler; pixelflow-graphics; pixelflow-pipeline; pixelflow-core; pixelflow-runtime.
6. \`cargo +1.99.0 clippy --workspace --all-targets --all-features -- -D warnings\`; \`cargo fmt --all -- --check\`; \`RUSTFLAGS="-D warnings" cargo check -p pixelflow-ir --no-default-features\` and the same for pixelflow-search; \`RUSTDOCFLAGS="-D warnings" cargo doc -p pixelflow-codegen --no-deps --document-private-items\`; ir/search rustdoc warning sets vs ${A}/run/rustdoc-base.*.txt by ${A}/run/rustdoc-base.md's method: no new warning.
7. \`bash scripts/check-intent.sh\`, \`bash scripts/check-doc-paths.sh\`, \`bash scripts/check-cfg-encapsulation.sh\`, \`bash scripts/check-bin-declarations.sh\`, \`bash scripts/check-feature-matrix.sh\` (cargo-hack is installed), \`python3 -I ${A}/gen/checklist.py\` (0/0).
8. Greps: \`grep -rnwE 'A3' --include=*.rs pixelflow-*\` and \`grep -nw 'A3' pixelflow-ir/intent.md\` empty; \`grep -rn 'fold_cost(' --include=*.rs\` only cost.rs; \`grep -rln '\\.wip' --include=*.rs --include=*.md . | grep -v '^./.wip/'\` empty.`

const GATE_SCHEMA = { type: 'object', properties: { all_pass: { type: 'boolean' },
  results: { type: 'array', items: { type: 'object', properties: { gate: { type: 'string' }, command: { type: 'string' }, pass: { type: 'boolean' }, detail: { type: 'string' } }, required: ['gate', 'command', 'pass', 'detail'] } },
  ir: { type: 'string' } }, required: ['all_pass', 'results', 'ir'] }
const FIND_SCHEMA = { type: 'object', properties: { findings: { type: 'array', items: { type: 'object', properties: {
  file: { type: 'string' }, line: { type: 'integer' }, claim: { type: 'string' }, evidence: { type: 'string' }, fix: { type: 'string' }, blocking: { type: 'boolean' } },
  required: ['file', 'line', 'claim', 'evidence', 'fix', 'blocking'] } } }, required: ['findings'] }

let round = 0, open = [], fails = [], gates = null
const history = []
while (round < 3) {
  round++
  phase('Fix')
  const fix = await agent(`${RULES}

${DESIGN}

${round === 1 ? FIXES : `You fix what round ${round - 1}'s gates and reviewers found in the A3 closing pass. For EACH item: verify against the code first; fix if real, else say why not with evidence.
Gate failures: ${JSON.stringify(fails)}
Blocking findings: ${JSON.stringify(open)}
The round-4 rulings (F1-F5) at the end of ${A}/rulings.md stay authoritative.`}

Implement, then run gates 1-3 (TAG=a3-close${round}), \`cargo test\` of every crate you touched (pixelflow-search in all three feature sets), \`cargo +1.99.0 clippy -p <touched> --all-targets --all-features -- -D warnings\`, fmt, the checklist (update ${A}/gen/classes.json for any new Reduce-site hit). Append to ${A}/run/fixes.md (per item: done how, or rejected why). Commit (subject "chore(wip): A3 closing fixes, round ${round} (not for merge)") and push. Return the per-item report with test names added.`, { label: `fix:c${round}`, phase: 'Fix', model: 'sonnet' })

  phase('Gates')
  gates = await agent(`${RULES}

Verify A3 (byte-identical on today's kernels). Do not edit tracked files. Run every gate with TAG=a3-c${round}, then commit the outputs only (\`git add .wip\`, subject "chore(wip): A3 closing gates, round ${round} (not for merge)") and push.

${GATES}`, { label: `gates:c${round}`, phase: 'Gates', model: 'sonnet', schema: GATE_SCHEMA })

  phase('Review')
  const lenses = [
    { key: 'soundness', text: `Lens: are F1 and F2 (and F3) right and complete? F1: can any placeholder still reach a check that judges closedness or placement by variance and pass silently (try the repro and variations: a leaked index inside a value bound, inside a nested fold, through \`Kernel::at\`, \`select\`, \`by_ref\`, codegen directly)? F2: does the lawful-member walk compute each member's variance correctly AFTER substitution (a member reading the substituted slot), decline when none is lawful, never change a body walk, and leave no other path where a rule (PeelFold, HalveFold, FactorFold, KnownBound, rebuild, the algebra rules) builds a value bound reading a coordinate or its own binder? Try to construct a counterexample under \`RuleSet::runtime()\`; a scratch crate goes in /home/user/scratch only. F3: one definition, both callers. Report a defect only with evidence.` },
    { key: 'literal', text: `Lens: the closing diff (\`git diff 8da7c86 -- . ':!.wip'\`) moves nothing on the literal path: no byte, key, node id, price, application count or extraction choice for an all-literal kernel; no new work on a literal fold (check ${G}/out/measure.a3-c${round}.txt vs base); Program::of's new refusal never fires on a production unit (every font program's units are closed — show why); the F2 member walk is reached only for a value-bound child. Also CLAUDE.md craft on the closing diff: docs say what things mean, no boolean args, no dead code, no stale text, comments in the file's register.` },
  ]
  const reviews = await parallel(lenses.map(l => () => agent(`${RULES}

${DESIGN}

Adversarial review of the A3 closing pass (rounds of fixes after commit 8da7c86; the whole A3 diff is \`git diff 3bfdaded -- . ':!.wip'\`). The rulings it implements are "Round 4" at the end of ${A}/rulings.md. Read-only on tracked files; no commits. Earlier closing rounds: ${JSON.stringify(history).slice(0, 6000)}

${l.text}`, { label: `review:${l.key}:c${round}`, phase: 'Review', schema: FIND_SCHEMA })))
  open = reviews.filter(Boolean).flatMap((r, i) => r.findings.map(f => ({ lens: lenses[i].key, ...f }))).filter(f => f.blocking)
  fails = gates ? gates.results.filter(r => !r.pass) : [{ gate: 'gates', detail: 'gate agent returned nothing' }]
  history.push({ round, fix: String(fix).slice(0, 2500), gate_failures: fails.map(f => f.gate + ': ' + String(f.detail).slice(0, 200)), blocking: open.map(f => f.lens + ': ' + f.claim) })
  log(`closing round ${round}: ${fails.length} gate failures, ${open.length} blocking findings`)
  if (fails.length === 0 && open.length === 0) break
}
return { rounds: round, gates, open_blocking: open, history }