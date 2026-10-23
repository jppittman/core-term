export const meta = {
  name: 'a3-polish',
  description: 'A3 polish: apply the closing pass non-blocking findings (false docs, dead derives, vacuous asserts), BACKLOG note, gate, one review',
  phases: [
    { title: 'Polish', detail: 'apply the listed findings' },
    { title: 'Gates', detail: 'full gate list' },
    { title: 'Review', detail: 'one reviewer on the polish diff' },
  ],
}
const A = '/home/user/pixelflow/.wip/a3'
const G = '/home/user/pixelflow/.wip/gates'
const BR = 'claude/integral-area-work-ojiej6'
const RULES = `Ground rules (hard):
- Repository /home/user/pixelflow, branch ${BR}; code base main 3bfdaded; the A3 diff is \`git diff 3bfdaded -- . ':!.wip'\`. Never build from a git worktree; never stash/reset/rebase/amend; never checkout a tracked file.
- Commit/push only where told: \`git add -A -- . ':!pixelflow-graphics/src/fonts/loop_blinn/program/tests.rs'\` only when that file is otherwise unmodified; no Cargo.toml/crate under .wip (scratch crates in /home/user/scratch); \`git lfs ls-files | grep -c .\` stays 68; subject as given, ending with your harness's attribution trailers; \`git push -q origin ${BR}\` (retry on network errors only).
- Never delete anything under /tmp/claude-0, /home/user/scratch or /home/user/pixelflow/.wip. Never \`cargo test --workspace\`. \`df -h /\` before builds; under 2 GB stop.
- No model names; no tracked file outside .wip cites a .wip path or ruling label.
- Clippy is CI's: \`cargo +1.99.0 clippy\`. For check-feature-matrix.sh prepend /home/user/scratch/cargo-hack-root/bin to PATH.
- Constraint: byte-identical on today's kernels (all-literal bounds); a literal fold does no new work.`

const GATES = `Gates (scripts in ${G}; each exits 0 only when identical to base): byteprobe.sh, gate1b.sh, gate2.sh + gate2x.sh, measure.sh (total Ir within +0.5% of base; report extract inclusive too), cargo test per crate (pixelflow-ir; pixelflow-search default/--all-features/--features saturation-telemetry; pixelflow-codegen; pixelflow-compiler; pixelflow-graphics; pixelflow-pipeline; pixelflow-core; pixelflow-runtime), \`cargo +1.99.0 clippy --workspace --all-targets --all-features -- -D warnings\`, \`cargo fmt --all -- --check\`, the two \`--no-default-features\` checks with -D warnings, the codegen \`-D warnings\` doc build, the ir/search rustdoc subset compare (${A}/run/rustdoc-base.md), scripts/check-intent.sh, check-doc-paths.sh, check-cfg-encapsulation.sh, check-bin-declarations.sh, check-feature-matrix.sh, \`python3 -I ${A}/gen/checklist.py\` (0/0), the B1 greps (\`grep -rnwE 'A3' --include=*.rs pixelflow-*\`, \`grep -nw 'A3' pixelflow-ir/intent.md\` empty), \`grep -rn 'fold_cost(' --include=*.rs\` only cost.rs, \`grep -rln '\\.wip' --include=*.rs --include=*.md . | grep -v '^./.wip/'\` empty.`

const ITEMS = `Apply every finding in ${A}/run/closing-nonblocking.json except the one marked blocking (already fixed in d4bcd11) and the one saying "No defect". Verify each against the code first; apply the fix as described or a better one that achieves the same; if one is wrong, say why with evidence. They are:
- F3 literal_twin cost: keep the literal fast path at the call site (an #[inline(always)] guard returning None unless the node is a Reduce with arity() > 1, calling a non-inlined body), re-measure; record the measured closing-pass Ir truthfully in ${A}/run/fixes.md.
- Docs made false: EGraph::admit_unit's caller (runtime::saturate, not Program::of); of_var's paragraph (a variance never decides closedness; closedness asks free_index; placement reads a variance only after codegen's Var arm refuses a non-binder Var); codegen lower.rs's panic split by case (coordinate = collapse skipped; index past the binder slots = an open kernel) and Binder::from_var's doc lists placeholders among None; literal_twin's and KnownBound's "cannot disagree"/"are the same" claims restated as what one construction buys; representative / substitute summary / plan L7 / "no builder makes one" texts brought to the Site walk; the GrowthPredictor doc (pending is iterated by fact); substitute's and plan_shape's decline condition stated in full (a bound class seeded wider than its term, whether it landed in a stale class or was minted over a child that did); literal_twin's "first test" wording; the plan's L7 sentence ("makes no union; the nodes it made before that fold stay, as orphans that cost classes, never soundness").
- Code: Site derives only Clone, Copy; an end-free Fold predicate (e.g. \`bound_may_read(self, reads: Variance) -> bool\`) that admits_bound and Site::member both call, so no fabricated BoundEnd; Site::member's Bound arm skips ENode::Reduce members; representative inlined or its doc says "under a body".
- Tests: replace the vacuous \`num_classes() >= classes\` assertion and the never-failing loop with assertions that can fail (what the docs promise: the plan was declined, P's fact still reads X, the value-bounded inner fold count is below the control's); optionally the (i + 1)·2 pin.
- Kernel::over's doc "an index cannot escape the fold that binds it" is false: say an index can escape through a clone, and a later fold that reclaims its placeholder captures it, which no check sees. Add a docs/BACKLOG.md row for that capture hole (pre-existing at main, not A3's: needs placeholders never reissued or an index identity outside Var(u8)), in BACKLOG's own row format.
- .wip only: spec.md J24 states the extraction hole generally (the cheapest member of a bound class reads a coordinate, which includes a class whose lawful member costs more), and J27's orphan wording.`

const GATE_SCHEMA = { type: 'object', properties: { all_pass: { type: 'boolean' },
  results: { type: 'array', items: { type: 'object', properties: { gate: { type: 'string' }, pass: { type: 'boolean' }, detail: { type: 'string' } }, required: ['gate', 'pass', 'detail'] } }, ir: { type: 'string' } }, required: ['all_pass', 'results', 'ir'] }
const FIND_SCHEMA = { type: 'object', properties: { findings: { type: 'array', items: { type: 'object', properties: {
  file: { type: 'string' }, line: { type: 'integer' }, claim: { type: 'string' }, evidence: { type: 'string' }, fix: { type: 'string' }, blocking: { type: 'boolean' } },
  required: ['file', 'line', 'claim', 'evidence', 'fix', 'blocking'] } } }, required: ['findings'] }

let round = 0, open = [], fails = [], gates = null
const history = []
while (round < 2) {
  round++
  phase('Polish')
  const fix = await agent(`${RULES}

${round === 1 ? ITEMS : `Fix what the polish round ${round - 1} gates and reviewer found. Verify each first.
Gate failures: ${JSON.stringify(fails)}
Findings: ${JSON.stringify(open)}`}

Then run the gates that cover what you changed (at least byteprobe.sh/gate1b.sh/gate2.sh+gate2x.sh with TAG=a3-polish${round}, cargo test of touched crates, clippy 1.99 on them, fmt, checklist). Append per-item outcomes to ${A}/run/fixes.md. Commit (subject "chore(wip): A3 polish, round ${round} (not for merge)") and push. Return the per-item report.`, { label: `polish:${round}`, phase: 'Polish', model: 'sonnet' })

  phase('Gates')
  gates = await agent(`${RULES}

Verify A3. Do not edit tracked files. Run every gate with TAG=a3-p${round}; commit the outputs only (\`git add .wip\`, subject "chore(wip): A3 polish gates, round ${round} (not for merge)") and push.

${GATES}`, { label: `gates:p${round}`, phase: 'Gates', model: 'sonnet', schema: GATE_SCHEMA })

  phase('Review')
  const rev = await agent(`${RULES}

Adversarial review of the A3 polish diff (\`git diff 7e1a2f3 -- . ':!.wip'\`; the whole A3 diff is \`git diff 3bfdaded -- . ':!.wip'\`) against the items it was asked to apply (${A}/run/closing-nonblocking.json and the list below) and CLAUDE.md. Read-only; no commits. Check: each item applied correctly; no doc it touched is now false; the end-free predicate and Site::member's Reduce skip are sound (decline never removed where it guarded L3); the new test assertions can fail; nothing moved on the literal path (check ${G}/out/measure.a3-p${round}.txt vs base); the BACKLOG row follows that file's format. Report a defect only with evidence; blocking = wrong behavior, a false doc, a violated rule, or a literal-path change. Earlier: ${JSON.stringify(history).slice(0, 4000)}

Items:
${ITEMS}`, { label: `review:p${round}`, phase: 'Review', schema: FIND_SCHEMA })
  open = (rev ? rev.findings : []).filter(f => f.blocking)
  fails = gates ? gates.results.filter(r => !r.pass) : [{ gate: 'gates', detail: 'gate agent returned nothing' }]
  history.push({ round, fix: String(fix).slice(0, 2000), gate_failures: fails.map(f => f.gate + ': ' + String(f.detail).slice(0, 200)), blocking: open.map(f => f.claim), nonblocking: (rev ? rev.findings : []).filter(f => !f.blocking).map(f => f.file + ':' + f.line + ' ' + f.claim) })
  log(`polish round ${round}: ${fails.length} gate failures, ${open.length} blocking`)
  if (fails.length === 0 && open.length === 0) break
}
return { rounds: round, gates, open_blocking: open, history }