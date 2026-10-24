export const meta = {
  name: 'a4-final',
  description: 'A4 final round: the closing pass\'s six blocking findings and its non-blocking ones, gates, one combined reviewer, at most two rounds',
  phases: [
    { title: 'Fix', detail: 'tests, docs, the zero-factor ruling' },
    { title: 'Gates', detail: 'full gate list' },
    { title: 'Review', detail: 'one reviewer: semantics, tests and docs' },
  ],
}
const A = '/home/user/pixelflow/.wip/a4'
const G = '/home/user/pixelflow/.wip/gates'
const BR = 'claude/integral-area-work-ojiej6'
const RULES = `Ground rules (hard):
- Repository /home/user/pixelflow, branch ${BR}; code base main a747770; the A4 diff is \`git diff a747770 -- . ':!.wip'\`. Never build from a git worktree; never stash/reset/rebase/amend; never checkout a tracked file (gate scripts restore their own probe file).
- Commit/push only where told: \`git add -A -- . ':!pixelflow-graphics/src/fonts/loop_blinn/program/tests.rs'\` only when that file is otherwise unmodified; no Cargo.toml/crate under .wip (scratch crates in /home/user/scratch); \`git lfs ls-files | grep -c .\` stays 68; subject as given, ending with your harness's attribution trailers; \`git push -q origin ${BR}\` (retry 4x on network errors only).
- Never delete anything under /tmp/claude-0 or /home/user/pixelflow/.wip. Never \`cargo test --workspace\`. \`df -h /\` before builds; under 3 GB free run \`cargo clean -q --profile dev -p pixelflow-pipeline\` once; still under 2 GB, stop.
- No model names; no tracked file outside .wip cites a .wip path or a ruling label.
- Clippy is CI's: \`cargo +1.99.0 clippy\` with CARGO_TARGET_DIR=/home/user/scratch/a3-close1/t199. For check-feature-matrix.sh prepend /home/user/scratch/cargo-hack-root/bin to PATH.
- Constraint: byte-identical on today's kernels; a value-bounded fold computes its fold or NaN on every tier and never hangs (except where ruling Z1 records the annihilation edge on the optimized path). Tier-dependent traffic pins are per tier; read every pin as NEON would.`

const DESIGN = `The design: ${A}/rulings.md (authoritative; its "Final round" section, Z1 and the docs discipline, governs this run), ${A}/spec.md, docs/language/2026-10-20-a-bound-is-a-value.md, CLAUDE.md.`

const ITEMS = `Apply every item in ${A}/run/c3-open.json (6 blocking, 7 non-blocking), verifying each against the code first, with these resolutions:
- Tests (blocking 1, non-blocking row-fold and hang-list items): add the in-domain value-bound tests at lo beside a literal hi (lo in {0, 1, 7, 1000}, hi literal 1003), at the top of the domain (MAX over [count(2^24-2), constant(2^24)) = 2^24-1; [count(2^24), constant(2^24)) = identity), and as pairs ([count(1000), count(1003)); MAX over [count(2^24-2), count(2^24)); [count(2^24), count(2^24))), all through both(); run the Y-only row-fold kernel through both() for n in {0, 7, NaN, +inf, 2^24+2}; add +inf and 2^24+2 to the nested-placement NaN lists. Re-run the reviewer's mutations MA2, MB2, MC2, MD2 (scripts in /home/user/scratch/a4-close4-guard-adv/mut/) against the new tests and record in ${A}/run/fixes.md that each now fails.
- Ruling Z1 (blocking 6 and the two zero-factor non-blocking items): docs in kernel.rs (over_values), fold_rules.rs (FactorFold paragraph, scoped to SUM and a finite body) and the plan cite the annihilation rule (\`0·y → 0\`, the Mul Annihilator) as the reason, not decision 5; the test pins +0.0 exactly through run_linked and NaN through run_raw and is named for that; one BACKLOG line per Z1. No FactorFold change.
- Docs (blocking 2-5 and the rest): the plan's A4 gate sentence states exactly what the tests run (after the new tests); passes.rs's module doc names every non-idempotent pass and says the guard leaves an all-literal kernel untouched; the mask-and-number test doc states the law, not register files; the totality doc and the plan's §5 cell say each bound is a literal, a value known per call or an enclosing fold's index, so the count hi − lo is known per call or once per trip of that fold; the two hang-rationale docs per tier (−∞ hangs unguarded on every tier; NaN also on NEON's ordered latch; x86's latch compare is unordered); emit/mod.rs:1619's test doc lists what a value-bounded loop actually adds.
Follow the "docs discipline" in the rulings: check every sentence you write; prefer deleting a doubtful clause.`

const GATES = `Gates (scripts in ${G}; each exits 0 only when identical to base): byteprobe.sh, gate1b.sh, gate2.sh + gate2x.sh, measure.sh (total Ir within +0.5% of base), cargo test per crate (pixelflow-ir; pixelflow-search default/--all-features/--features saturation-telemetry; pixelflow-codegen under PIXELFLOW_ISA=avx2 and avx512; pixelflow-compiler; pixelflow-graphics; pixelflow-pipeline; pixelflow-core; pixelflow-runtime), \`cargo +1.99.0 clippy --workspace --all-targets --all-features -- -D warnings\`, \`cargo fmt --all -- --check\`, the two \`--no-default-features\` checks with -D warnings, the codegen \`-D warnings\` doc build, ir/search rustdoc compare (\`python3 -I ${A}/rustdoc_norm.py --compare\` vs ${A}/rustdoc-base.*.txt), scripts/check-intent.sh, check-doc-paths.sh, check-cfg-encapsulation.sh, check-bin-declarations.sh, check-emit-boundary.sh, check-feature-matrix.sh; \`grep -rnwE 'A4' --include=*.rs pixelflow-*\` hits all true after A4; \`grep -rln '\\.wip' --include=*.rs --include=*.md . | grep -v '^./.wip/'\` empty.`

const GATE_SCHEMA = { type: 'object', properties: { all_pass: { type: 'boolean' },
  results: { type: 'array', items: { type: 'object', properties: { gate: { type: 'string' }, pass: { type: 'boolean' }, detail: { type: 'string' } }, required: ['gate', 'pass', 'detail'] } }, ir: { type: 'string' } }, required: ['all_pass', 'results', 'ir'] }
const FIND_SCHEMA = { type: 'object', properties: { findings: { type: 'array', items: { type: 'object', properties: {
  file: { type: 'string' }, line: { type: 'integer' }, claim: { type: 'string' }, evidence: { type: 'string' }, fix: { type: 'string' }, blocking: { type: 'boolean' } },
  required: ['file', 'line', 'claim', 'evidence', 'fix', 'blocking'] } } }, required: ['findings'] }

let round = 0, open = [], fails = [], gates = null, nonblocking = []
const history = []
while (round < 2) {
  round++
  phase('Fix')
  const fix = await agent(`${RULES}

${DESIGN}

${round === 1 ? ITEMS : `Fix what the final round's gates and reviewer found; verify each first; apply the plainly correct non-blocking ones that are defects in A4's own diff.
Gate failures: ${JSON.stringify(fails)}
Blocking: ${JSON.stringify(open)}
Non-blocking: ${JSON.stringify(nonblocking).slice(0, 8000)}`}

Then run gates 1-3 (TAG=a4-final${round}), cargo test of every crate you touched (pixelflow-codegen under PIXELFLOW_ISA=avx2 and avx512), clippy 1.99 on them, fmt. Append per-item outcomes to ${A}/run/fixes.md. Commit (subject "chore(wip): A4 final fixes, round ${round} (not for merge)") and push. Return the per-item report.`, { label: `fix:f${round}`, phase: 'Fix', model: 'sonnet' })

  phase('Gates')
  gates = await agent(`${RULES}

Verify A4. Do not edit tracked files. Run every gate with TAG=a4-f${round}; commit the outputs only (\`git add .wip\`, subject "chore(wip): A4 final gates, round ${round} (not for merge)") and push.

${GATES}`, { label: `gates:f${round}`, phase: 'Gates', model: 'sonnet', schema: GATE_SCHEMA })

  phase('Review')
  const rev = await agent(`${RULES}

${DESIGN}

Adversarial review of A4 (\`git diff a747770 -- . ':!.wip'\`), focused on the final round (\`git diff 98dcbca -- . ':!.wip'\`). Read-only on tracked files; no commits. Three questions:
1. Semantics and pins: for every bound value class at lo, at hi, and at both ends of a pair (in-domain incl. 0, 1, 1000, 2^24; out-of-domain NaN, ±inf, -1, 2.5, 2^24+2), is there a test that fails if the guard mishandles it? Mutate passes/guard.rs in a scratch copy under /home/user/scratch to check (keep the op counts, move one constant or one arm). Does the code still compute the fold or NaN (Z1's edge aside) on both x86 tiers, and would it on NEON?
2. Docs: is every doc sentence the final round wrote or changed true of the code? Check each one; report a false one as blocking.
3. Craft: no vacuous assertion, no duplicated helper, no boolean args, comments that parse.
Report only defects shown with evidence. Earlier final rounds: ${JSON.stringify(history).slice(0, 4000)}`, { label: `review:f${round}`, phase: 'Review', schema: FIND_SCHEMA })
  const all = (rev ? rev.findings : [])
  open = all.filter(f => f.blocking)
  nonblocking = all.filter(f => !f.blocking)
  fails = gates ? gates.results.filter(r => !r.pass) : [{ gate: 'gates', detail: 'gate agent returned nothing' }]
  history.push({ round, fix: String(fix).slice(0, 2000), gate_failures: fails.map(f => f.gate + ': ' + String(f.detail).slice(0, 200)), blocking: open.map(f => f.claim), nonblocking: nonblocking.map(f => f.file + ':' + f.line + ' ' + f.claim) })
  log(`final round ${round}: ${fails.length} gate failures, ${open.length} blocking, ${nonblocking.length} non-blocking`)
  if (fails.length === 0 && open.length === 0) break
}
return { rounds: round, gates, open_blocking: open, nonblocking, history }