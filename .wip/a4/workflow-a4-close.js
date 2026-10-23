export const meta = {
  name: 'a4-close',
  description: 'A4 closing pass: fix the four round-3 blocking findings and the plainly correct nits, gate, two reviewers, loop',
  phases: [
    { title: 'Fix', detail: 'blocking findings and nits' },
    { title: 'Gates', detail: 'full gate list' },
    { title: 'Review', detail: 'semantics; literal path and craft' },
  ],
}
const A = '/home/user/pixelflow/.wip/a4'
const G = '/home/user/pixelflow/.wip/gates'
const BR = 'claude/integral-area-work-ojiej6'
const RULES = `Ground rules (hard):
- Repository /home/user/pixelflow, branch ${BR}; code base main a747770; the A4 diff is \`git diff a747770 -- . ':!.wip'\`. Never build from a git worktree; never stash/reset/rebase/amend; never checkout a tracked file (gate scripts restore their own probe file).
- Commit/push only where told: \`git add -A -- . ':!pixelflow-graphics/src/fonts/loop_blinn/program/tests.rs'\` only when that file is otherwise unmodified; no Cargo.toml/crate under .wip (scratch crates in /home/user/scratch); \`git lfs ls-files | grep -c .\` stays 68; subject as given, ending with your harness's attribution trailers; \`git push -q origin ${BR}\` (retry 4x on network errors only).
- Never delete anything under /tmp/claude-0 or /home/user/pixelflow/.wip. Never \`cargo test --workspace\`. \`df -h /\` before builds; under 3 GB free run \`cargo clean -q --profile dev -p pixelflow-pipeline\` once (its bins are the big reclaimable item); still under 2 GB, stop.
- No model names; no tracked file outside .wip cites a .wip path or a ruling label.
- Clippy is CI's: \`cargo +1.99.0 clippy\` with CARGO_TARGET_DIR=/home/user/scratch/a3-close1/t199. For check-feature-matrix.sh prepend /home/user/scratch/cargo-hack-root/bin to PATH.
- Constraint: byte-identical on today's kernels (all-literal bounds); a value-bounded fold computes its fold or NaN on every tier and never hangs. Tier-dependent traffic pins are per tier (NEON keeps what AVX-512's opmask file cannot; read every pin as NEON would — the NEON leg runs only in CI).`

const DESIGN = `The design: ${A}/rulings.md (authoritative), ${A}/spec.md, docs/language/2026-10-20-a-bound-is-a-value.md, CLAUDE.md.`

const ITEMS = `Apply every item in ${A}/run/r3-open.json (open_blocking: 4; nonblocking: 8), verifying each against the code first:
- Blocking 1 (semantics): pin a value lo outside the domain beside a literal hi. JIT test through the file's \`both\`/\`within\` helpers: \`Kernel::over_values(Monoid::SUM, &count(lo), &Kernel::constant(9.0), |k| Kernel::x().add(k))\` for lo in {NaN, -inf, +inf, -1, 2.5, 2^24+2} → NaN at every sample, with the hang watchdog the file already uses; a pair of two values with lo out of domain and hi in it (lo = NaN, hi = count(9)); a guard.rs unit test that \`[u, Lit(8))\` has dom reading u (two Le, one Floor, one Eq) beside \`a_literal_bound_adds_no_domain_test\`. Show the new tests fail under the reviewer's mutation (diff in ${A}/run/r4/mutation.diff.txt) and pass at head; record that in fixes.md.
- Blocking 2: \`a_bound_on_a_lattice_binder_fails_loudly\`'s doc gives L3's real reason (a bound is fixed before the lattice is traversed; only the lane binder would vary a mask within a batch); align \`assert_off_the_lattice\`'s doc wording.
- Blocking 3: docs/language/2026-07-24-totality-and-the-cost-model.md :110-112 and the table row :117 state what holds after A4 (a fold's count is a literal or a value known per call; the other faces static until B1), linking the plan.
- Blocking 4: the docs that justify refusing a mask bound name the real reason for both ends (a mask is not a count, whichever end; AVX-512's opmask panic is the tier divergence that made it urgent for lo), wherever that justification appears (plan, lower.rs, CLAUDE.md, emit docs).
- Non-blocking: BoundRefusal::BoundSort's doc (lowering now raises it for a computed mask bound); CLAUDE.md:180's domain sentence reads unambiguously; emit/mod.rs's loop-nest paragraph covers the guarded fold that reads no coordinate; passes.rs:24's module doc (the guard has an identity fast path too); lower.rs's test helper duplicated (\`a_fold_bounded_by\` vs \`a_guarded_fold_over_a_count\` — one helper); cell_grid.rs's dump tests use a unique temp path as the codebase's corpus tests do; regalloc/local.rs:430's comment parses and says what the saturating sum is for; the .wip spec item may be ignored.`

const GATES = `Gates (scripts in ${G}; each exits 0 only when identical to base): byteprobe.sh, gate1b.sh, gate2.sh + gate2x.sh, measure.sh (total Ir within +0.5% of base), cargo test per crate (pixelflow-ir; pixelflow-search default/--all-features/--features saturation-telemetry; pixelflow-codegen under PIXELFLOW_ISA=avx2 and avx512; pixelflow-compiler; pixelflow-graphics; pixelflow-pipeline; pixelflow-core; pixelflow-runtime), \`cargo +1.99.0 clippy --workspace --all-targets --all-features -- -D warnings\`, \`cargo fmt --all -- --check\`, the two \`--no-default-features\` checks with -D warnings, the codegen \`-D warnings\` doc build, ir/search rustdoc compare (\`python3 -I ${A}/rustdoc_norm.py --compare\` against ${A}/rustdoc-base.*.txt), scripts/check-intent.sh, check-doc-paths.sh, check-cfg-encapsulation.sh, check-bin-declarations.sh, check-emit-boundary.sh, check-feature-matrix.sh; \`grep -rnwE 'A4' --include=*.rs pixelflow-*\` hits all true after A4; \`grep -rln '\\.wip' --include=*.rs --include=*.md . | grep -v '^./.wip/'\` empty.`

const GATE_SCHEMA = { type: 'object', properties: { all_pass: { type: 'boolean' },
  results: { type: 'array', items: { type: 'object', properties: { gate: { type: 'string' }, pass: { type: 'boolean' }, detail: { type: 'string' } }, required: ['gate', 'pass', 'detail'] } }, ir: { type: 'string' } }, required: ['all_pass', 'results', 'ir'] }
const FIND_SCHEMA = { type: 'object', properties: { findings: { type: 'array', items: { type: 'object', properties: {
  file: { type: 'string' }, line: { type: 'integer' }, claim: { type: 'string' }, evidence: { type: 'string' }, fix: { type: 'string' }, blocking: { type: 'boolean' } },
  required: ['file', 'line', 'claim', 'evidence', 'fix', 'blocking'] } } }, required: ['findings'] }

let round = 0, open = [], fails = [], gates = null, nonblocking = []
const history = []
while (round < 3) {
  round++
  phase('Fix')
  const fix = await agent(`${RULES}

${DESIGN}

${round === 1 ? ITEMS : `Fix what closing round ${round - 1}'s gates and reviewers found; verify each first. Also apply the plainly correct non-blocking ones that are defects in A4's own diff (false docs, vacuous tests); skip taste.
Gate failures: ${JSON.stringify(fails)}
Blocking: ${JSON.stringify(open)}
Non-blocking: ${JSON.stringify(nonblocking).slice(0, 8000)}`}

Then run gates 1-3 (byteprobe.sh/gate1b.sh/gate2.sh+gate2x.sh, TAG=a4-close${round}), cargo test of every crate you touched (pixelflow-codegen under both PIXELFLOW_ISA=avx2 and avx512), clippy 1.99 on them, fmt. Append per-item outcomes to ${A}/run/fixes.md. Commit (subject "chore(wip): A4 closing fixes, round ${round} (not for merge)") and push. Return the per-item report.`, { label: `fix:c${round}`, phase: 'Fix', model: 'sonnet' })

  phase('Gates')
  gates = await agent(`${RULES}

Verify A4. Do not edit tracked files. Run every gate with TAG=a4-c${round}; commit the outputs only (\`git add .wip\`, subject "chore(wip): A4 closing gates, round ${round} (not for merge)") and push.

${GATES}`, { label: `gates:c${round}`, phase: 'Gates', model: 'sonnet', schema: GATE_SCHEMA })

  phase('Review')
  const lenses = [
    { key: 'semantics', text: `Lens: run-time correctness of value-bounded folds on every tier, and whether the tests pin it. For each bound class at either end (lo and hi separately and together: 0, -0.0, 1, 7, 2^24, reversed pair, 2.5, -1, NaN, ±inf, 2^24+2, a value lo reading an enclosing index), is there a test that would fail if the guard's handling of that end were broken? Run mutations of passes/guard.rs in a scratch copy under /home/user/scratch to find gaps. Is every doc about the guard, the domain, L3 and the mask refusal true? Construct counterexamples if you can.` },
    { key: 'literal-craft', text: `Lens: the closing diff (\`git diff 4954553 -- . ':!.wip'\`) moves nothing on the literal path (check ${G}/out/measure.a4-c${round}.txt vs base and the byte gates), and it meets CLAUDE.md: docs say what things mean and are true, no boolean args, one definition per rule, no dead code, no vacuous tests, comments that parse, no churn on untouched lines, nothing names a model or a .wip path.` },
  ]
  const reviews = await parallel(lenses.map(l => () => agent(`${RULES}

${DESIGN}

Adversarial review of A4 (\`git diff a747770 -- . ':!.wip'\`), focused on the closing pass. Read-only on tracked files; no commits. Report only defects you can show with evidence; blocking = wrong values, a hang, a literal-path change, a false doc, a violated rule, a missing pin for a guarded behaviour. Earlier closing rounds: ${JSON.stringify(history).slice(0, 5000)}

${l.text}`, { label: `review:${l.key}:c${round}`, phase: 'Review', schema: FIND_SCHEMA })))
  const all = reviews.filter(Boolean).flatMap((r, i) => r.findings.map(f => ({ lens: lenses[i].key, ...f })))
  open = all.filter(f => f.blocking)
  nonblocking = all.filter(f => !f.blocking)
  fails = gates ? gates.results.filter(r => !r.pass) : [{ gate: 'gates', detail: 'gate agent returned nothing' }]
  history.push({ round, fix: String(fix).slice(0, 2000), gate_failures: fails.map(f => f.gate + ': ' + String(f.detail).slice(0, 200)), blocking: open.map(f => f.lens + ': ' + f.claim), nonblocking: nonblocking.map(f => f.file + ':' + f.line + ' ' + f.claim) })
  log(`closing round ${round}: ${fails.length} gate failures, ${open.length} blocking, ${nonblocking.length} non-blocking`)
  if (fails.length === 0 && open.length === 0) break
}
return { rounds: round, gates, open_blocking: open, nonblocking, history }