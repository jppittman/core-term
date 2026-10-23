export const meta = {
  name: 'a3-value-bounds',
  description: 'A3: a fold bound may be a term in the IR and e-graph; byte-identical on today\'s kernels, gated, reviewed',
  phases: [
    { title: 'Base', detail: 'Ir measurement and the Reduce-site checklist at main' },
    { title: 'Implement IR', detail: 'pixelflow-ir: fold.rs, variance.rs, arena.rs, kernel.rs' },
    { title: 'Implement search', detail: 'pixelflow-search: node, graph, insert, fold_rules, cost, extract' },
    { title: 'Implement tail', detail: 'codegen, compiler, docs; first byte/key gates' },
    { title: 'Gates', detail: 'bytes, keys, saturation, Ir, tests, lint, greps' },
    { title: 'Review', detail: 'four adversarial lenses' },
    { title: 'Fix', detail: 'verified findings and gate failures' },
  ],
}
const A = '/home/user/scratch/bound-value/a3'
const N = '/home/user/scratch/bound-value/narrow'
const G = '/home/user/scratch/egraph/arena/gates'
const RULES = `Ground rules (hard):
- Repository /home/user/pixelflow, branch claude/integral-area-work-ojiej6, base main 3bfdaded. Work only in that checkout; never build from a git worktree. Never commit, never push. Never stash/reset/rebase/checkout tracked files (the gate scripts restore their own probe file; leave that to them).
- Never delete anything under /tmp/claude-0 or /home/user/scratch, including files you wrote: overwrite or use a new name.
- Never run \`cargo test --workspace\`; test crate by crate.
- Disk: \`df -h /\` before every build. Under 1.5 GB free: you may run \`cargo clean --profile dev -p <crate>\` for core-term, pixelflow-runtime, pixelflow-ml, actor-scheduler, xtask only (never another crate, never release, never --target). Still under 1.0 GB: stop and report.
- No model names or identifiers in any file you write.
- Your notes and outputs go under ${A}/run/ (mkdir -p; never delete).`

const DESIGN = `## The design (authoritative, in this order)
1. ${A}/rulings.md — rulings B1, B2, N1-N10 on the critique. Where a ruling and the spec differ, the ruling wins.
2. ${A}/spec.md — the A3 implementation spec (1,219 lines), written at f772852c. The base is now 3bfdaded (#1446 narrowed FoldHeader's storage: \`lo\`, \`hi\`, \`stride\` are u32 fields narrowed by \`field()\`, with \`lo_held\`/\`hi_held: Held\`; every reader still returns u64; a Value bound already stores Held::Value with count 0; \`FoldHeader::store\` is the one writer). Re-grep; never trust a spec line number. ${A}/tree/ is a stale export of f772852c: read the live tree only.
3. ${A}/critique.md — the review the rulings answer (read the finding a ruling names for its detail).
4. docs/language/2026-10-20-a-bound-is-a-value.md §1-§4, §6's A3 bullet, §8 — the plan.

Fixed constraints:
- BYTE-IDENTICAL on today's kernels, where every bound is a literal: byte_probe on both tiers, the 29 font programs' keys and digests, saturation telemetry including the extraction fields. A literal fold must build node-for-node the same arena (no Const pushed then discarded), run no L3 variance walk, and price exactly as today.
- Not A3 (spec §0 "Is not"): codegen still refuses a value bound by name (VALUE_BOUND in arena_to_schedule); FoldHeader::to_bits/from_bits keep refusing a value bound; kernel! keeps literal bounds (its close_fold call sites change mechanically only).
- Every intent.md stays within 80 lines (\`bash scripts/check-intent.sh\`).
- CLAUDE.md style: match the surrounding comment density and idiom; rustdoc on new pub items says what they mean; no boolean args; #[must_use] on new pub fns returning a value; no visibility widening except what the spec names. Keep ${A}/run/public-surface.md listing every new or changed pub item (decision 7 approved the API; the PR must still list it).
- The Reduce-site checklist: ${A}/gen/checklist.py with its classification table ${A}/gen/classes.json (built at base by the Base phase). Every new hit you create gets a row with its class and a one-line reason; a site you change gets its class updated.`

const GATES = `## Gates (TAG given in the prompt; report each with its command and result; a gate you could not run is a fail)
1. byte_probe: \`PIXELFLOW_ISA=avx512 cargo run --release -q -p pixelflow-codegen --example byte_probe > ${A}/run/byteprobe.TAG.avx512.txt 2> ${A}/run/byteprobe.TAG.avx512.err\`, same for avx2. Pass = cmp-equal to ${N}/byteprobe.final.avx512.txt and ${N}/byteprobe.final.avx2.txt.
2. Keys: \`bash ${G}/gate1b.sh a3-TAG\` (ignore its exit status); pass = ${G}/gate1b.a3-TAG.txt cmp-equal to ${G}/gate1b.narrow-final.txt.
3. Saturation: \`bash ${G}/gate2.sh a3-TAG\` (ignore its exit status); pass = ${G}/gate2.a3-TAG.txt cmp-equal to ${G}/gate2.narrow-final.txt AND \`python3 -I ${G}/gate2x_fields.py ${G}/gate2.a3-TAG.jsonl\` equal to the same on ${G}/gate2.narrow-final.jsonl.
After 2 and 3, \`git status --short pixelflow-graphics\` shows no change to pixelflow-graphics/src/fonts/loop_blinn/program/tests.rs.`

const FULL_GATES = `${GATES}
4. Compile cost: \`bash ${G}/measure.sh a3-TAG\` (read ${G}/README.md "Gate 5" first). Compare total Ir and Saturation::saturate / extraction inclusive Ir against ${G}/measure.a3-base.txt. Pass = total Ir within +0.5% of base. Callgrind is deterministic: a larger rise on all-literal kernels means work runs on the literal path that should not (an L3 walk, a discovery, a pricing pass); name the function from cg.a3-TAG.out vs cg.a3-base.out (callgrind_annotate --inclusive=yes).
5. Tests per crate: pixelflow-ir; pixelflow-search (default, --all-features, --features saturation-telemetry); pixelflow-codegen; pixelflow-compiler; pixelflow-graphics; pixelflow-pipeline; pixelflow-core.
6. \`cargo clippy --workspace --all-targets --all-features -- -D warnings\`; \`cargo fmt --all -- --check\`; \`RUSTDOCFLAGS="-D warnings" cargo doc -p pixelflow-ir -p pixelflow-search -p pixelflow-codegen --no-deps --document-private-items\`.
7. \`bash scripts/check-intent.sh\`; \`bash scripts/check-doc-paths.sh\`; \`bash scripts/check-cfg-encapsulation.sh\`; \`python3 -I ${A}/gen/checklist.py\` — 0 unclassified, 0 stale.
8. Greps (spec §11 as amended by the rulings), each with its hits and a verdict:
   a. B1: \`grep -rnwE 'A3' --include=*.rs pixelflow-*\` and \`grep -nw 'A3' pixelflow-ir/intent.md\`: every hit is true after A3 (none says a thing waits on A3 or is "until A3"; history like "since A3" is fine). Also the spec's phrase grep \`grep -rnE 'until A3|before A3|A3 lifts|A3 gives|A3 carries|A3 of|can be built until A3' --include=*.rs pixelflow-*\` is empty.
   b. \`grep -rn 'VALUE_BOUND' --include=*.rs\`: only fold.rs (definition, to_bits, its test), program/lower.rs (the refusal and its test), doc links.
   c. \`grep -rn 'KNOWN_RANGE' pixelflow-search/src/egraph/cost.rs pixelflow-search/src/egraph/extract.rs\`: at most a test-local import.
   d. \`grep -rn 'fn close_over\\|\\.close_over(\\|\\.close_fold(' --include=*.rs\`: every call takes FoldTerms.
   e. \`grep -rn 'fold_cost(' --include=*.rs\`: cost.rs and the extractor (ruling N3), every call with an explicit u64 trip count.
   f. \`grep -rn 'var_variance' pixelflow-search\`: empty. \`grep -n 'Variance::of_var\\|Self::ALL\\|Variance::ALL' pixelflow-ir/src/variance.rs\`: the Var arm is Variance::of_var.
   g. \`grep -n 'KnownBound' pixelflow-search/src/egraph/fold_rules.rs\`: definition, last entry of fold_rules(), tests.
   h. \`grep -rn 'real_sorted' --include=*.rs\`: empty (N10 renamed it takes_a_value_bound).
9. \`git diff --stat\`.`

const GATE_SCHEMA = { type: 'object', properties: {
  all_pass: { type: 'boolean' },
  results: { type: 'array', items: { type: 'object', properties: { gate: { type: 'string' }, command: { type: 'string' }, pass: { type: 'boolean' }, detail: { type: 'string' } }, required: ['gate', 'command', 'pass', 'detail'] } },
  ir: { type: 'string', description: 'total / saturate / extraction Ir, base vs head, with % deltas' },
  disk_free_end: { type: 'string' } },
  required: ['all_pass', 'results', 'ir', 'disk_free_end'] }
const FIND_SCHEMA = { type: 'object', properties: { findings: { type: 'array', items: { type: 'object', properties: {
  file: { type: 'string' }, line: { type: 'integer' }, claim: { type: 'string' }, evidence: { type: 'string' }, fix: { type: 'string' },
  blocking: { type: 'boolean', description: 'true for a byte/key/price/saturation change, an unsound value-bound path, a plan/spec/ruling violation, a broken invariant, a CLAUDE.md rule violation, a missed site or missing pinned test; false for taste' } },
  required: ['file', 'line', 'claim', 'evidence', 'fix', 'blocking'] } } }, required: ['findings'] }

phase('Base')
const base = await agent(`${RULES}

The tree must be clean at 3bfdaded (\`git status --short\` empty, \`git rev-parse HEAD\`); if not, stop and report. Two jobs, before anyone edits:

1. Compile-cost base: read ${G}/README.md, then \`bash ${G}/measure.sh a3-base\`. Report its JSON (total Ir, saturate and extraction inclusive Ir, RSS, wall).

2. The Reduce-site checklist (ruling B2). Build ${A}/gen/checklist.py (python3, stdlib only; adapt /home/user/scratch/bound-value/a2b/gen/checklist.py if useful) and ${A}/gen/classes.json. The grep: \`grep -rn 'Reduce {' --include=*.rs pixelflow-*/src pixelflow-*/examples pixelflow-*/tests pixelflow-pipeline/src/bin\`. Key each hit so line drift does not break it: (file, enclosing fn name, the hit line's trimmed text), with a per-key count. classes.json maps each key to {class, reason}, class one of: already-correct, changed-by-A3, left (reason), tests, n/a (doc/definition), A4, A5. Seed it from spec.md §5 (the classified sites) and critique.md §B2's table (the rest). The script prints UNCLASSIFIED (hits with no row) and STALE (rows with no hit) and exits non-zero if either is non-empty. Run it at base: it must report 0 and 0 — classify every hit honestly by reading it (does it walk only the body where it should take every child?); do not mark a site already-correct without reading it. Write ${A}/gen/README.md (5 lines: how to run, how to add a row).

Report: measure JSON; checklist hit count, class counts, and the run output; df at end.`, { label: 'base', phase: 'Base', model: 'sonnet' })
log('base: ' + String(base).slice(0, 300))

phase('Implement IR')
const ir = await agent(`${RULES}

${DESIGN}

You are implementer 1 of 3. Implement spec.md §11 steps 1-4 inside pixelflow-ir ONLY: fold.rs (§2: BoundEnd, BoundRefusal, Fold::count — pub(crate), N10's one n <= EXACT_BOUND function that both KnownRange::admits and Fold::count call — Bound::discover, ESTIMATED_TRIPS, priced_trips, FoldHeader::new accepting a value bound with its sort/stride checks, the VALUE_BOUND/KNOWN_RANGE texts), variance.rs (§3.1-3.2: of_var, the Var arm, L4, and N1's per-node transfer function shared by the forward pass and a rooted memoized walk), arena.rs (§3.3-3.5: free_index(es), push_reduce per N1/N4 — an out-of-domain or non-integer Const bound stays a Value bound, a Lit::Mask bound is BoundSort — FoldTerms, close_over with N2's discovery before the copy, close_fold, lowest_free_binder(roots), substitute_vars_with's precondition doc per N5), passes.rs's sibling, kernel.rs (§4: over's call, over_values pub(crate), refusing BoundArithmetic and, per N10, a Param/Buffer root with its own message; N9's Kernel::over doc fix), and pixelflow-ir's tests from §10 (fold.rs, variance.rs, passes.rs, kernel.rs, arena.rs unit tests, tests/value_bounds.rs). Also B1's rewording of every A3 mention in pixelflow-ir (fold.rs, passes.rs, intent.md) as A3 makes it true.
Other crates may stop compiling (implementers 2 and 3 do them), but pixelflow-ir must be green alone: \`cargo test -p pixelflow-ir\`, \`cargo clippy -p pixelflow-ir --all-targets --all-features -- -D warnings\`, \`cargo fmt -p pixelflow-ir -- --check\`. Update ${A}/gen/classes.json for the pixelflow-ir sites you add or change and run the checklist. Write ${A}/run/impl-ir.md: what you changed (by item), every judgement call with the rejected alternative, every place you departed from spec or rulings and why, and the public surface you added (also into ${A}/run/public-surface.md). Return that file's content.`, { label: 'impl-ir', phase: 'Implement IR', model: 'sonnet' })

phase('Implement search')
const search = await agent(`${RULES}

${DESIGN}

You are implementer 2 of 3. Implementer 1 finished pixelflow-ir (uncommitted; read ${A}/run/impl-ir.md and \`git diff -- pixelflow-ir\`). Implement spec.md §6-§7 and §11 step 5 in pixelflow-search: node.rs (L4 via Variance::of_var; var_variance gone; docs), graph.rs (EGraph::add's L3 backstop — skipped outright for a node with no value bound), insert.rs (value-bound children in children() order: body, then lo, then hi), fold_rules.rs (KnownBound per L8, appended LAST to fold_rules() so every RuleIdx is kept; L6 gating; FactorFold under a value bound per decision 5 — declines only a known empty range; HalveFold's rule-made bound arithmetic; substitute/rebuild per L7 and N5), cost.rs and extract.rs per ruling N3 (a Value bound whose child extracts to an in-domain integer literal is priced at that count, else ESTIMATED_TRIPS; the DP holds the bound child's pick before it prices the fold; fold_cost(fold, trips, body) and below take the trips; on an all-literal fold the visit order and every price must be exactly today's), and the tests §10 names for pixelflow-search, including N6's T-S-6 (ends by extracting with CostModel::latency_prior and asserts the arena) and N7's T-S-4b, N3's pin (a class holding both twins of a fold with n > 32 prices both at n), and tests/root_shapes.rs. N9: narrow pixelflow-search/src/egraph/intent.md's E6 restatement to known ranges. B1 rewording for every A3 mention in pixelflow-search. If pixelflow-ir needs a change, make it and note it.
Iterate with \`cargo check -p pixelflow-search --all-targets --all-features\`, then \`cargo test -p pixelflow-search\` (default, --all-features, --features saturation-telemetry), clippy -D warnings for the crate. Update ${A}/gen/classes.json for your sites. Write ${A}/run/impl-search.md (changes, judgement calls, departures) and append to ${A}/run/public-surface.md. Return the notes.`, { label: 'impl-search', phase: 'Implement search', model: 'sonnet' })

phase('Implement tail')
const tail = await agent(`${RULES}

${DESIGN}

You are implementer 3 of 3. pixelflow-ir and pixelflow-search are done (uncommitted; read ${A}/run/impl-ir.md, ${A}/run/impl-search.md, \`git diff\`). Do spec.md §11 steps 6-8:
- pixelflow-codegen: the program/lower.rs comment and T-C-1 (§8, §10); codegen still refuses a value bound by name.
- pixelflow-compiler: the mechanical tail (§9) — signature-only changes at its close_fold/close_over call sites; the arena it builds must not change.
- Every other crate the compiler finds broken (pixelflow-graphics, pixelflow-pipeline, pixelflow-core, pixelflow-runtime, pixelflow-ml): fix mechanically. N9: give KNOWN_RANGE a true reason at pixelflow-pipeline/examples/glyph_phase_split.rs (production glyphs before B2).
- Docs per spec §12 as amended by rulings N3 (plan §3.2/§4: pricing reads the bound children's extracted picks), N4 (plan §3.1's push_reduce sentence; the A4 bullet gains: the guard handles a rule-made out-of-domain literal), N5 (§3.4: substitute_vars_with's precondition, no L7 claim), N8 (A4 bullet's serializer list gains pixelflow-graphics/tests/production_glyph_arena_dump.rs; §10 says the L8/HalveFold pins are e-graph-level because the key refuses value bounds until A4). Plan A3 bullet gains **Done** with the facts you can state (fill the byte_probe fact only after gate 1 passes). BACKLOG row: landed gains A3, next A4.
- B1 rewording for every remaining A3 mention in pixelflow-*.
Then \`cargo check --workspace --all-targets --all-features\`, \`cargo test -p pixelflow-codegen\`, \`cargo test -p pixelflow-compiler\`, and run gates 1-3 below with TAG=impl. A difference is a behavior change on the literal path: find it and fix it (it may be in implementers 1 or 2's code). Update ${A}/gen/classes.json. Write ${A}/run/impl-tail.md (changes, judgement calls, gate results) and return it.

${GATES}`, { label: 'impl-tail', phase: 'Implement tail', model: 'sonnet' })

let round = 0, gates = null, open = []
const history = []
while (round < 3) {
  round++
  phase('Gates')
  gates = await agent(`${RULES}

You verify an uncommitted change (A3 of docs/language/2026-10-20-a-bound-is-a-value.md) that must be byte-identical on today's kernels. Do not edit source files. Run every gate with TAG=r${round}.

${FULL_GATES}`, { label: `gates:r${round}`, phase: 'Gates', model: 'sonnet', schema: GATE_SCHEMA })

  phase('Review')
  const lenses = [
    { key: 'bytes', text: `Lens: nothing moves on the literal path. Today every bound is a literal; find any place where A3 could change a byte, a key, a node id, a price, a rule-application count or an extraction choice for such a fold. Read specifically: (1) the extractor DP diff for evaluation-ORDER changes, not only arithmetic — holding the bound child's pick before pricing must not reorder visits or tie-breaks on a fold with no value-bound child; (2) close_over/close_fold/over: a literal Kernel::over must push the same nodes in the same order (no Const pushed then discarded; N2 says discovery happens before the copy); (3) push_reduce and EGraph::add: no L3 variance walk, no discovery work and no allocation when both bounds are Lit (also confirm via the Ir gate's numbers: ${G}/measure.a3-r${round}.txt vs ${G}/measure.a3-base.txt); (4) KnownBound appended last, returns before reading anything on a literal fold, so every RuleIdx and every application is unchanged; FactorFold's condition agrees with today's whenever the range is known; (5) variance: only placeholders change; (6) Display/dump/corpus/key formats unchanged.` },
    { key: 'semantics', text: `Lens: soundness of the new value-bound paths, which no byte gate exercises. Check against the plan's laws L2-L8 and §1.4: discovery (Bound::discover, Fold::count, the N10 shared n <= EXACT_BOUND function), domain refusal only at user surfaces with push_reduce keeping an out-of-domain Const as a Value bound (N4), BoundSort for non-{SUM,PRODUCT,MIN,MAX} and Lit::Mask, the stride rule, L2's normalization only for a pair made known by the read, L3 (bound reads no coordinate and not its own binder; N1's rooted memoized walk computes the same answer as the forward pass — look for a missed node kind, Ref handling, or a memo keyed wrongly), L4 variance union in both arena and e-graph, L6 rules gated on known ranges, L7 substitution into nested bounds always and into bodies unless rebound (and N5's precondition stated, not claimed as L7), L8 KnownBound correctness (the Lit twin is a true equal, monoid/binder/stride preserved, empty ranges), FactorFold under a value bound (decision 5: declines only a known empty range; is factoring sound when the trip count is unknown or zero?), HalveFold's rule-made bound arithmetic, insert/extract/witness walking value-bound children in body,lo,hi order, N3 pricing (in-domain integer literal pick → that count, else ESTIMATED_TRIPS; the twins pin). Write a small scratch test under ${A}/run/ only if you need to demonstrate a defect (never in the repo).` },
    { key: 'conformance', text: `Lens: completeness against spec.md + rulings.md (authoritative) + the plan. Go ruling by ruling — B1, B2, N1, N2, N3, N4, N5, N6, N7, N8, N9, N10 — and state for each whether it is done, with file:line evidence. Then spec §10's test list: every named pin exists and asserts what the spec says (T-IR-*, T-K-*, T-S-*, T-S-4b, T-S-6, T-C-1, tests/value_bounds.rs, the twins pin). Then §12's docs. Then the classification in ${A}/gen/classes.json: are the rows for new or changed sites honest (read the site)? ${A}/run/public-surface.md lists every new or changed pub item (diff it against \`git diff\`). Nothing of A4/A6 built (codegen, key, kernel! value bounds).` },
    { key: 'craft', text: `Lens: code quality per CLAUDE.md and the surrounding code. Rustdoc says what a thing means (denotation), not mechanism; no stale docs naming removed or renamed items (real_sorted, var_variance, old close_over/close_fold signatures, "until A3"); no visibility widened beyond the spec's list; no boolean args; functions under 4 args unless the spec names the exception; guard clauses over nested else; no dead code or unused helpers; no duplicated definition of a rule (one per-node variance transfer function, one n <= EXACT_BOUND check); comments in the file's register, no churn on untouched lines; nothing names a model; intent.md files ≤ 80 lines.` },
  ]
  const reviews = await parallel(lenses.map(l => () => agent(`${RULES}

Adversarial review of an uncommitted change (A3) in /home/user/pixelflow: \`git diff HEAD\` (HEAD is main 3bfdaded). Read-only: no edits to the repo; \`cargo check\` only if you need types. Report only defects you can show with evidence; an empty list is a good answer. Earlier rounds' findings and their outcome: ${JSON.stringify(history).slice(0, 6000)}

${DESIGN}

${l.text}`, { label: `review:${l.key}:r${round}`, phase: 'Review', schema: FIND_SCHEMA })))
  open = reviews.filter(Boolean).flatMap((r, i) => r.findings.map(f => ({ lens: lenses[i].key, ...f }))).filter(f => f.blocking)
  const fails = gates ? gates.results.filter(r => !r.pass) : [{ gate: 'gates', detail: 'gate agent returned nothing' }]
  log(`round ${round}: ${fails.length} gate failures, ${open.length} blocking findings`)
  if (fails.length === 0 && open.length === 0) break
  if (round === 3) break
  phase('Fix')
  const fix = await agent(`${RULES}

${DESIGN}

You fix an uncommitted change (A3). For EACH item: verify it against the code first; fix it if real, or say why not with evidence. Reproduce each gate failure, root-cause it, fix it. Then rerun the failing gates (TAG=fix${round}) and \`cargo test\` of the crates you touched, and the checklist.

${GATES}

Gate failures: ${JSON.stringify(fails)}
Blocking findings: ${JSON.stringify(open)}

Append to ${A}/run/fixes.md. Report per item: fixed (how) or rejected (why).`, { label: `fix:r${round}`, phase: 'Fix', model: 'sonnet' })
  history.push({ round, findings: open.map(f => f.lens + ': ' + f.claim), gate_failures: fails.map(f => f.gate), fix_report: String(fix).slice(0, 3000) })
}
return { base, ir, search, tail, rounds: round, gates, open_blocking: open, history }