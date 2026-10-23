export const meta = {
  name: 'a3-value-bounds-v2',
  description: 'A3: a fold bound may be a term in the IR and e-graph; byte-identical on today\'s kernels; gated, reviewed, WIP pushed',
  phases: [
    { title: 'Base', detail: 'Reduce-site checklist and rustdoc warning base at main' },
    { title: 'Implement IR', detail: 'pixelflow-ir: fold.rs, variance.rs, arena.rs, kernel.rs, passes.rs' },
    { title: 'Implement search', detail: 'pixelflow-search: node, graph, insert, fold_rules, cost, extract' },
    { title: 'Implement tail', detail: 'codegen, compiler, other callers, docs; first byte/key gates; push' },
    { title: 'Gates', detail: 'bytes, keys, saturation, Ir, tests, lint, docs, greps' },
    { title: 'Review', detail: 'four adversarial lenses' },
    { title: 'Fix', detail: 'verified findings and gate failures; push' },
  ],
}
const A = '/home/user/pixelflow/.wip/a3'
const G = '/home/user/pixelflow/.wip/gates'
const BR = 'claude/integral-area-work-ojiej6'
const RULES = `Ground rules (hard):
- Repository /home/user/pixelflow, branch ${BR}. The code base is main 3bfdaded; HEAD carries extra commits that touch only .wip/ (working notes) plus, as the run goes on, local "chore(wip)" commits of A3 itself. The A3 diff is always \`git diff 3bfdaded -- . ':!.wip'\`. Never build from a git worktree. Never stash/reset/rebase/amend; never checkout a tracked file (the gate scripts restore their own probe file).
- Commit or push only where your task says so. A commit is \`git add -A -- . ':!pixelflow-graphics/src/fonts/loop_blinn/program/tests.rs'\` only if that file is unmodified otherwise (the gate probe must never be committed), subject exactly as given, conventional format, ending with your harness's attribution trailer lines. \`git lfs ls-files | grep -c .\` must not rise (no LFS file: .wip/.gitignore excludes *.jsonl). Push: \`git push -q origin ${BR}\` (retry up to 4 times, 2/4/8/16 s, on a network error only).
- Never delete anything under /tmp/claude-0, /home/user/scratch or /home/user/pixelflow/.wip, including files you wrote: overwrite or use a new name.
- Never run \`cargo test --workspace\`; test crate by crate.
- \`df -h /\` before every build; under 2 GB free, stop and report.
- No model names or identifiers in any file you write, nor any path under .wip/ cited from a tracked file outside .wip/.
- Your notes go under ${A}/run/ (mkdir -p).`

const DESIGN = `## The design (authoritative, in this order)
1. ${A}/rulings.md — round 1 (B1, B2, N1-N10) and round 2 (J-acceptances, C2-C16, S1-S6). Where a ruling and the spec differ, the ruling wins; round 2 wins over round 1.
2. ${A}/spec.md — the A3 implementation spec at 3bfdaded (2,466 lines; its file:line references are at 3bfdaded — re-grep before trusting one).
3. ${A}/critique.soundness.md and ${A}/critique.completeness.md — the critiques round 2 rules on (read the finding a ruling names).
4. docs/language/2026-10-20-a-bound-is-a-value.md §1-§4, §6's A3 bullet, §8 — the plan.

Fixed constraints:
- BYTE-IDENTICAL on today's kernels, where every bound is a literal: byte_probe on both tiers, the 29 font programs' nodes/keys/digests, saturation telemetry including the extraction fields (spec §1.3, §7.5). A literal fold builds node-for-node the same arena, runs no L3 walk and no discovery allocation, and prices exactly as today.
- Not A3 (spec §0): codegen refuses a value bound by name (VALUE_BOUND in arena_to_schedule); FoldHeader::to_bits/from_bits keep refusing one; kernel! keeps literal bounds.
- Every intent.md stays within 80 lines (\`bash scripts/check-intent.sh\`).
- CLAUDE.md style: match the surrounding comment density and idiom; rustdoc on new items says what they mean; no boolean args; #[must_use] on new pub fns returning a value; no visibility widening beyond the spec. Keep ${A}/run/public-surface.md listing every new or changed pub item.
- The Reduce-site checklist ${A}/gen/checklist.py + ${A}/gen/classes.json (ruling C10). Every hit you create or change gets its row (class + one-line reason).`

const GATES = `## Byte/key gates (scripts in ${G}, read ${G}/README.md; each exits 0 only when identical to base; TAG given in the prompt)
1. \`bash ${G}/byteprobe.sh TAG\` (both tiers vs base).
2. \`bash ${G}/gate1b.sh TAG\` (29 programs' n/nodes/key/digest vs base; self-checks its probe).
3. \`bash ${G}/gate2.sh TAG && bash ${G}/gate2x.sh TAG\` (saturation tuple and extraction fields vs base).
After them \`git status --short pixelflow-graphics\` is empty (the probe file restored).`

const FULL_GATES = `${GATES}
4. Compile cost: \`bash ${G}/measure.sh TAG\`; read out/measure.TAG.txt's vs_base. Pass = total Ir within +0.5% of base. A larger rise on all-literal kernels means work runs on the literal path that should not; name the function by diffing \`callgrind_annotate --inclusive=yes\` of out/cg.TAG.out vs out/cg.base.out.
5. Tests per crate: pixelflow-ir; pixelflow-search (default, --all-features, --features saturation-telemetry); pixelflow-codegen; pixelflow-compiler; pixelflow-graphics; pixelflow-pipeline; pixelflow-core; pixelflow-runtime.
6. \`cargo clippy --workspace --all-targets --all-features -- -D warnings\`; \`cargo fmt --all -- --check\`; \`RUSTFLAGS="-D warnings" cargo check -p pixelflow-ir --no-default-features\`; \`RUSTFLAGS="-D warnings" cargo check -p pixelflow-search --no-default-features\`; \`RUSTDOCFLAGS="-D warnings" cargo doc -p pixelflow-codegen --no-deps --document-private-items\`; for pixelflow-ir and pixelflow-search, the rustdoc warning set compared with ${A}/run/rustdoc-base.{pixelflow-ir,pixelflow-search}.txt by the method the Base phase wrote in ${A}/run/rustdoc-base.md: no new warning (ruling C4).
7. \`bash scripts/check-intent.sh\`; \`bash scripts/check-doc-paths.sh\`; \`bash scripts/check-cfg-encapsulation.sh\`; \`bash scripts/check-feature-matrix.sh\` if it runs locally in under 15 minutes (else say so); \`python3 -I ${A}/gen/checklist.py\` — 0 unclassified, 0 stale.
8. spec §11's greps as amended by the rulings, each with its hits and a verdict. Including: \`grep -rnwE 'A3' --include=*.rs pixelflow-*\` and \`grep -nw 'A3' pixelflow-ir/intent.md\` EMPTY (B1, round 2); \`grep -rn 'fold_cost(' --include=*.rs\` only cost.rs (round 2 J7); \`grep -rn 'real_sorted\\|var_variance' --include=*.rs\` empty; \`grep -rn 'VALUE_BOUND' --include=*.rs\` only fold.rs and program/lower.rs and doc links; \`grep -rln '\\.wip' --include=*.rs --include=*.md . | grep -v '^./.wip/'\` empty.
9. \`git diff --stat 3bfdaded -- . ':!.wip'\`.`

const GATE_SCHEMA = { type: 'object', properties: {
  all_pass: { type: 'boolean' },
  results: { type: 'array', items: { type: 'object', properties: { gate: { type: 'string' }, command: { type: 'string' }, pass: { type: 'boolean' }, detail: { type: 'string' } }, required: ['gate', 'command', 'pass', 'detail'] } },
  ir: { type: 'string', description: 'total / saturate / extraction Ir, base vs head, with % deltas' },
  disk_free_end: { type: 'string' } },
  required: ['all_pass', 'results', 'ir', 'disk_free_end'] }
const FIND_SCHEMA = { type: 'object', properties: { findings: { type: 'array', items: { type: 'object', properties: {
  file: { type: 'string' }, line: { type: 'integer' }, claim: { type: 'string' }, evidence: { type: 'string' }, fix: { type: 'string' },
  blocking: { type: 'boolean', description: 'true for a byte/key/price/saturation change, an unsound value-bound path, a plan/spec/ruling violation, a broken invariant, a CLAUDE.md rule violation, a missed site or a missing pinned test; false for taste' } },
  required: ['file', 'line', 'claim', 'evidence', 'fix', 'blocking'] } } }, required: ['findings'] }

phase('Base')
const base = await agent(`${RULES}

The code must be main's: \`git diff --quiet 3bfdaded -- . ':!.wip'\` and \`git status --short -- . ':!.wip'\` empty; if not, stop and report. Two jobs before anyone edits code (compile-cost and byte bases already exist in ${G}/out/*.base.*; do not re-record them):

1. The Reduce-site checklist (rulings B2, C10). Build ${A}/gen/checklist.py (python3 -I, stdlib only) and ${A}/gen/classes.json from spec.md §5's table. The grep: \`grep -rn 'Reduce {' --include=*.rs pixelflow-*/src pixelflow-*/examples pixelflow-*/tests pixelflow-pipeline/src/bin\`. Key each hit by (file, enclosing fn name, the hit line's trimmed text), with a per-key count, so line drift does not break it. A row: {class, reason}; class one of already-correct, changed-by-A3, left, tests, n/a, A4, A5. The script prints UNCLASSIFIED (hits with no row) and STALE (rows with no hit) and exits non-zero if either is non-empty. At base it must report 0 and 0; if spec §5's table disagrees with a hit you read, classify it by reading it and note the disagreement. Rows for sites A3 will rewrite (spec §5's list plus arena.rs:1757, extract.rs:1684, insert.rs:173) are keyed by their base text now; implementers update them. Write ${A}/gen/README.md (how to run, how to add a row).
2. The rustdoc base (ruling C4): \`cargo doc -p pixelflow-ir --no-deps --document-private-items 2>&1\` and the same for pixelflow-search. Normalize each warning to (file, message) without line numbers into ${A}/run/rustdoc-base.pixelflow-ir.txt and ${A}/run/rustdoc-base.pixelflow-search.txt (sorted, with counts), and write ${A}/run/rustdoc-base.md with the exact normalization command so a later run compares the same way. Also confirm \`RUSTDOCFLAGS="-D warnings" cargo doc -p pixelflow-codegen --no-deps --document-private-items\` passes at base.

Then commit (subject "chore(wip): A3 Reduce-site checklist and rustdoc base (not for merge)") and push. Report: checklist hit count, class counts, the run output, disagreements with §5, rustdoc warning counts, the commit sha.`, { label: 'base', phase: 'Base', model: 'sonnet' })
log('base: ' + String(base).slice(0, 300))

phase('Implement IR')
const ir = await agent(`${RULES}

${DESIGN}

You are implementer 1 of 3. Implement spec.md §11's order of work, steps for pixelflow-ir ONLY: fold.rs (§2), variance.rs and arena.rs (§3), kernel.rs (§4), passes.rs's sibling and every pixelflow-ir site §5 classifies changed-by-A3, pixelflow-ir's tests from §10 (unit tests and tests/value_bounds.rs), and B1's rewording of every A3 mention in pixelflow-ir (fold.rs, passes.rs, arena.rs, intent.md). Apply round 2: C2 (expect(dead_code) on over_values and user_bound), C3, C5, C8, J5's doc, J13, S1/S2 docs where they land in pixelflow-ir.
Other crates may stop compiling (implementers 2 and 3 do them), but pixelflow-ir must be green alone: \`cargo test -p pixelflow-ir\`, \`cargo clippy -p pixelflow-ir --all-targets --all-features -- -D warnings\`, \`RUSTFLAGS="-D warnings" cargo check -p pixelflow-ir --no-default-features\`, \`cargo fmt -p pixelflow-ir -- --check\`. Update ${A}/gen/classes.json for the sites you change and run the checklist (pixelflow-ir rows clean). Write ${A}/run/impl-ir.md: what you changed by item, every judgement call with the rejected alternative, every departure from spec or rulings and why, the public surface (also into ${A}/run/public-surface.md). Commit locally — do NOT push (the workspace does not compile yet) — subject "chore(wip): A3 in pixelflow-ir (not for merge)". Return impl-ir.md's content.`, { label: 'impl-ir', phase: 'Implement IR', model: 'sonnet' })

phase('Implement search')
const search = await agent(`${RULES}

${DESIGN}

You are implementer 2 of 3. Implementer 1 finished pixelflow-ir (committed locally; read ${A}/run/impl-ir.md and \`git diff 3bfdaded -- pixelflow-ir\`). Implement spec.md §6-§7 and §11's pixelflow-search steps: node.rs, graph.rs (EGraph::add's backstop, skipped outright for a node with no value bound), insert.rs, fold_rules.rs (KnownBound appended LAST; FactorFold under decision 5 with S6's doc; HalveFold; substitute/rebuild; module doc per C15), cost.rs and extract.rs per round 2's J7 (literal twin through node_cost and below; fold_cost stays in cost.rs; §7.5's identity on literal folds; C1's "before value bounds"; C11's imports), intent.md per N9, telemetry/anytime untouched (S3 is a doc row only), and pixelflow-search's §10 tests including T-S-4b, T-S-6 with C7's existence assertion, the twins pin, T-S-15 for root_shapes.rs. B1 rewording for every A3 mention in pixelflow-search (insert.rs:521 per C9). If pixelflow-ir needs a change, make it and note it.
Green: \`cargo test -p pixelflow-search\` (default, --all-features, --features saturation-telemetry), \`cargo clippy -p pixelflow-search --all-targets --all-features -- -D warnings\`, \`RUSTFLAGS="-D warnings" cargo check -p pixelflow-search --no-default-features\`, fmt. Update ${A}/gen/classes.json; write ${A}/run/impl-search.md (changes, judgement calls, departures); append to ${A}/run/public-surface.md. Commit locally, do NOT push, subject "chore(wip): A3 in pixelflow-search (not for merge)". Return the notes.`, { label: 'impl-search', phase: 'Implement search', model: 'sonnet' })

phase('Implement tail')
const tail = await agent(`${RULES}

${DESIGN}

You are implementer 3 of 3. pixelflow-ir and pixelflow-search are done (committed locally; read ${A}/run/impl-ir.md, ${A}/run/impl-search.md, \`git diff 3bfdaded -- . ':!.wip'\`). Do spec.md §8, §9, §12 and the rest of §11:
- pixelflow-codegen: §8 (the lower.rs comment and T-C-1); codegen still refuses a value bound by name.
- pixelflow-compiler: §9's mechanical tail; the arena it builds must not change.
- Every other crate the compiler finds broken (pixelflow-graphics, pixelflow-pipeline, pixelflow-core, pixelflow-runtime, pixelflow-ml, core-term): fix mechanically. N9's glyph_phase_split.rs reason.
- Docs per §12 with rounds 1-2: the plan (N3 as amended by J7, N4 as amended by J13, N5, N8, C14's B3 sentence, C16 — facts, no J-labels, no .wip path), the-arena-is-typed rows, BACKLOG (landed gains A3, next A4), ontology "Unrolling", intent.md files within 80 lines. The plan's A3 bullet gains **Done** with facts (fill the byte facts only after gates 1-3 pass).
Then \`cargo check --workspace --all-targets --all-features\`, \`cargo test -p pixelflow-codegen\`, \`cargo test -p pixelflow-compiler\`, and gates 1-3 below with TAG=a3-impl. A difference is a behavior change on the literal path: find it and fix it (it may be in implementer 1's or 2's code). Update ${A}/gen/classes.json. Write ${A}/run/impl-tail.md (changes, judgement calls, gate results). When the workspace compiles and gates 1-3 pass, commit (subject "chore(wip): A3 tail, docs and first gates (not for merge)") and push — this is the first push of A3's code. If gates 1-3 cannot be made to pass, commit locally without pushing and say why. Return impl-tail.md's content and the commit sha.

${GATES}`, { label: 'impl-tail', phase: 'Implement tail', model: 'sonnet' })

let round = 0, gates = null, open = []
const history = []
while (round < 3) {
  round++
  phase('Gates')
  gates = await agent(`${RULES}

You verify A3 of docs/language/2026-10-20-a-bound-is-a-value.md, which must be byte-identical on today's kernels. Do not edit tracked files; outputs go in .wip. Run every gate with TAG=a3-r${round}. Then commit the gate outputs only (\`git add .wip\`; subject "chore(wip): A3 gate outputs, round ${round} (not for merge)") and push.

${FULL_GATES}`, { label: `gates:r${round}`, phase: 'Gates', model: 'sonnet', schema: GATE_SCHEMA })

  phase('Review')
  const lenses = [
    { key: 'bytes', text: `Lens: nothing moves on the literal path. Today every bound is a literal; find any place A3 could change a byte, a key, a node id, a price, a rule-application count or an extraction choice for such a fold (spec §1.3 and §7.5 argue there is none: check the code, not the argument). Read specifically: (1) the extractor diff for evaluation-ORDER and tie-break changes, not only arithmetic; the literal twin must return before any work when the fold has no value-bound child; (2) close_over/close_fold/over: a literal Kernel::over pushes the same nodes in the same order (no Const pushed then discarded; discovery before the copy, N2); (3) push_reduce and EGraph::add do no L3 walk, no discovery allocation and no extra pass when both bounds are Lit — confirm with the Ir gate (${G}/out/measure.a3-r${round}.txt vs measure.base.txt); (4) KnownBound appended last and returning before reading anything on a literal fold, so every RuleIdx and application count holds; FactorFold's guard agrees with today's on known ranges; (5) variance changes only placeholders; (6) Display/dump/corpus/key formats unchanged.` },
    { key: 'semantics', text: `Lens: soundness of the new value-bound paths, which no byte gate exercises. Check against the plan's laws L2-L8 and §1.4 and the spec's §1: discovery (Bound::discover, Fold::count and N10's one n <= EXACT_BOUND function), N4 as amended (push_reduce keeps an out-of-domain or non-integer Const as a Value bound; BoundOutOfDomain at over_values; Kernel::over with KnownRange::new's message), BoundSort for non-{SUM,PRODUCT,MIN,MAX} and Lit::Mask literals, the stride rule, L2's normalization only for a pair made known by the read, L3 via N1's rooted memoized walk (same answer as the forward pass; Ref handling; memo keyed right; placeholders CONST per J5 and re-checked at close), L4 in arena and e-graph, L6 gating, L7 substitution and N5's precondition, L8 KnownBound (a true equality; monoid, binder, stride kept; reversed discovered pair → empty), FactorFold under decision 5, HalveFold's rule-made bound, insert/extract walking value-bound children in body, lo, hi order, J7 literal-twin pricing (an in-domain integer pick → that count, else ESTIMATED_TRIPS; the twins pin). The holes S1, S2, C14 are accepted as documented — check the docs say them exactly; do not report them as defects. Demonstrate a defect with a scratch test under ${A}/run/ only (never in the repo).` },
    { key: 'conformance', text: `Lens: completeness against spec.md + rulings.md (both rounds; authoritative) + the plan. Go ruling by ruling — B1, B2, N1-N10, then round 2's J7, J5, J12, J3, J13, C2-C16, S1-S6 — and state for each: done, with file:line evidence, or not. Then spec §10's test list: every named pin exists and asserts what the spec and C7/C8 say. Then §12's docs and C15/C16. Then ${A}/gen/classes.json: are the rows for new or changed sites honest (read the site)? ${A}/run/public-surface.md lists every new or changed pub item (check against the diff). Nothing of A4/A6 is built.` },
    { key: 'craft', text: `Lens: code quality per CLAUDE.md and the surrounding code. Rustdoc says what a thing means (denotation), not mechanism; no stale docs naming removed or renamed items (real_sorted, var_variance, old close_over/close_fold signatures, "until A3", fold_cost(fold, range, 0)); no broken or private intra-doc links (C5); no visibility widened beyond the spec; no boolean args; functions under 4 args unless the spec names the exception; guard clauses over nested else; no dead code or unused helpers beyond C2's expected pair; one definition per rule (one per-node variance transfer, one n <= EXACT_BOUND check); comments in the file's register, no churn on untouched lines; nothing names a model or a .wip path; intent.md files ≤ 80 lines.` },
  ]
  const reviews = await parallel(lenses.map(l => () => agent(`${RULES}

Adversarial review of A3 in /home/user/pixelflow: \`git diff 3bfdaded -- . ':!.wip'\`. Read-only: no edits to tracked files, no commits; \`cargo check\` only if you need types. Report only defects you can show with evidence; an empty list is a good answer. Earlier rounds' findings and their outcome: ${JSON.stringify(history).slice(0, 8000)}

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

You fix A3. For EACH item: verify it against the code first; fix it if real, or say why not with evidence. Reproduce each gate failure, root-cause it, fix it. Then rerun gates 1-3 below (TAG=a3-fix${round}), \`cargo test\` of every crate you touched, clippy -D warnings for them, fmt, and the checklist.

${GATES}

Gate failures: ${JSON.stringify(fails)}
Blocking findings: ${JSON.stringify(open)}

Append to ${A}/run/fixes.md (per item: fixed how, or rejected why). Then commit (subject "chore(wip): A3 fixes, round ${round} (not for merge)") and push. Return the per-item report.`, { label: `fix:r${round}`, phase: 'Fix', model: 'sonnet' })
  history.push({ round, findings: open.map(f => f.lens + ': ' + f.claim), gate_failures: fails.map(f => f.gate + ': ' + String(f.detail).slice(0, 200)), fix_report: String(fix).slice(0, 3000) })
}
return { base, ir, search, tail, rounds: round, gates, open_blocking: open, history }