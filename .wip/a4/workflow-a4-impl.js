export const meta = {
  name: 'a4-value-bounds-codegen',
  description: 'A4: value bounds in codegen and the key; byte-identical on today\'s kernels; per-tier JIT tests; gated, reviewed, WIP pushed',
  phases: [
    { title: 'Implement IR', detail: 'key marker, guard pass, over_values public' },
    { title: 'Implement codegen', detail: 'lowering asserts, operands, Trips, EmitTraffic, emitters, JIT tests' },
    { title: 'Implement tail', detail: 'serializers, corpus versions, search docs and pin, docs; first gates; push' },
    { title: 'Gates', detail: 'bytes, keys, saturation, Ir, tests per tier, lint, docs, greps' },
    { title: 'Review', detail: 'five adversarial lenses' },
    { title: 'Fix', detail: 'verified findings and gate failures; push' },
  ],
}
const A = '/home/user/pixelflow/.wip/a4'
const G = '/home/user/pixelflow/.wip/gates'
const BR = 'claude/integral-area-work-ojiej6'
const RULES = `Ground rules (hard):
- Repository /home/user/pixelflow, branch ${BR}. The code base is main a747770; the A4 diff is \`git diff a747770 -- . ':!.wip'\` (HEAD also carries chore(wip) commits; .wip/ holds working notes). Never build from a git worktree. Never stash/reset/rebase/amend; never checkout a tracked file (the gate scripts restore their own probe file).
- Commit or push only where your task says so: \`git add -A -- . ':!pixelflow-graphics/src/fonts/loop_blinn/program/tests.rs'\` only when that file is otherwise unmodified; never add a Cargo.toml or crate under .wip (scratch crates live in /home/user/scratch); \`git lfs ls-files | grep -c .\` stays 68; subject exactly as given, ending with your harness's attribution trailer lines; push \`git push -q origin ${BR}\` (retry 4 times 2/4/8/16 s on a network error only).
- Never delete anything under /tmp/claude-0, /home/user/scratch or /home/user/pixelflow/.wip.
- Never \`cargo test --workspace\`; test crate by crate. \`df -h /\` before builds; under 2 GB free, stop.
- No model names; no tracked file outside .wip cites a .wip path or a ruling label.
- Clippy is CI's: \`cargo +1.99.0 clippy ...\` (installed; a separate CARGO_TARGET_DIR like /home/user/scratch/a3-close1/t199 avoids thrashing the default target). Builds, tests and measure.sh use the default toolchain. For scripts/check-feature-matrix.sh, prepend /home/user/scratch/cargo-hack-root/bin to PATH.
- Your notes go under ${A}/run/ (mkdir -p).`

const DESIGN = `## The design (authoritative, in this order)
1. ${A}/rulings.md — rulings on the spec and its three critiques (wins over the spec).
2. ${A}/spec.md — the A4 implementation spec at a747770 (1,286 lines; re-grep before trusting a line number).
3. ${A}/critique.semantics.md, ${A}/critique.completeness.md, ${A}/critique.emitter.md — read the finding a ruling names.
4. docs/language/2026-10-20-a-bound-is-a-value.md (the plan), CLAUDE.md ("If guards its arms", "Floating point at the edges", "The language is a DAG").

Fixed constraints:
- BYTE-IDENTICAL on today's kernels (every bound a literal): byte_probe both tiers, the 29 font programs' nodes/keys/digests, saturation telemetry incl. extraction fields; total Ir within +0.5% of the a747770 base; a literal fold does no new work.
- A value-bounded fold computes its fold or NaN on every tier, never hangs, never converts a trip count to an integer, never reads past a buffer.
- The extraction-side hole stays loud (ruling (a)): no extraction change.
- Every intent.md within 80 lines; CLAUDE.md style (match comment density and idiom, rustdoc says what a thing means, no boolean args, #[must_use] on new pub fns returning a value, no visibility widening beyond the spec). Keep ${A}/run/public-surface.md listing every new or changed pub item.`

const GATES = `## Byte/key gates (scripts in ${G}; README there; each exits 0 only when identical to base; TAG given)
1. \`bash ${G}/byteprobe.sh TAG\`; 2. \`bash ${G}/gate1b.sh TAG\`; 3. \`bash ${G}/gate2.sh TAG && bash ${G}/gate2x.sh TAG\`; then \`git status --short pixelflow-graphics\` empty. (If a gate exits 3 on .build.lock, an sccache daemon inherited fd 9: \`sccache --stop-server\` then start it from a shell without fd 9 — README.)`

const FULL_GATES = `${GATES}
4. \`bash ${G}/measure.sh TAG\`: total Ir within +0.5% of base (out/measure.TAG.txt vs_base).
5. Tests per crate: pixelflow-ir; pixelflow-search (default, --all-features, --features saturation-telemetry); pixelflow-codegen under \`PIXELFLOW_ISA=avx2\` AND \`PIXELFLOW_ISA=avx512\` (the new JIT tests must run on both; report their names and counts per tier); pixelflow-compiler; pixelflow-graphics; pixelflow-pipeline; pixelflow-core; pixelflow-runtime. Also \`cargo run -q -p xtask -- isa-matrix --smoke\` if it runs locally.
6. \`cargo +1.99.0 clippy --workspace --all-targets --all-features -- -D warnings\`; \`cargo fmt --all -- --check\`; \`RUSTFLAGS="-D warnings" cargo check -p pixelflow-ir --no-default-features\` and the same for pixelflow-search; \`RUSTDOCFLAGS="-D warnings" cargo doc -p pixelflow-codegen --no-deps --document-private-items\`; ir/search rustdoc warning sets compared with ${A}/rustdoc-base.{pixelflow-ir,pixelflow-search}.txt via \`python3 -I ${A}/rustdoc_norm.py --compare\` (no new warning).
7. \`bash scripts/check-intent.sh\`, \`check-doc-paths.sh\`, \`check-cfg-encapsulation.sh\`, \`check-bin-declarations.sh\`, \`check-emit-boundary.sh\`, \`check-feature-matrix.sh\`.
8. spec §9's greps as amended by the rulings (C3, C4, C12), each with hits and verdict; \`grep -rnwE 'A4' --include=*.rs pixelflow-*\`: every hit true after A4 (nothing "until A4"); \`grep -rln '\\.wip' --include=*.rs --include=*.md . | grep -v '^./.wip/'\` empty.
9. \`git diff --stat a747770 -- . ':!.wip'\`.`

const GATE_SCHEMA = { type: 'object', properties: { all_pass: { type: 'boolean' },
  results: { type: 'array', items: { type: 'object', properties: { gate: { type: 'string' }, command: { type: 'string' }, pass: { type: 'boolean' }, detail: { type: 'string' } }, required: ['gate', 'command', 'pass', 'detail'] } },
  ir: { type: 'string' } }, required: ['all_pass', 'results', 'ir'] }
const FIND_SCHEMA = { type: 'object', properties: { findings: { type: 'array', items: { type: 'object', properties: {
  file: { type: 'string' }, line: { type: 'integer' }, claim: { type: 'string' }, evidence: { type: 'string' }, fix: { type: 'string' },
  blocking: { type: 'boolean', description: 'true for wrong code/values on any tier, a hang, a byte/key/price/saturation change on today\'s kernels, a plan/spec/ruling violation, a broken invariant, a false doc, a CLAUDE.md rule violation, a missed site or missing pinned test; false for taste' } },
  required: ['file', 'line', 'claim', 'evidence', 'fix', 'blocking'] } } }, required: ['findings'] }

phase('Implement IR')
const ir = await agent(`${RULES}

${DESIGN}

You are implementer 1 of 3. Implement the spec's pixelflow-ir part: the key's Value marker (\`FoldHeader::to_bits\`/\`from_bits\`, \`admits_a_value\`, the const assert), the guard pass (\`passes/guard.rs\`, last in \`passes::legalize\`), \`Kernel::over_values\` public with its doc (F5) and the user surface (drop the \`expect(dead_code)\` pair), the VALUE_BOUND/KNOWN_RANGE texts as the spec says, docs under changed code, and pixelflow-ir's §8 tests (C1's key assertions). pixelflow-ir must be green alone: \`cargo test -p pixelflow-ir\`, \`cargo +1.99.0 clippy -p pixelflow-ir --all-targets --all-features -- -D warnings\`, the --no-default-features check, fmt. Other crates may break. Write ${A}/run/impl-ir.md (changes, judgement calls with rejected alternatives, departures) and ${A}/run/public-surface.md. Commit locally, do NOT push, subject "chore(wip): A4 in pixelflow-ir (not for merge)". Return the notes.`, { label: 'impl-ir', phase: 'Implement IR', model: 'sonnet' })

phase('Implement codegen')
const cg = await agent(`${RULES}

${DESIGN}

You are implementer 2 of 3. pixelflow-ir is done (committed locally; read ${A}/run/impl-ir.md and \`git diff a747770 -- pixelflow-ir\`). Implement the spec's pixelflow-codegen part: lowering (the VALUE_BOUND refusal goes; P1; P2 counting readers per F1/E2; E1's mask-valued-bound refusal through the one predicate the emitter uses, or the documented fallback), the bounds as operands and the carve (§1.6, §2, §3), \`Trips\` and its Display, \`dynamic_memory_ops\`, EmitTraffic, every per-ISA emitter change (x86-64 AVX2, AVX-512, aarch64 NEON — the aarch64 code must compile and be correct even though this host cannot run it: \`cargo check -p pixelflow-codegen --target aarch64-unknown-linux-gnu\` if the target is installed, else say so), the §3.4 cells corrected per E3/E4, and the §8 tests: the plan's JIT gate list in pixelflow-codegen/tests/ against scalar Rust (with F3's n=0 identities and the runner named), F4's \`Σ_{i<8} Σ_{j∈[i,8)} (X + j)\`, the unit tests (C7/E5's Write via \`Ir::embed\`), C9's shapes. Run \`PIXELFLOW_ISA=avx2 cargo test -p pixelflow-codegen\` and \`PIXELFLOW_ISA=avx512 cargo test -p pixelflow-codegen\`, clippy 1.99 on the crate, fmt. Write ${A}/run/impl-codegen.md; append to public-surface.md. Commit locally, do NOT push, subject "chore(wip): A4 in pixelflow-codegen (not for merge)". Return the notes.`, { label: 'impl-codegen', phase: 'Implement codegen', model: 'sonnet' })

phase('Implement tail')
const tail = await agent(`${RULES}

${DESIGN}

You are implementer 3 of 3. pixelflow-ir and pixelflow-codegen are done (committed locally; read ${A}/run/impl-ir.md, ${A}/run/impl-codegen.md, the diff). Do the rest of the spec: the serializers (§4: pixelflow-graphics render/cell_grid.rs dump_arena, tests/production_glyph_arena_dump.rs dump_arena, src/test_support.rs program_digest; pixelflow-pipeline collapse_bench/corpus.rs to v5 with C2's round-trip test, bin/corpus_gaps.rs with C8's new test module, the training corpus schema text; pixelflow-compiler arena_to_tokens), the search docs (C5) and ruling (a)'s pin plus the egraph intent.md line, every other crate the compiler finds broken, and the docs (§10 with the rulings: the plan's A4 Done line in facts and ruling (a)'s sentence, §5 rows marked A4, CLAUDE.md and DENOTATION rows the plan assigns A4, the ontology rows, intent.md files within 80 lines, the BACKLOG row ruling (a) names, C6). Then \`cargo check --workspace --all-targets --all-features\`, tests of the crates you touched, and gates 1-3 below with TAG=a4-impl; a difference is a behavior change on the literal path — find and fix it (anywhere). Write ${A}/run/impl-tail.md. When the workspace compiles and gates 1-3 pass, commit (subject "chore(wip): A4 tail, docs and first gates (not for merge)") and push; else commit locally and say why. Return the notes and the sha.

${GATES}`, { label: 'impl-tail', phase: 'Implement tail', model: 'sonnet' })

let round = 0, gates = null, open = []
const history = []
while (round < 3) {
  round++
  phase('Gates')
  gates = await agent(`${RULES}

Verify A4 of docs/language/2026-10-20-a-bound-is-a-value.md (byte-identical on today's kernels). Do not edit tracked files. Run every gate with TAG=a4-r${round}; then commit the outputs only (\`git add .wip\`, subject "chore(wip): A4 gate outputs, round ${round} (not for merge)") and push.

${FULL_GATES}`, { label: `gates:r${round}`, phase: 'Gates', model: 'sonnet', schema: GATE_SCHEMA })

  phase('Review')
  const lenses = [
    { key: 'bytes', text: `Lens: nothing moves on the literal path. Find any place A4 could change a byte, key, node id, price, rule-application count or extraction choice for an all-literal kernel: the guard pass must be a no-op on a fold with no value bound (no node pushed, no rebuild); the key's literal encoding and every serializer's literal output must be unchanged bit for bit; lowering's new asserts and the operands change must add no def, no register, no frame slot for a literal fold; Trips' Known path prints and counts as before. Check the Ir gate (${G}/out/measure.a4-r${round}.txt vs base).` },
    { key: 'semantics', text: `Lens: run-time correctness of value-bounded folds on every tier. For each bound class (0, -0.0, 1, 7, 1000, 2^24, a reversed pair, 2.5, -1, NaN, ±inf, 2^24+2, a value lo reading an enclosing index, triangular), trace the guard pass's output and the emitted loop: the fold's value or NaN, no hang (the f32 counter stalls at 2^24; NaN under FCMGE never fires), no integer conversion, no read past a buffer. Is the guard's mask uniform per batch by L3 so "If guards its arms" never runs the loop with a garbage count? Does P2 now count readers and refuse a second? Is E1's mask refusal exact (one definition) or is the fallback documented with the divergence pinned? Are the JIT tests' oracles right (F3's identities)? Do the tests actually run on both x86 tiers? Construct counterexamples in a scratch crate under /home/user/scratch if needed.` },
    { key: 'emitter', text: `Lens: codegen mechanics. Check the diff in pixelflow-codegen's program/ and emit/ against the spec §3 and the rulings E3/E4: the bounds as operands reach the carve, ownership, assert_closed and liveness as claimed; hi is live to the latch; the seed path for a value lo (LinearScan::pass → home → copy_of) is exercised by F4's test; per-ISA compare/branch encodings and immediate limits; the aarch64 fixed-size skip; frame and slot accounting; EmitTraffic's counts for a value-bounded fold (per-trip reload of an uncarried hi). The aarch64 emitter cannot run here: read it with extra care.` },
    { key: 'conformance', text: `Lens: completeness against spec.md + rulings.md (authoritative) + the plan's A4 bullet and gate list. Go ruling by ruling and state done/not with file:line. Every test the plan's A4 gate names exists with a correct assertion and runs on avx2 and avx512. Every serializer the plan lists learns the marker; corpus versions bumped as ruled. Docs per §10 and the rulings (plan Done line in facts; ruling (a) sentence; BACKLOG row; ontology/DENOTATION/CLAUDE.md rows). public-surface.md lists every new or changed pub item. Nothing of A5-A7 or B built.` },
    { key: 'craft', text: `Lens: code quality per CLAUDE.md. Rustdoc says what a thing means; no stale docs (VALUE_BOUND's old meaning, "until A4", "no bits", over_values' dead_code note); no boolean args; functions under 4 args; guard clauses; one definition per rule (the mask predicate, the domain test, the marker); platform cfg only at file/module seams (check-cfg-encapsulation); no dead code; comments in the file's register; no churn on untouched lines; nothing names a model or a .wip path; intent.md ≤ 80 lines.` },
  ]
  const reviews = await parallel(lenses.map(l => () => agent(`${RULES}

Adversarial review of A4 in /home/user/pixelflow: \`git diff a747770 -- . ':!.wip'\`. Read-only on tracked files; no commits. Report only defects you can show with evidence; an empty list is a good answer. Earlier rounds: ${JSON.stringify(history).slice(0, 8000)}

${DESIGN}

${l.text}`, { label: `review:${l.key}:r${round}`, phase: 'Review', schema: FIND_SCHEMA })))
  open = reviews.filter(Boolean).flatMap((r, i) => r.findings.map(f => ({ lens: lenses[i].key, ...f }))).filter(f => f.blocking)
  const nonblocking = reviews.filter(Boolean).flatMap((r, i) => r.findings.map(f => ({ lens: lenses[i].key, ...f }))).filter(f => !f.blocking)
  const fails = gates ? gates.results.filter(r => !r.pass) : [{ gate: 'gates', detail: 'gate agent returned nothing' }]
  log(`round ${round}: ${fails.length} gate failures, ${open.length} blocking, ${nonblocking.length} non-blocking`)
  history.push({ round, findings: open.map(f => f.lens + ': ' + f.claim), nonblocking: nonblocking.map(f => f.lens + ' ' + f.file + ':' + f.line + ' ' + f.claim), gate_failures: fails.map(f => f.gate + ': ' + String(f.detail).slice(0, 200)) })
  if (fails.length === 0 && open.length === 0) break
  if (round === 3) break
  phase('Fix')
  const fix = await agent(`${RULES}

${DESIGN}

You fix A4. For EACH item: verify it against the code first; fix it if real, or say why not with evidence. Reproduce each gate failure, root-cause it, fix it. Also apply the plainly correct non-blocking findings that are defects in A4's own diff (false docs, vacuous tests, wrong messages); skip taste. Then rerun gates 1-3 (TAG=a4-fix${round}), \`cargo test\` of every crate you touched (pixelflow-codegen under both PIXELFLOW_ISA=avx2 and avx512), clippy 1.99 on them, fmt.

${GATES}

Gate failures: ${JSON.stringify(fails)}
Blocking findings: ${JSON.stringify(open)}
Non-blocking findings: ${JSON.stringify(nonblocking).slice(0, 12000)}

Append to ${A}/run/fixes.md (per item: fixed how, or rejected why). Commit (subject "chore(wip): A4 fixes, round ${round} (not for merge)") and push. Return the per-item report.`, { label: `fix:r${round}`, phase: 'Fix', model: 'sonnet' })
  history[history.length - 1].fix_report = String(fix).slice(0, 3000)
}
return { ir, cg, tail, rounds: round, gates, open_blocking: open, history }