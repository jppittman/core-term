export const meta = {
  name: 'a4-design',
  description: 'A4 design: rebase the gate harness at main, write the A4 spec (value bounds in codegen and the key), three adversarial critiques',
  phases: [
    { title: 'Harness', detail: 're-verify bases at main, re-record compile cost' },
    { title: 'Spec', detail: 'A4 implementation spec at a747770' },
    { title: 'Critique', detail: 'codegen semantics, completeness and identity, emitter mechanics' },
  ],
}
const W = '/home/user/pixelflow/.wip'
const AR = '/home/user/scratch/a3-wip-archive/.wip/a3'
const RULES = `Ground rules (hard):
- Repository /home/user/pixelflow, branch claude/integral-area-work-ojiej6 at main a747770 (A3 merged as #1447). Never build from a git worktree. Never commit, push, stash, reset or rebase. Never edit tracked files (the gate scripts append and restore their own probe file).
- Never delete anything under /tmp/claude-0, /home/user/scratch or ${W}. Scratch crates live in /home/user/scratch, never in the repo.
- Never \`cargo test --workspace\`. \`df -h /\` before builds; under 2 GB free, stop.
- No model names in any file you write.`

const FIND_SCHEMA = { type: 'object', properties: { findings: { type: 'array', items: { type: 'object', properties: {
  id: { type: 'string' }, spec_section: { type: 'string' }, file: { type: 'string' }, line: { type: 'integer' },
  claim: { type: 'string' }, evidence: { type: 'string' }, fix: { type: 'string' },
  blocking: { type: 'boolean', description: 'true if implementing the spec as written would produce wrong code or values for a value-bounded fold on any tier, hang, change a byte/key/price on today\'s kernels, violate the plan/CLAUDE.md, miss a site, or make a gate unpassable' } },
  required: ['id', 'spec_section', 'file', 'line', 'claim', 'evidence', 'fix', 'blocking'] } }, verdict: { type: 'string' } }, required: ['findings', 'verdict'] }

const harness = () => agent(`${RULES}

The gate harness is in ${W}/gates/ (read its README.md). Its bases in out/*.base.* were recorded at 3bfdaded; main is now a747770 (A3), which is byte-identical on today's kernels by construction. At the current tree (\`git status --short\` shows only .wip/; if anything else, stop):
1. Run \`bash ${W}/gates/byteprobe.sh a4-main\`, \`gate1b.sh a4-main\`, \`gate2.sh a4-main && gate2x.sh a4-main\`: each must exit 0 (identical to base). If one differs, stop and report the diff.
2. Re-record the compile-cost base for main: \`GATES_REBASE=1 bash ${W}/gates/measure.sh base\` (A3 moved Ir by +0.19%); report the old and new JSON.
3. Rustdoc bases at main for pixelflow-ir, pixelflow-search and pixelflow-codegen, normalized by ${W}/a4/rustdoc_norm.py (read it; the method is in ${AR}/run/rustdoc-base.md) into ${W}/a4/rustdoc-base.<crate>.txt, and confirm \`RUSTDOCFLAGS="-D warnings" cargo doc -p pixelflow-codegen --no-deps --document-private-items\` passes.
4. Update ${W}/gates/README.md's "Bases" paragraph: verified at a747770, measure re-recorded there.
Report each result.`, { label: 'harness', phase: 'Harness', model: 'sonnet' })

const specAndCritique = async () => {
  const spec = await agent(`${RULES}

Write the A4 implementation spec to ${W}/a4/spec.md (read-only on the repo otherwise). A4 is step A4 of docs/language/2026-10-20-a-bound-is-a-value.md: read the plan in full (§0-§8), especially §1.4 (the domain), L5 (the guard: outside the domain the fold is NaN), §3 (representation, placement rule §3.3), §6's A4 bullet and its gate list, §8's decisions (decision 7: over_values public in A4). A3 is landed at a747770: read its code (pixelflow-ir fold.rs/arena.rs/variance.rs/kernel.rs, pixelflow-search egraph/{node,graph,insert,fold_rules,extract,cost}.rs, pixelflow-codegen/src/program/lower.rs's VALUE_BOUND refusal) and its design record ${AR}/spec.md (§13 J24-J28: the extraction-side hole A4 must decide) and ${AR}/rulings.md. Read CLAUDE.md's "If guards its arms", "Floating point at the edges" and "The language is a DAG" — they bind the guard's design — and pixelflow-codegen's intent.md files, its program/ (lowering, scopes/placement, the loop carve), emit/ (per-ISA emitters: x86-64 AVX2, AVX-512, aarch64 NEON), and how a fold's trip count reaches the loop today.

The spec must let a careful implementer do A4 without design decisions of their own. Cover, with file:line at a747770:
0. What A4 is and is not (A5-A7 and B are not).
1. The semantics, as laws: what a value-bounded fold computes at run time on every tier, for every bound value: in the domain (integer in [0, 2^24], lo <= hi or the reversed-pair rule), outside it (NaN, never a hang, never an out-of-range trip count), NaN, non-integers, negative, 2^24+2. Where the guard lives (IR-level legalization before scheduling, as an \`If\`? — say why, given "If guards its arms"), what the post-collapse assert checks, and how the trip count reaches the loop (the operand and carve change; "Trips" and its Display; EmitTraffic).
2. Placement (§3.3): a bound read only by the loop; a guarded fold hoisted to Scope::Body when its body reads no coordinate (the carve sees the If, not the Reduce, as its def — the single-reader assumption); a bound that varies with nothing but uniforms is computed once per call.
3. Codegen changes by file, per ISA where they differ, with the register and label consequences.
4. The key: the Value marker in FoldHeader::to_bits's count field and from_bits's inverse, with the const assert; every serializer that writes a header without its children (the plan's list, plus any the grep finds) learns the marker; the training corpus's format version bump.
5. over_values public (decision 7), with its doc and the user-surface refusals.
6. The extraction-side hole (A3's J24): give the options (leave it loud; constrain a bound class's pick to a member that keeps L3; a per-use pick) with costs, and RECOMMEND one, stating it is a decision for review.
7. Byte identity on today's kernels: every changed site on the literal path, and why it moves no byte, key, price or saturation result.
8. Tests: the plan's gate list verbatim as named tests in pixelflow-codegen/tests/ against scalar Rust on every tier (PIXELFLOW_ISA=avx2|avx512 and aarch64 via the Cross-target ABI job or the existing per-tier harness — say which), plus unit tests per changed function, the key round-trip, each serializer's marker.
9. Order of work and the grep-gated checklist (VALUE_BOUND's remaining uses, KNOWN_RANGE's, every \`known()\`/\`known_range().expect\` site in codegen).
10. Docs: plan (A4 Done line; §5 rows marked A4), CLAUDE.md:175-ish and DENOTATION rows the plan assigns to A4, the ontology rows ("Fold (Reduce)", "Range", "Totality"), intent.md files with their 80-line budgets.
11. Judgement calls, each with the rejected alternative.
Return a 25-line summary: section sizes, the decisions you could not make (with options), and the riskiest part.`, { label: 'spec', phase: 'Spec' })

  const lenses = [
    { key: 'semantics', text: `Lens: run-time semantics and soundness. For every bound value class (in-domain integers incl. 0, 1, 2^24; reversed pair; non-integer; negative; NaN; ±inf; 2^24+2; a bound varying per lane? — impossible by L3, check the spec relies on it correctly), does the spec's guard + loop compute exactly the fold's value or NaN, on every tier, with no hang and no out-of-range trip count, no read past a buffer, and no UB in the emitted code? Is the guard an If whose semantics ("If guards its arms": both arms may run in a mixed batch) are safe — can the unguarded arm run a fold with a garbage trip count in a mixed batch? Is the trip count's f32→int conversion right on each tier (cvttps2dq's i32::MIN on NaN/overflow, FCVTZS saturation)? Is the key marker sound (from_bits(to_bits(h)) == h; no literal collides with the marker)? Is the recommended extraction choice sound?` },
    { key: 'completeness', text: `Lens: completeness and byte identity. Every site the plan's A4 bullet names is in the spec with file:line; rerun the greps yourself (VALUE_BOUND, KNOWN_RANGE, known_range().expect, to_bits/from_bits callers, every serializer of a header, corpus format version, over_values visibility). Every test in the plan's gate list is a named test with an assertion. Byte identity on today's all-literal kernels: find any changed site on the literal path that could move a byte, key, node id, price or saturation count. Docs and intent budgets (\`bash scripts/check-intent.sh\`). Nothing of A5-A7 or B built; nothing left for the implementer to decide.` },
    { key: 'emitter', text: `Lens: the codegen mechanics. Read pixelflow-codegen's program/ (lowering, scopes, placement, the carve, the allocation tail) and emit/ (x86-64 AVX2, AVX-512, aarch64 NEON) and check the spec's operand/carve change against them: how a loop's trip count is held today (immediate? register? the counter is an f32 lane?), what changes when it is a value (a register live across the loop; spills; the per-ISA compare/branch; encodings with imm limits; the aarch64 fixed-size skip from CL7), how a guarded fold placed in Scope::Body interacts with the single-reader assumption and the scope machinery, and whether the post-collapse assert and EmitTraffic are specified precisely enough to implement. Point at exact functions.` },
  ]
  const crits = await parallel(lenses.map(l => () => agent(`${RULES}

Adversarial critique of the A4 implementation spec ${W}/a4/spec.md against the live code at a747770, the plan docs/language/2026-10-20-a-bound-is-a-value.md, A3's record (${AR}/spec.md, ${AR}/rulings.md) and CLAUDE.md. Read-only. Report only defects you can show with evidence (file:line); rank them; an empty list is fine. Also write your critique as prose to ${W}/a4/critique.${l.key}.md.

${l.text}`, { label: `critique:${l.key}`, phase: 'Critique', schema: FIND_SCHEMA })))
  return { spec, crits: crits.map((c, i) => ({ lens: lenses[i].key, ...(c || { findings: [], verdict: 'agent returned nothing' }) })) }
}

const [h, s] = await parallel([harness, specAndCritique])
return { harness: h, spec: s && s.spec, critiques: s && s.crits }