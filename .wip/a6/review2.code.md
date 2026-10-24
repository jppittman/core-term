# A6 review 2: the round-1 code, by experiment

HEAD `079c01d` on `claude/integral-area-work-ojiej6`. The lens is `git diff 6c60dfa -- '*.rs'`, read against
`.wip/a6/rulings.md` (Round 1 included). I did not edit, commit or stash anything in the repo, and I ran no gate
script. Builds used the repo's target dir: `CARGO_TARGET_DIR=/home/user/pixelflow/target.noindex
RUSTFLAGS="-C llvm-args=-fp-contract=fast"`. Free disk stayed at 2.8 GB or more.

**Verdict.** There is no behavioural defect, and no stage accepts what a later one refuses after round 1. The
folded `F32` arm changes nothing that `kernel!` or `kernel_raw!` can reach. Both new pins fail under the mutation
they exist for. One finding is blocking: a sentence A6 wrote in a test doc is false of the code. It is the twin of
round 1's D9, fixed in the function's doc but not in the test's. Three non-blocking findings follow.

## How it was tested

Everything below lives in `/home/user/scratch/a6-review-stages` (extended) and `/home/user/scratch/a6-review-laws`
(extended).

- **`compiler-copy-r1/`** is a copy of `pixelflow-compiler/src` at HEAD (`diff -r` clean before the probe), with a
  `review2_probe` module appended to `lower.rs`. For each input it runs `parse` and `sema::analyze`, then lowering
  with no sema in front (the unit tests' `lower_unanalyzed`). The output is in `probe_lowering_r1.txt`.
  `tests/kernel_typed_parameters.rs` and `tests/binding_times.rs` are verbatim copies, used for mutations. Every
  mutation was reverted by copying the file back from the repo.
- **`gen_cases_r2.py`** writes 21 cases, each under `kernel!` and `kernel_raw!`, as `examples/r2__*.rs`. It adds
  files and removes none. They were run with `cargo check --keep-going --message-format=json` (`check_r2.json`)
  and summarized by `report_r2.py` into `report_r2.txt`, which records the first error, its span and its full text.
- **`a6-review-laws/tests/r2_messages.rs`** reads the three runtime refusal texts the real `kernel!` emits: the
  host function, a staged host function, and `write_into`.
- **Reruns at HEAD:**
  - `cargo test -p pixelflow-compiler` under `PIXELFLOW_ISA=avx512` and `=avx2`: 319 and 319 passed, with 6
    doctests ignored (`r2_compiler_tests_{avx512,avx2}.txt`).
  - The stages suite (`accepted` 13, `adversarial` 1, `mask_outer` 1): green (`r2_stages_tests_default.txt`).
  - `cargo +1.99.0 clippy -p pixelflow-compiler --all-targets --all-features -- -D warnings`: exit 0.
  - `rustfmt --check --edition 2024` on the touched compiler files: exit 0.
  - `git diff 29980bf HEAD -- pixelflow-ir/src` is doc lines only, and `pixelflow-core/src` is untouched.

## What holds

**`RangeScope::usize_named`'s folded `F32` arm (`sema.rs:2006-2013`).** An `f32` const in a bound is refused by
sema with M2, spanned at the name, with "help: declare the count as `const R: usize`". The result is the same under
`kernel!` and `kernel_raw!` (`report_r2.txt` flags no pair that differs) in all 13 shapes tried:

- alone, at `lo`, and under `+ 1`;
- `N + R` with `N` a `usize` const;
- `(R)`;
- beside a count, `(n..R)`;
- under `.any`, inside an enclosing fold `(i..R)`, and in a helper;
- declared after the entry, as a `pub const`, built as `N as f32`, and in a staged entry.

`kernel_raw!` has no path of its own here. `expand` runs `parser::parse(..).and_then(sema::analyze)` for both
macros (`lib.rs:413-424`), so lowering's name check is reached only without sema. There, an unshadowed `f32` const
is refused with "`R` as a range's bound: a bound that is a value is an enclosing fold's index or an entry's count
parameter, a `usize`". That is the same text `probe_lowering.txt` recorded at `6c60dfa`, because lowering's name
check (`lower.rs:683`) runs before `range_bounds` and always did. It is a backstop's wording, not M2's, and no macro
input reaches it.

After the fold, sema's only defence for an `f32` const is `bound_named`. That defence is pinned. With mutation F2
(`bound_named` passes any `SymbolKind::Const`), sema accepts `(0..R)`, lowering refuses it, and
`sema::tests::a_range_bound_is_a_usize_and_reads_no_coordinate` fails (149 passed, 1 failed). The ruling's "the
f32-const bound row still pins M2's help" is true.

The fold's one behavioural change is lowering-only. An `f32` const that a count binding shadows is now resolved to
the binding:

- `const R: f32 = 2.0; pub fn f(R: usize)` with `(0..R)` lowers with `hi=Value(Uniform(0))`.
- An index `|R|` with `(0..R)` inside lowers with the index `Var(5)`.

At `6c60dfa` the arm would have refused both: the name check passes a `Count` binding, and `range_bounds` then hit
`ConstValue::F32`. A shadowed `usize` const still resolves to the const (`[0..2)`). Sema refuses every one of these
shadows first ("parameter `R` shadows the `const R` of this block", and the fold-index twin). So the new comment at
`lower.rs:680-682` holds for the `usize` const it describes, and the `f32` case now resolves innermost-first, as
sema does. Ruling S3's "unreachable" was strictly true only through sema, which is the case that matters.

**M3 and M7 (`sema.rs:1778-1793`, `:691-701`).** The texts are as emitted in `report_r2.txt`. M7's note now names
only the host function: an items-form entry's host function does refuse a count past 2²⁴. M3's new clause, that
arithmetic on an index waits for bounds typed as integers, is true of the plan. See N1 for M3's help as a whole.

**`COUNT_DOMAIN` (`emit.rs:267-269`).** The rendered runtime texts, from `r2_messages.rs` through the real
`kernel!`, are byte-identical for the host function, a staged host function and `write_into`:

> kernel!: `sum_to`'s count `n` is 16777217, past 16777216 (2^24): a count is an integer in `0..=2^24`;
> `2^24 + 1` would be written as `2^24`, a fold bounded by it would run one trip short with no NaN, and a larger
> count would name none (§1.4 of docs/language/2026-10-20-a-bound-is-a-value.md)

`2^24` itself is accepted. `16777217 as f32 == 16777216.0`, and 2²⁴ + 2 and up are exact past the domain, so
"name none" holds. A precision note, not a finding: "one trip short" describes a count at a fold's `hi`. At `lo`
nothing changes, because every `hi` beside it is ≤ 2²⁴ (a literal is checked, a count refused), so both the written
range and Rust's are empty. The refusal stands on the domain clause, which comes first.

**The emit-test pin (`emit.rs:1075-1083`).** Baseline: pass.

- **Mutation A,** `#panics` deleted from `write_into`'s quote: `a_count_parameter_is_refused_on_the_host_and_declared_as_an_f32`
  FAILED at `emit.rs:1078` ("write_into documents its panic: impl SumToArgs { …").
- **Mutation B,** the `Args` text reworded to the entry's "If a count argument is past 2^24": FAILED at `:1079`.

Taking the slice from `impl SumToArgs` tells the two write sites apart, which was round 1's N1.

**The `should_panic` pin (`kernel_typed_parameters.rs:286-293`).** Baseline: 18 passed. Mutation D made a staged
entry's host function refuse nothing (`counts.iter().filter(|_| !staged)` in `emit_entry`). That one test fails
with "test did not panic as expected at tests/kernel_typed_parameters.rs:290:4", and the other 17 pass. Nothing
else panics with a staged count of 2²⁴ + 1, so the pin is specific to the staged host refusal, the gap N1 named.

**Stage agreement after round 1.** Round 1's only code change with behaviour is the arm above. Every accepted
spelling of the round-1 stages suite still builds and matches rustc at HEAD. Every refusal tried is made at
expansion by sema, with lowering agreeing behind it.

## Findings, most severe first

### B1. Blocking. A test doc A6 wrote states M2's help wrongly (the twin of round-1 D9)

`pixelflow-compiler/src/sema.rs:2869-2871`, the doc of `a_range_bound_is_a_usize_and_reads_no_coordinate`:

> The help of a name that is an `f32` is the entry's `n: usize`, where an entry of the items form can take one.

`not_a_usize_bound` (`sema.rs:1000-1013`) chooses the help by what the name is. Two items-form entries, which can
take a count, show the sentence is false:

- `const R: f32 = 4.0; pub fn f() -> f32 { (0..R)… }` gets "help: declare the count as `const R: usize`"
  (`report_r2.txt`, `f32c_alone`). The test's own row at `:2916-2921` pins exactly that.
- `pub fn f() -> f32 { let n = 4.0; (0..n)… }` gets no help at all (`report_r2.txt`, `m2_let_items`, and the probe).

The sentence was written by `6c60dfa`. `git show 29980bf:pixelflow-compiler/src/sema.rs | grep -c "help of a name
that is an"` gives 0. Round 1's D9 corrected the same claim in `not_a_usize_bound`'s doc (`:996-999`) and was ruled
blocking. This copy, three lines above the test that contradicts it, was missed.

Fix (delete before qualifying): drop the sentence. The function's doc already states the rule. Otherwise reword it:
"The help is by what the name is: an `f32` const's is `const R: usize`, an entry's `f32` parameter's the per-call
count, and any other has none."

### N1. Non-blocking. M3's help is not chosen by what the name is, and two of its forms refuse what it offers

`sema.rs:1789-1791` gives every M3 the same help: "a count known at expansion is a `usize` const…; one that changes
per call is computed by the host and passed as a parameter of its own; arithmetic on an index waits for bounds
typed as integers".

In the closure form and in a helper, M3 can only be about an index, since neither takes a count (M6, M7).
`report_r2.txt` shows following the middle clause leads straight to another refusal:

- the closure form `|m: usize| … (0..m) …` is refused with M7 (`m3_follow_closure`);
- a helper `fn h(a: f32, m: usize)` is refused with M6 (`m3_follow_helper`).

In an items entry, the index clause is noise for a count, and the host clause is noise for an index. Ruling S2
made M2's help depend on role and spelling for the same reason, and M3's was left alone. `bound_named` has already
looked up the name's `SymbolKind` (`Index` or `Parameter`) by the time `range_bounds` reaches M3.

Fix: give the host clause for a count parameter and the index clause for an index.

### N2. Non-blocking. `COUNT_DOMAIN`'s text is pinned by no test

LAWS-2 put this text in the generated `# Panics`, in `write_into`'s doc and in J10. Mutation G replaced the const
with `"MUTATION G: no domain text"`. Every test that reads a refusal stays green:

- `emit::` 8/8;
- `binding_times` 12/12;
- `kernel_typed_parameters` 18/18.

`grep -rn "one trip short\|integer in \`0..=2^24\`\|name none" pixelflow-compiler/{src,tests}` finds only the
const and a doc comment (`binding_times.rs:228`). The pins stop at "past 16777216 (2^24)" and "a count in this
record is past 2^24", the prefix before the text. Round 1's "asserts `write_into`'s own `# Panics` and its domain
text" holds for that prefix, not for `COUNT_DOMAIN`.

Fix: the emit test is in the same module, so add `write_into.contains(COUNT_DOMAIN)`, and the same for the host
function's section.

### N3. Non-blocking. A test doc A6 wrote keeps the "decided per call" that round-1 D8 corrected

`sema.rs:3072-3074`: "…and so is a pair with a value, which is decided per call (L2)". D8 changed the same clause in
`range_bounds`' doc (`:1954`) to "decided at run time". A pair with an index, such as `(i..2)` in an outer fold, is
decided on each outer trip. The test's own example `5..n` is per call, so this is the weaker of the twins, and D8 was
not ruled blocking.

Fix: "decided at run time".

## Scratch files

All under `/home/user/scratch/`:

- `a6-review-stages/compiler-copy-r1/`: the HEAD copy, the probe module and the copied tests;
- `a6-review-stages/probe_lowering_r1.txt`;
- `a6-review-stages/gen_cases_r2.py`, `cases_r2.txt`, `examples/r2__*.rs`, `check_r2.json`, `report_r2.py` and
  `report_r2.txt`;
- `a6-review-stages/r2_compiler_tests_avx512.txt`, `r2_compiler_tests_avx2.txt` and `r2_stages_tests_default.txt`;
- `a6-review-laws/tests/r2_messages.rs`.
