# Post-submit quality pipeline

Three workflows run against every push to `main`. Together they answer: did
this commit break tests, make them flaky, or slow the hot paths — and if it
broke them, who backs it out?

```
push to main
  ├── Postsubmit Flake Detection ──(conclusion: failure)──▶ Automatic Revert
  │     5× per (OS, suite); flaky ⇒ issue, consistent ⇒ fail
  └── Benchmark Regression Check
        Criterion vs gh-pages baseline; >25% ⇒ issue + commit comment
```

## Flake detection (`postsubmit-flake-detection.yaml`)

Each (OS × suite) job builds the tests once, then runs the suite **5 times**
with nextest (per-test 10-minute cap from `.config/nextest.toml`). The five
iterations classify the commit:

| Result | Meaning | Action |
|---|---|---|
| 5/5 pass | healthy | nothing |
| 1–4/5 fail | **flaky** | file/update a `flaky-test` issue; workflow stays green |
| 0/5 pass (or build fails) | **consistently broken** | workflow fails → automatic revert |

The distinction is the point: a flake reverted is a lie recorded (the next
commit "fixes" it by luck), and a hard breakage merely issue-filed rots on
main. The `flake-report` job runs with `continue-on-error` so an issue-filing
hiccup can never masquerade as a postsubmit failure and trigger a revert.

## Automatic revert (`automatic-revert.yaml`)

Fires on a failed flake-detection run. Creates a tracking issue and a revert
PR (branch `revert-failed-postsubmit-<sha7>`), with guards:

- **Dedup** — one revert branch per failing SHA, however many runs report it.
- **Loop guard** — if the failing commit is itself an automated revert,
  re-reverting would re-land the original bad commit; it escalates to an
  issue instead.
- **Merge commits** — reverted against the first parent (`-m 1`).
- **Conflicts** — a revert that doesn't apply cleanly becomes an issue
  asking for a manual revert, not a broken branch.

Reverts are proposed as PRs, not pushed to `main` directly: presubmit still
gets to check that the revert itself builds.

## Benchmark regression (`benchmark_regression.yaml`)

Runs `cargo bench --workspace --benches -- --output-format bencher` and feeds
the result to `benchmark-action/github-action-benchmark`, which keeps the
baseline series on the `gh-pages` branch. A benchmark more than **25%** slower
than baseline gets a commit comment and one open `performance-regression`
issue (subsequent regressions comment on the existing issue rather than
piling up new ones). Runs are serialized by a concurrency group so baselines
land in commit order.

Perf regressions do **not** auto-revert: hosted-runner noise makes a
threshold breach evidence, not proof. The issue is the escalation path.

`--benches` only works because every `[lib]` in the workspace sets
`bench = false` and every `[[bench]]` target is Criterion with
`harness = false` — the libtest harness rejects `--output-format bencher`.
Keep new crates on that convention or the benchmark job will fail at
flag-parse time.
