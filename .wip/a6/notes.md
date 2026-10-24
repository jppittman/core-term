# A6 working notes

A6 of docs/language/2026-10-20-a-bound-is-a-value.md: `kernel!` bounds are typed expressions (§3.4).
A bound is a literal, a `usize` const (with `+ - * /`), an enclosing fold's index, or an
`n: usize` entry parameter (decision 2, JP 2026-10-22). Gate: rustc_is_the_oracle gains a
triangular case and a per-call count case; the `Args` write refuses 2^24 + 1;
fold_is_kernel_over.rs green; byte_probe empty.

Base: main 29980bf (A5 merged, #1449). Design carried from A5: design.md (with R-A6 at its end).
This directory is dropped (`git rm -r .wip`) before the PR is marked ready.

## State
- [ ] design: harness re-verify at 29980bf, spec, critiques
- [ ] rulings (D-A6-1 public API: to JP with a recommendation if it adds a public type)
- [ ] implement
- [ ] review rounds
- [ ] close
