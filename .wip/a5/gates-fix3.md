# A5 gates, rounds 3 and 4

Tree: round 3 is 2121763, then 26b7d1c (the gate probe that 2121763 captured, removed). Gates were run on
26b7d1c with tag `a5-fix3b`.

| Check | Result |
|---|---|
| byteprobe.sh a5-fix3b | exit 0, avx512 and avx2 identical to base |
| gate1b.sh a5-fix3b | exit 0, identical (29 lines) |
| gate2.sh / gate2x.sh a5-fix3b | exit 0 / 0, identical |
| `PIXELFLOW_ISA=avx2` and `=avx512` `cargo test -p pixelflow-compiler` | 299 + 299 passed, 0 failed |
| `cargo test -p pixelflow-ir`, `-p pixelflow-search --features provenance-journal`, `-p pixelflow-codegen --lib` | 933 passed, 0 failed |
| clippy 1.99, workspace, all targets, all features, `-D warnings` | exit 0 |
| `cargo fmt --all -- --check` | clean |
| check-intent, check-doc-paths, check-cfg-encapsulation, check-emit-boundary, check-bin-declarations | all OK |
| R3C-1 mutation (remove `leading_colon` guard) | `a_type_named_from_the_crate_root_is_refused` FAILED, then restored |
| CI on 20d8144 | 23/23 green |

The first round-3 run (`a5-fix3`) failed gate2 with E0428, which was the harness's own fault and not A5's.
A commit taken while gate1b was building recorded the appended probes, so the next restore put them back.
Rule added to gates/README.md.

Round 4 (doc only): `cargo test -p pixelflow-compiler --lib` 143, `ir_insert` 13, `pixelflow-ir --lib` 310;
clippy 1.99 on compiler, ir and search is exit 0; rustdoc NEW 0 for all three; fmt and the doc scripts are OK.
