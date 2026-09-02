# R2G spread vs. budget (round 3)

> **Instrument change (2026-09-02 forward-port).** Two things about how a
> Phase 3 anytime curve is measured changed with the port, so a re-run does
> not reproduce the numbers below even if nothing else changed: the
> application budget now binds **mid-scan** rather than between rule sweeps
> (`app_actual == app_target` exactly, no overshoot), and the reported cost
> is the **DAG** cost the emitted kernel pays rather than the extraction DP's
> tree total (#1117). Full statement:
> [docs/results/2026-09-02-phase3-instrument-changes.md](2026-09-02-phase3-instrument-changes.md).

Budget ladder: `[100, 200, 400, 800, 1600, 3200]`.

Per-tier expression counts (trajectory rows grouped by `expr_name`):

- **bezier**: 80
- **dev**: 783
- **sh**: 100
- **train**: 3356

## train

### band all

| B | n_expr | zero-spread (all 12) | zero-spread (guided-only) | qualifies | spread Q1/median/Q3 | unguided differs | top stop reasons |
|---:|---:|---:|---:|:---:|---|---:|---|
| 100 | 3356 | 55.9% | 63.8% | no | 0.000/0.000/0.043 | 44.1% (n=3356) | Quiesced:21489, ApplicationBudget:18783 |
| 200 | 3356 | 64.2% | 72.0% | no | 0.000/0.000/0.068 | 35.8% (n=3356) | Quiesced:26422, ApplicationBudget:13850 |
| 400 | 3356 | 67.3% | 81.7% | no | 0.000/0.000/0.013 | 32.7% (n=3356) | Quiesced:30396, ApplicationBudget:9876 |
| 800 | 3356 | 70.1% | 88.9% | no | 0.000/0.000/0.002 | 29.9% (n=3356) | Quiesced:34191, ApplicationBudget:6081 |
| 1600 | 3356 | 72.6% | 91.2% | no | 0.000/0.000/0.001 | 27.4% (n=3356) | Quiesced:36630, ApplicationBudget:3642 |
| 3200 | 3356 | 73.2% | 91.1% | no | 0.000/0.000/0.000 | 26.8% (n=3356) | Quiesced:37724, ApplicationBudget:2548 |

### band 51-100

| B | n_expr | zero-spread (all 12) | zero-spread (guided-only) | qualifies | spread Q1/median/Q3 | unguided differs | top stop reasons |
|---:|---:|---:|---:|:---:|---|---:|---|
| 100 | 551 | 43.4% | 66.8% | no | 0.000/0.003/0.007 | 56.6% (n=551) | ApplicationBudget:4283, Quiesced:2329 |
| 200 | 551 | 72.2% | 91.1% | no | 0.000/0.000/0.002 | 27.8% (n=551) | Quiesced:4939, ApplicationBudget:1673 |
| 400 | 551 | 78.4% | 94.2% | no | 0.000/0.000/0.000 | 21.6% (n=551) | Quiesced:5805, ApplicationBudget:807 |
| 800 | 551 | 77.9% | 95.1% | no | 0.000/0.000/0.000 | 22.1% (n=551) | Quiesced:6207, ApplicationBudget:405 |
| 1600 | 551 | 77.1% | 94.9% | no | 0.000/0.000/0.000 | 22.9% (n=551) | Quiesced:6367, ApplicationBudget:245 |
| 3200 | 551 | 77.7% | 94.9% | no | 0.000/0.000/0.000 | 22.3% (n=551) | Quiesced:6430, ApplicationBudget:182 |

### band 101-250

| B | n_expr | zero-spread (all 12) | zero-spread (guided-only) | qualifies | spread Q1/median/Q3 | unguided differs | top stop reasons |
|---:|---:|---:|---:|:---:|---|---:|---|
| 100 | 540 | 5.4% | 11.1% | **YES** | 0.004/0.013/0.543 | 94.6% (n=540) | ApplicationBudget:6469, Quiesced:11 |
| 200 | 540 | 21.5% | 36.1% | **YES** | 0.001/0.020/0.540 | 78.5% (n=540) | ApplicationBudget:5037, Quiesced:1443 |
| 400 | 540 | 33.7% | 74.1% | no | 0.000/0.004/0.509 | 66.3% (n=540) | Quiesced:3632, ApplicationBudget:2848 |
| 800 | 540 | 43.7% | 84.1% | no | 0.000/0.001/0.055 | 56.3% (n=540) | Quiesced:4928, ApplicationBudget:1552 |
| 1600 | 540 | 49.6% | 84.8% | no | 0.000/0.000/0.003 | 50.4% (n=540) | Quiesced:5578, ApplicationBudget:902 |
| 3200 | 540 | 50.7% | 84.4% | no | 0.000/0.000/0.002 | 49.3% (n=540) | Quiesced:5846, ApplicationBudget:634 |

### band 251-1000

| B | n_expr | zero-spread (all 12) | zero-spread (guided-only) | qualifies | spread Q1/median/Q3 | unguided differs | top stop reasons |
|---:|---:|---:|---:|:---:|---|---:|---|
| 100 | 529 | 0.0% | 0.0% | **YES** | 0.362/0.496/5.348 | 100.0% (n=529) | ApplicationBudget:6348 |
| 200 | 529 | 0.0% | 0.0% | **YES** | 0.365/0.518/5.367 | 100.0% (n=529) | ApplicationBudget:6348 |
| 400 | 529 | 0.0% | 19.1% | **YES** | 0.366/0.498/4.628 | 100.0% (n=529) | ApplicationBudget:5863, Quiesced:485 |
| 800 | 529 | 9.3% | 54.1% | no | 0.005/0.133/0.546 | 90.7% (n=529) | ApplicationBudget:3884, Quiesced:2464 |
| 1600 | 529 | 19.8% | 67.5% | no | 0.000/0.002/0.087 | 80.2% (n=529) | Quiesced:4049, ApplicationBudget:2299 |
| 3200 | 529 | 22.1% | 67.9% | no | 0.000/0.001/0.059 | 77.9% (n=529) | Quiesced:4754, ApplicationBudget:1594 |

## dev

### band all

| B | n_expr | zero-spread (all 12) | zero-spread (guided-only) | qualifies | spread Q1/median/Q3 | unguided differs | top stop reasons |
|---:|---:|---:|---:|:---:|---|---:|---|
| 100 | 783 | 60.0% | 67.9% | no | 0.000/0.000/0.011 | 40.0% (n=783) | Quiesced:5546, ApplicationBudget:3850 |
| 200 | 783 | 67.8% | 77.1% | no | 0.000/0.000/0.006 | 32.2% (n=783) | Quiesced:6684, ApplicationBudget:2712 |
| 400 | 783 | 71.4% | 86.5% | no | 0.000/0.000/0.002 | 28.6% (n=783) | Quiesced:7541, ApplicationBudget:1855 |
| 800 | 783 | 73.6% | 90.9% | no | 0.000/0.000/0.001 | 26.3% (n=783) | Quiesced:8267, ApplicationBudget:1129 |
| 1600 | 783 | 75.4% | 93.0% | no | 0.000/0.000/0.000 | 24.6% (n=783) | Quiesced:8752, ApplicationBudget:644 |
| 3200 | 783 | 76.0% | 93.0% | no | 0.000/0.000/0.000 | 24.0% (n=783) | Quiesced:8943, ApplicationBudget:453 |

### band 51-100

| B | n_expr | zero-spread (all 12) | zero-spread (guided-only) | qualifies | spread Q1/median/Q3 | unguided differs | top stop reasons |
|---:|---:|---:|---:|:---:|---|---:|---|
| 100 | 126 | 42.1% | 65.1% | no | 0.000/0.003/0.014 | 57.9% (n=126) | ApplicationBudget:984, Quiesced:528 |
| 200 | 126 | 65.1% | 89.7% | no | 0.000/0.000/0.003 | 34.9% (n=126) | Quiesced:1119, ApplicationBudget:393 |
| 400 | 126 | 73.0% | 92.1% | no | 0.000/0.000/0.002 | 27.0% (n=126) | Quiesced:1290, ApplicationBudget:222 |
| 800 | 126 | 70.6% | 92.9% | no | 0.000/0.000/0.002 | 29.4% (n=126) | Quiesced:1386, ApplicationBudget:126 |
| 1600 | 126 | 70.6% | 92.1% | no | 0.000/0.000/0.002 | 29.4% (n=126) | Quiesced:1427, ApplicationBudget:85 |
| 3200 | 126 | 69.8% | 92.1% | no | 0.000/0.000/0.002 | 30.2% (n=126) | Quiesced:1460, ApplicationBudget:52 |

### band 101-250

| B | n_expr | zero-spread (all 12) | zero-spread (guided-only) | qualifies | spread Q1/median/Q3 | unguided differs | top stop reasons |
|---:|---:|---:|---:|:---:|---|---:|---|
| 100 | 116 | 2.6% | 7.8% | **YES** | 0.003/0.008/0.215 | 97.4% (n=116) | ApplicationBudget:1392 |
| 200 | 116 | 24.1% | 42.2% | **YES** | 0.000/0.007/0.219 | 75.9% (n=116) | ApplicationBudget:1028, Quiesced:364 |
| 400 | 116 | 37.1% | 85.3% | no | 0.000/0.002/0.073 | 62.9% (n=116) | Quiesced:855, ApplicationBudget:537 |
| 800 | 116 | 50.9% | 86.2% | no | 0.000/0.000/0.003 | 48.3% (n=116) | Quiesced:1109, ApplicationBudget:283 |
| 1600 | 116 | 52.6% | 87.1% | no | 0.000/0.000/0.002 | 47.4% (n=116) | Quiesced:1246, ApplicationBudget:146 |
| 3200 | 116 | 54.3% | 87.1% | no | 0.000/0.000/0.002 | 45.7% (n=116) | Quiesced:1304, ApplicationBudget:88 |

### band 251-1000

| B | n_expr | zero-spread (all 12) | zero-spread (guided-only) | qualifies | spread Q1/median/Q3 | unguided differs | top stop reasons |
|---:|---:|---:|---:|:---:|---|---:|---|
| 100 | 92 | 0.0% | 0.0% | **YES** | 0.374/0.558/5.174 | 100.0% (n=92) | ApplicationBudget:1104 |
| 200 | 92 | 0.0% | 0.0% | **YES** | 0.377/0.562/5.020 | 100.0% (n=92) | ApplicationBudget:1104 |
| 400 | 92 | 0.0% | 21.7% | **YES** | 0.361/0.495/4.891 | 100.0% (n=92) | ApplicationBudget:1019, Quiesced:85 |
| 800 | 92 | 6.5% | 57.6% | no | 0.004/0.187/0.634 | 93.5% (n=92) | ApplicationBudget:679, Quiesced:425 |
| 1600 | 92 | 19.6% | 75.0% | no | 0.000/0.002/0.069 | 80.4% (n=92) | Quiesced:728, ApplicationBudget:376 |
| 3200 | 92 | 23.9% | 75.0% | no | 0.000/0.001/0.013 | 76.1% (n=92) | Quiesced:820, ApplicationBudget:284 |

## sh

### band all

| B | n_expr | zero-spread (all 12) | zero-spread (guided-only) | qualifies | spread Q1/median/Q3 | unguided differs | top stop reasons |
|---:|---:|---:|---:|:---:|---|---:|---|
| 100 | 100 | 0.0% | 3.0% | **YES** | 0.013/0.073/0.110 | 100.0% (n=100) | ApplicationBudget:1189, Quiesced:11 |
| 200 | 100 | 0.0% | 5.0% | **YES** | 0.013/0.078/0.116 | 100.0% (n=100) | ApplicationBudget:1165, Quiesced:35 |
| 400 | 100 | 0.0% | 7.0% | **YES** | 0.038/0.086/0.125 | 100.0% (n=100) | ApplicationBudget:1145, Quiesced:55 |
| 800 | 100 | 4.0% | 19.0% | **YES** | 0.038/0.101/0.126 | 96.0% (n=100) | ApplicationBudget:1134, Quiesced:66 |
| 1600 | 100 | 7.0% | 15.0% | **YES** | 0.010/0.068/0.116 | 93.0% (n=100) | ApplicationBudget:1129, Quiesced:71 |
| 3200 | 100 | 8.0% | 9.0% | **YES** | 0.018/0.076/0.129 | 92.0% (n=100) | ApplicationBudget:1123, Quiesced:77 |

### band 51-100

| B | n_expr | zero-spread (all 12) | zero-spread (guided-only) | qualifies | spread Q1/median/Q3 | unguided differs | top stop reasons |
|---:|---:|---:|---:|:---:|---|---:|---|
| 100 | 49 | 0.0% | 0.0% | **YES** | 0.007/0.016/0.104 | 100.0% (n=49) | ApplicationBudget:588 |
| 200 | 49 | 0.0% | 4.1% | **YES** | 0.008/0.021/0.106 | 100.0% (n=49) | ApplicationBudget:578, Quiesced:10 |
| 400 | 49 | 0.0% | 8.2% | **YES** | 0.006/0.062/0.111 | 100.0% (n=49) | ApplicationBudget:577, Quiesced:11 |
| 800 | 49 | 2.0% | 22.4% | **YES** | 0.008/0.067/0.115 | 98.0% (n=49) | ApplicationBudget:577, Quiesced:11 |
| 1600 | 49 | 8.2% | 14.3% | **YES** | 0.005/0.047/0.105 | 91.8% (n=49) | ApplicationBudget:572, Quiesced:16 |
| 3200 | 49 | 8.2% | 10.2% | **YES** | 0.010/0.060/0.102 | 91.8% (n=49) | ApplicationBudget:569, Quiesced:19 |

### band 101-250

| B | n_expr | zero-spread (all 12) | zero-spread (guided-only) | qualifies | spread Q1/median/Q3 | unguided differs | top stop reasons |
|---:|---:|---:|---:|:---:|---|---:|---|
| 100 | 46 | 0.0% | 2.2% | **YES** | 0.062/0.091/0.112 | 100.0% (n=46) | ApplicationBudget:552 |
| 200 | 46 | 0.0% | 0.0% | **YES** | 0.070/0.106/0.128 | 100.0% (n=46) | ApplicationBudget:552 |
| 400 | 46 | 0.0% | 0.0% | **YES** | 0.074/0.110/0.139 | 100.0% (n=46) | ApplicationBudget:552 |
| 800 | 46 | 0.0% | 10.9% | **YES** | 0.079/0.110/0.139 | 100.0% (n=46) | ApplicationBudget:552 |
| 1600 | 46 | 0.0% | 10.9% | **YES** | 0.051/0.093/0.147 | 100.0% (n=46) | ApplicationBudget:552 |
| 3200 | 46 | 2.2% | 2.2% | **YES** | 0.052/0.100/0.251 | 97.8% (n=46) | ApplicationBudget:552 |

## bezier

### band all

| B | n_expr | zero-spread (all 12) | zero-spread (guided-only) | qualifies | spread Q1/median/Q3 | unguided differs | top stop reasons |
|---:|---:|---:|---:|:---:|---|---:|---|
| 100 | 80 | 25.0% | 76.2% | no | 0.075/0.100/0.122 | 75.0% (n=80) | ApplicationBudget:960 |
| 200 | 80 | 25.0% | 43.8% | **YES** | 0.091/0.254/0.254 | 75.0% (n=80) | ApplicationBudget:960 |
| 400 | 80 | 25.0% | 25.0% | **YES** | 0.212/0.283/0.283 | 75.0% (n=80) | ApplicationBudget:960 |
| 800 | 80 | 25.0% | 25.0% | **YES** | 0.205/0.363/0.435 | 75.0% (n=80) | ApplicationBudget:740, Quiesced:220 |
| 1600 | 80 | 25.0% | 25.0% | **YES** | 0.299/0.467/0.565 | 75.0% (n=80) | ApplicationBudget:740, Quiesced:220 |
| 3200 | 80 | 25.0% | 25.0% | **YES** | 0.374/0.566/0.657 | 75.0% (n=80) | ApplicationBudget:740, Quiesced:220 |

### band 51-100

| B | n_expr | zero-spread (all 12) | zero-spread (guided-only) | qualifies | spread Q1/median/Q3 | unguided differs | top stop reasons |
|---:|---:|---:|---:|:---:|---|---:|---|
| 100 | 61 | 32.8% | 100.0% | no | 0.000/0.100/0.100 | 67.2% (n=61) | ApplicationBudget:732 |
| 200 | 61 | 32.8% | 57.4% | no | 0.000/0.122/0.254 | 67.2% (n=61) | ApplicationBudget:732 |
| 400 | 61 | 32.8% | 32.8% | **YES** | 0.000/0.283/0.283 | 67.2% (n=61) | ApplicationBudget:732 |
| 800 | 61 | 32.8% | 32.8% | **YES** | 0.000/0.273/0.435 | 67.2% (n=61) | ApplicationBudget:512, Quiesced:220 |
| 1600 | 61 | 32.8% | 32.8% | **YES** | 0.000/0.398/0.565 | 67.2% (n=61) | ApplicationBudget:512, Quiesced:220 |
| 3200 | 61 | 32.8% | 32.8% | **YES** | 0.000/0.499/0.657 | 67.2% (n=61) | ApplicationBudget:512, Quiesced:220 |

### band 101-250

| B | n_expr | zero-spread (all 12) | zero-spread (guided-only) | qualifies | spread Q1/median/Q3 | unguided differs | top stop reasons |
|---:|---:|---:|---:|:---:|---|---:|---|
| 100 | 19 | 0.0% | 0.0% | **YES** | 0.171/0.171/0.171 | 100.0% (n=19) | ApplicationBudget:228 |
| 200 | 19 | 0.0% | 0.0% | **YES** | 0.263/0.263/0.263 | 100.0% (n=19) | ApplicationBudget:228 |
| 400 | 19 | 0.0% | 0.0% | **YES** | 0.363/0.363/0.363 | 100.0% (n=19) | ApplicationBudget:228 |
| 800 | 19 | 0.0% | 0.0% | **YES** | 0.363/0.363/0.363 | 100.0% (n=19) | ApplicationBudget:228 |
| 1600 | 19 | 0.0% | 0.0% | **YES** | 0.467/0.467/0.467 | 100.0% (n=19) | ApplicationBudget:228 |
| 3200 | 19 | 0.0% | 0.0% | **YES** | 0.566/0.566/0.566 | 100.0% (n=19) | ApplicationBudget:228 |

