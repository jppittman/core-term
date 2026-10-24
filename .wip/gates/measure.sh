#!/usr/bin/env bash
# Gate 5: measured, never asserted. usage: measure.sh TAG
# Builds the probe, runs callgrind on zz_prof_all_cold (out/cg.TAG.out, kept),
# then 5 native runs for median peak RSS (KiB, ru_maxrss as /usr/bin/time -f %M
# reports it; runstat.py, since this host has no /usr/bin/time) and median wall ms.
# JSON on stdout and in out/measure.TAG.txt; add a row to the ledger.
# TAG=base records out/measure.base.txt; otherwise the JSON also carries the
# change against it. Exit 0 whenever the measurement ran.
set -u -o pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib.sh"
check_tag "${1:-}"
TAG=$1
RUNS=5
RES=$OUT/measure.$TAG.txt
CG=$OUT/cg.$TAG.out
guard_base "$TAG" "$OUT/measure.base.txt" "$OUT/cg.base.out"

build_probe measure

echo "callgrind -> $CG" >&2
valgrind --tool=callgrind --callgrind-out-file="$CG" \
    "$BIN" --ignored --nocapture --test-threads=1 zz_prof_all_cold \
    >"$OUT/cg.$TAG.stdout" 2>"$OUT/cg.$TAG.err" || die "callgrind run failed; see $OUT/cg.$TAG.err"
[[ -s "$CG" ]] || die "callgrind wrote no $CG"

rss=() wall=()
for ((i = 1; i <= RUNS; i++)); do
    # runstat.py prints "<peak RSS KiB> <wall ms>": ru_maxrss from wait4, the
    # figure `/usr/bin/time -f %M` prints (this host has no /usr/bin/time).
    stat=$(python3 "$GATES/runstat.py" "$BIN" --ignored --nocapture --test-threads=1 zz_prof_all_cold \
        2>"$OUT/measure.$TAG.native.err") || die "native run $i failed; see $OUT/measure.$TAG.native.err"
    read -r kib ms <<<"$stat"
    [[ -n "${kib:-}" && -n "${ms:-}" ]] || die "native run $i printed no measurement; see $OUT/measure.$TAG.native.err"
    rss+=("$kib")
    wall+=("$ms")
done
join() { local IFS=,; echo "$*"; }

base_json=()
[[ "$TAG" != base && -s "$OUT/measure.base.txt" ]] && base_json=("$OUT/measure.base.txt")
python3 "$GATES/measure_report.py" "$TAG" "$CG" "$(join "${rss[@]}")" "$(join "${wall[@]}")" "${base_json[@]}" \
    >"$RES.tmp" || die "could not report $CG"
mv "$RES.tmp" "$RES"
cat "$RES"
exit 0
