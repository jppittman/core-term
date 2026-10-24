#!/usr/bin/env bash
# Gate 1: pixelflow-codegen's byte_probe, per ISA tier, must not move.
# usage: byteprobe.sh TAG
# Runs `PIXELFLOW_ISA=<tier> cargo run --release -q -p pixelflow-codegen --example byte_probe`
# for avx512 then avx2 into out/byteprobe.TAG.{avx512,avx2}.txt (stderr to .err).
# TAG=base records both; otherwise each must be cmp-equal to its base, and the
# script exits 0 only if both are.
set -u -o pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib.sh"
check_tag "${1:-}"
TAG=$1
TIERS=(avx512 avx2)
for tier in "${TIERS[@]}"; do
    guard_base "$TAG" "$OUT/byteprobe.base.$tier.txt"
done

cd "$REPO" || die "no $REPO"
status=0
for tier in "${TIERS[@]}"; do
    df_guard
    res=$OUT/byteprobe.$TAG.$tier.txt
    err=$OUT/byteprobe.$TAG.$tier.err
    PIXELFLOW_ISA=$tier cargo run --release -q -p pixelflow-codegen --example byte_probe >"$res" 2>"$err" \
        || { tail -n 30 "$err" >&2; die "byte_probe failed under PIXELFLOW_ISA=$tier; see $err"; }
    [[ -s "$res" ]] || die "byte_probe printed nothing under PIXELFLOW_ISA=$tier"
    echo "$tier: wrote $res ($(wc -l <"$res") lines, sha256 $(sha256sum "$res" | cut -c1-16)...)"
    [[ "$TAG" == base ]] && continue
    echo "== $tier"
    compare_to_base "$res" "$OUT/byteprobe.base.$tier.txt" || status=1
done
[[ "$TAG" == base ]] && { echo "recorded as base"; exit 0; }
exit $status
