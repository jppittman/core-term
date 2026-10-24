#!/usr/bin/env bash
# Shared by the gate scripts. Source it; it is not a command.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
#
# Exit codes the scripts use:  0 = recorded / identical,  1 = differs from base,
# 2 = the harness itself failed (build, probe output, bad arguments),
# 3 = refused to start (disk under 2 GB, tests.rs already modified, lock held).

set -u -o pipefail

REPO=/home/user/pixelflow
GATES=$REPO/.wip/gates
OUT=$GATES/out
BINDIR=/home/user/scratch/gates-bin
PROBES=$GATES/probes-gate1b.rs
TESTS_RS=pixelflow-graphics/src/fonts/loop_blinn/program/tests.rs
MIN_FREE_BYTES=$((2 * 1024 * 1024 * 1024))

mkdir -p "$OUT" "$BINDIR"

die() { echo "$(basename "$0"): $*" >&2; exit "${DIE_CODE:-2}"; }

# A tag is a file-name fragment: it names the outputs this run overwrites.
check_tag() {
    [[ "${1:-}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "usage: $(basename "$0") TAG   (TAG = letters, digits, . _ -)"
}

# `base` is reserved for the recorded bases: re-recording needs GATES_REBASE=1,
# so a stray `TAG=base` run cannot replace what every comparison reads.
guard_base() { # TAG FILE...
    local tag=$1
    shift
    [[ "$tag" == base && "${GATES_REBASE:-0}" != 1 ]] || return 0
    local f
    for f in "$@"; do
        [[ ! -s "$f" ]] || DIE_CODE=3 die "$f already holds a recorded base; set GATES_REBASE=1 to replace it deliberately"
    done
}

# `df -h /` before every build; under 2 GB free, stop.
df_guard() {
    df -h / | tail -n 1 >&2
    local avail
    avail=$(df -B1 --output=avail / | tail -n 1 | tr -d ' ')
    [[ "$avail" =~ ^[0-9]+$ ]] || die "could not read free space from df"
    ((avail >= MIN_FREE_BYTES)) || DIE_CODE=3 die "STOP: under 2 GB free on / ($avail bytes)"
}

restore_tests() {
    git -C "$REPO" checkout -- "$TESTS_RS"
}

# build_probe NAME [cargo args...]
#
# Builds pixelflow-graphics' release lib test binary with the probes appended to
# tests.rs, copies it to $BINDIR/bin.NAME and sets BIN to that path. Call it
# directly (not in $(...)), because BIN is its result. It owns the EXIT/INT/TERM
# trap: tests.rs is restored by the trap and again right after the build.
build_probe() {
    local name=$1
    shift
    cd "$REPO" || die "no $REPO"
    df_guard

    # One build at a time: two scripts appending the probes would race.
    exec 9>"$GATES/.build.lock"
    flock -n 9 || DIE_CODE=3 die "another gate build holds $GATES/.build.lock"

    [[ -z "$(git status --porcelain -- "$TESTS_RS")" ]] \
        || DIE_CODE=3 die "$TESTS_RS is modified; refusing to start (restore it first)"
    [[ -r "$PROBES" ]] || die "missing $PROBES"

    trap restore_tests EXIT INT TERM
    cat "$PROBES" >>"$TESTS_RS" || die "could not append the probes"

    local json=$BINDIR/build.$name.json log=$BINDIR/build.$name.log
    cargo test --release -p pixelflow-graphics --lib --no-run --message-format=json "$@" \
        >"$json" 2>"$log"
    local status=$?
    restore_tests
    exec 9>&-
    if ((status != 0)); then
        tail -n 40 "$log" >&2
        die "cargo build failed ($status); see $log"
    fi
    [[ -z "$(git status --porcelain -- "$TESTS_RS")" ]] || die "$TESTS_RS is still modified after the build"
    # Restored: nothing more for the trap to do, and it must not revert an
    # edit someone makes while the probe runs.
    trap - EXIT INT TERM

    local exe
    exe=$(python3 - "$json" <<'PY'
import json, sys
found = []
for line in open(sys.argv[1]):
    try:
        msg = json.loads(line)
    except ValueError:
        continue
    if msg.get("reason") != "compiler-artifact" or not msg.get("executable"):
        continue
    target = msg["target"]
    if target["name"] == "pixelflow_graphics" and "lib" in target["kind"] and msg["profile"]["test"]:
        found.append(msg["executable"])
if len(found) != 1:
    sys.exit("expected exactly one pixelflow_graphics lib test executable, found %r" % found)
print(found[0])
PY
    ) || die "could not find the test executable in $json"
    BIN=$BINDIR/bin.$name
    cp "$exe" "$BIN" || die "could not copy $exe"
    echo "built $BIN (from $exe)" >&2
}

# probe_lines FILE: the ZZGATE lines of a probe's stdout, with libtest's
# "test ... " prefix glued onto the first one stripped.
probe_lines() {
    sed -n 's/.*\(ZZGATE .*\)$/\1/p' "$1"
}

# compare_to_base NEW BASE: 0 and "IDENTICAL" if the files are byte-equal, else
# 1 and the first lines of `diff BASE NEW`.
compare_to_base() {
    [[ -r "$2" ]] || DIE_CODE=2 die "no recorded base $2; run this script with TAG=base at a clean tree first"
    if cmp -s "$1" "$2"; then
        echo "IDENTICAL to $(basename "$2")"
        return 0
    fi
    echo "DIFFERS from $(basename "$2") (< base, > $(basename "$1")):"
    diff "$2" "$1" | head -n 60
    return 1
}
