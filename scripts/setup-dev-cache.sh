#!/usr/bin/env bash
#
# Cargo's `target-dir` defaults to `<checkout>/target`, so every fresh
# `git worktree add` (or fresh clone/container) starts a full rebuild of the
# entire dependency graph from zero -- there is nothing shared between them.
# sccache fixes this the same way CI's rust.yaml/postsubmit-flake-detection.yaml
# workflows do (see their `env:` blocks): it caches compiled objects by a hash
# of (source, compiler flags) rather than by filesystem location, so it works
# across worktrees, branches, and containers without needing them to share a
# `target/` directory (which would risk lock contention between concurrent
# builds) -- it's a content-addressed cache, not a shared-location one.
#
# `incremental` is deliberately LEFT ALONE, and the reason is worth stating
# because the obvious move is to turn it off. sccache cannot cache an
# incremental compilation, so forcing `incremental = false` does raise the
# hit rate -- but it raises it on compilations that were never shared
# anyway, and it costs every human on the machine their edit-rebuild loop.
#
# The two concerns turn out not to overlap. Cargo applies incremental only
# to *workspace/path* crates in the dev profile; registry dependencies are
# always built non-incrementally. And the workspace's own crates can never
# be shared between worktrees regardless (see the `-C metadata` note below),
# while the ~150 registry crates are exactly what sccache does share. So
# with incremental left at its default: workspace crates compile
# incrementally and pass through sccache as non-cacheable, dependencies
# still come from the cache, and nobody gives anything up. Measured on this
# machine, a clean `cargo build -p pixelflow-codegen` with incremental at
# its default created 3 incremental dirs AND took 2 of 2 cacheable compiles
# from cache.
#
# One real footgun, which is why this is documented rather than left
# implicit: setting `CARGO_INCREMENTAL=1` *explicitly* makes sccache refuse
# to run at all -- `sccache: incremental compilation is prohibited: Unset
# CARGO_INCREMENTAL to continue`, and the build fails. Cargo arriving at
# incremental by default is fine; naming it in the environment is not. Do
# not set that variable while `rustc-wrapper` is sccache.
#
# `SCCACHE_CACHE_SIZE` also goes with it, raised from the client's 10G
# default -- but by less than intuition suggests, and the reason why is worth
# writing down so nobody re-inflates it from the wrong evidence later.
#
# It is tempting to size this off `du -sh target/` (tens of GB per worktree
# on this project). Don't: sccache only ever stores objects for compilations
# it can key by (preprocessed source, flags) alone, and two things this
# workspace has a lot of fall outside that --
#   - `--crate-type proc-macro` (pixelflow-compiler, actor-scheduler-macros)
#     and `--crate-type bin`/no-crate-type invocations (every `--test`,
#     `--bench`, and `bin` target `--all-targets` builds) are refused by
#     sccache outright (`CannotCache(crate-type, ...)` in its own trace log).
#   - Every one of this *workspace's own* ~13 crates -- pixelflow-core, -ir,
#     -search, -compiler, -graphics, -runtime, -codegen, core-term, xtask,
#     actor-scheduler, -pipeline, -ml -- misses on every rebuild in a
#     *different* worktree, cache warm or not. Cargo assigns path
#     dependencies a `-C metadata=`/`-C extra-filename=` fingerprint derived
#     from the crate's own absolute manifest path, which becomes part of
#     sccache's Rust cache key; two worktrees never share that path, so two
#     worktrees never share that key. This is Cargo's fingerprinting, not an
#     sccache bug, and no flag here fixes it (confirmed locally: forcing
#     `CARGO_PROFILE_DEV_DEBUG=0` to rule out debuginfo path embedding still
#     missed).
# Net effect, measured on this machine: only this project's ~150 third-party
# dependency crates (built once, at the same `~/.cargo/registry` path
# regardless of which worktree asks) actually get shared. Two full
# `cargo build --workspace --all-targets` runs in two different worktrees,
# dev profile, sequentially -- cold then warm -- grew the on-disk cache to
# ~340 MiB, not gigabytes; the second run's 71.7% Rust hit rate (132/184
# cacheable compiles; ~200 more per run are the crate-type-excluded kind
# above and never enter that ratio) is entirely dependency-graph reuse.
# 10G was still plenty for what's actually cacheable in one profile, but a
# generous multiple covers other profiles/ISA variants (release, bench,
# dist, `xtask isa-matrix`) landing their own copies of that same ~150-crate
# graph in the store without risking eviction thrash. Override with your own
# `SCCACHE_CACHE_SIZE` env var (takes precedence over `[env]` config) if your
# machine's disk budget differs.
#
# This only touches your personal `~/.cargo/config.toml`, never the repo's
# checked-in `.cargo/config.toml` -- forcing either setting on every
# contributor there would break anyone who hasn't installed sccache.
set -euo pipefail

if command -v sccache >/dev/null 2>&1; then
  echo "sccache already installed: $(command -v sccache)"
else
  echo "Installing sccache..."
  cargo install sccache --locked
fi

cargo_config="${CARGO_HOME:-$HOME/.cargo}/config.toml"
mkdir -p "$(dirname "$cargo_config")"
touch "$cargo_config"

# Ensure a `[build]` section exists, then ensure it carries both keys --
# each check is independent so a partially-configured file (e.g. someone set
# rustc-wrapper by hand already) still gets the other key added.
if ! grep -q '^\[build\]' "$cargo_config"; then
  printf '\n[build]\n' >> "$cargo_config"
fi

add_build_key() {
  local key=$1 value=$2
  if grep -q "^${key}[[:space:]]*=" "$cargo_config"; then
    echo "note: $cargo_config already sets $key; leaving it as-is:"
    grep "^${key}[[:space:]]*=" "$cargo_config"
    return
  fi
  awk -v line="${key} = ${value}" '
    /^\[build\]/ && !done { print; print line; done = 1; next }
    { print }
  ' "$cargo_config" > "$cargo_config.tmp"
  mv "$cargo_config.tmp" "$cargo_config"
  echo "Added $key = $value to $cargo_config"
}

add_build_key "rustc-wrapper" '"sccache"'

# Same idempotence contract as add_build_key, but for the `[env]` table --
# cargo forwards these to every rustc (and thus every `sccache rustc ...`
# wrapper) invocation it spawns, regardless of the wrapper process's own cwd.
# That's what makes this reach agents working in arbitrary worktrees: a env
# var exported from a shell profile would not, since agent shells are
# non-interactive and don't source one.
if ! grep -q '^\[env\]' "$cargo_config"; then
  printf '\n[env]\n' >> "$cargo_config"
fi

add_env_key() {
  local key=$1 value=$2
  if grep -q "^${key}[[:space:]]*=" "$cargo_config"; then
    echo "note: $cargo_config already sets $key; leaving it as-is:"
    grep "^${key}[[:space:]]*=" "$cargo_config"
    return
  fi
  awk -v line="${key} = ${value}" '
    /^\[env\]/ && !done { print; print line; done = 1; next }
    { print }
  ' "$cargo_config" > "$cargo_config.tmp"
  mv "$cargo_config.tmp" "$cargo_config"
  echo "Added $key = $value to $cargo_config"
}

add_env_key "SCCACHE_CACHE_SIZE" '"20G"'

echo "Every cargo build/check/test on this machine now shares a compilation cache across worktrees and clones of any project."
echo "note: SCCACHE_CACHE_SIZE is read once, when the sccache server starts."
echo "If a server was already running under the old (client-default 10G)"
echo "limit, run 'sccache --stop-server' once so the next build restarts it"
echo "under the new limit."
