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
# `incremental = false` goes with it, not optionally: rustc's incremental
# compilation state lives in `target/incremental/` and is itself a per-location
# cache, which is exactly what sccache can't see or reuse -- measured locally,
# 161 of 172 compiler invocations in a clean build were "non-cacheable:
# incremental" until this was turned off. The trade is real (editing one file
# and rebuilding in a worktree you've already warmed up loses incremental's
# finer-grained reuse), but it's the right default for what this script is
# for: a *new* worktree/container is exactly the case incremental can't help
# with anyway (there is no prior incremental state to reuse), and that's where
# most of the wasted time in this project's build actually goes. Override
# per-invocation with `CARGO_INCREMENTAL=1` if you're iterating heavily in one
# worktree and want that instead.
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
add_build_key "incremental" "false"

echo "Every cargo build/check/test on this machine now shares a compilation cache across worktrees and clones of any project."
