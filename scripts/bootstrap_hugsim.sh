#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base_dir="$root_dir/3rdparty/HUGSIM"
overlay_dir="$root_dir/overlay"
base_commit="62c690d39fd90020e68a196bd8bcc1c4d4191f2e"

if [[ ! -d "$base_dir/.git" && ! -f "$base_dir/.git" ]]; then
  git -C "$root_dir" submodule update --init --recursive
fi

actual_commit="$(git -C "$base_dir" rev-parse HEAD)"
if [[ "$actual_commit" != "$base_commit" ]]; then
  echo "HUGSIM is at $actual_commit; expected $base_commit." >&2
  echo "Run: git -C 3rdparty/HUGSIM checkout --detach $base_commit" >&2
  exit 1
fi

if [[ ! -d "$overlay_dir" ]]; then
  echo "Missing source overlay: $overlay_dir" >&2
  exit 1
fi

cp -a "$overlay_dir/." "$base_dir/"
echo "Copied the DecoupleGS source overlay to 3rdparty/HUGSIM."
