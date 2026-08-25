#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CUDA_ID OUTPUT_DIR" >&2
  exit 2
fi

planner_root="${VAD_SIM_ROOT:-}"
planner_env="${DECOUPLEGS_AD_ENV:-decouplegs-ad}"
if [[ -z "$planner_root" || ! -d "$planner_root" ]]; then
  echo "Set VAD_SIM_ROOT to a VAD_SIM checkout." >&2
  exit 3
fi
cd "$planner_root"
exec conda run --no-capture-output -n "$planner_env" env CUDA_VISIBLE_DEVICES="$1" \
  python tools/closeloop/e2e.py \
  projects/configs/VAD/VAD_base_e2e.py ckpts/VAD_base.pth \
  --launcher none --eval bbox --tmpdir tmp --seed 0 --deterministic --output "$2"
