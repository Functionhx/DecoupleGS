# DecoupleGS real-checkpoint visual demo

This directory is a persistent, reproducible render of HUGSIM `scene-0013`
with the 3DRealCar asset `2024_07_05_10_58_02` placed 14 metres ahead of the
front camera at 400 x 225 resolution.

The main artifacts are:

- `effects_full.png`: background plus all cumulative effect stages;
- `effects_zoom.png`: automatically cropped vehicle comparison;
- `effects_difference.png`: 20x adjacent-stage difference maps;
- `metrics.json`: inputs, timings, and pixel-change metrics;
- `00_raw.png` through `04_full_shadow.png`: individual stages.

Reproduce from the repository root with:

```bash
conda activate decouplegs
python tools/render_decouplegs_effects.py \
  /path/to/hugsim/scenes/nuscenes/scene-0013 \
  /path/to/3DRealCar/2024_07_05_10_58_02/gs.pth \
  artifacts/decouplegs-effects-demo \
  --scale 0.5 --distance 14
```

The public asset has no author-provided HDRI OLS calibration sidecar, so this
demo records its relighting mode as `adaptive_fallback` rather than claiming a
paper-exact OLS result.
