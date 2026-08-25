<div align="center">

# DecoupleGS

### Interactive 3D Gaussian Splatting for End-to-End Autonomous Driving Testing

**Unofficial, independent reimplementation of the ECCV 2026 paper**

[![Paper](https://img.shields.io/badge/arXiv-2608.01761-b31b1b.svg)](https://arxiv.org/abs/2608.01761)
[![Base](https://img.shields.io/badge/base-HUGSIM-6f42c1.svg)](https://github.com/hyzhou404/HUGSIM)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Siying Li · Ying Ni · Jie Sun · Jian Sun · Haotian Shi<br>
<sub>Original paper authors — this repository is not affiliated with or endorsed by them.</sub>

<img src="artifacts/decouplegs-effects-demo/effects_full.png" alt="DecoupleGS cumulative module visualization" width="100%">

<sub>Static background, canonical insertion, real-SH rotation, opacity grounding, relighting, and contact shadow on a public HUGSIM/3DRealCar checkpoint.</sub>

</div>

> [!IMPORTANT]
> This is an **independent reimplementation from the public paper and supplementary material**, not the authors' code. It intentionally uses HUGSIM as a pinned submodule rather than copying or forking its repository.

## Overview

DecoupleGS separates a driving scene into a persistent static 3DGS background and independently controllable canonical vehicle assets. At runtime each vehicle is compressed, relit, registered to the map and road surface, transformed into the world frame, and composed with the background for one unified rasterization pass.

```text
static background ─────────────────────────────────────────────┐
canonical assets → VQ decode → relight → SE(3) / SH rotation ─┼→ unified primitive stream
HD map + trajectory → DTW → SE(2) → opacity grounding ────────┘             │
                                                                            ▼
                                                             one CUDA rasterization
                                                                            │
                                                             UniAD / VAD closed loop
```

The implementation follows the disclosed settings: 30k background iterations, 20k vehicle iterations, 0.005 pruning threshold, 1024/512 codebooks, 0.99 EMA, five-pixel mask dilation, 27-D local SH probes, and a 2 Hz closed-loop policy update.

## Visual results

<p align="center">
  <img src="artifacts/decouplegs-effects-demo/effects_zoom.png" alt="Vehicle crop module comparison" width="82%">
</p>

The checked-in visual is generated from a public HUGSIM scene and 3DRealCar asset. It shows the cumulative effect of canonical insertion, SH rotation, opacity grounding, relighting, and contact shadows. The original HDRI supervision and target renderer are not public, so the relighting stage uses the documented public replacement calibration.

## Implemented components

| Paper component | Implementation status |
|---|---|
| Object-centric canonical decomposition | Implemented |
| Degree 0–3 real-SH Wigner-D rotation | Implemented and equivariance-tested |
| Importance pruning + K-Means / EMA VQ | Implemented |
| DTW + SE(2) Orthogonal Procrustes + grounding | Implemented |
| Local SH probe + affine OLS relighting | Implemented with a public replacement protocol |
| Parametric contact shadow | Implemented |
| Unified static/dynamic rasterization | Implemented on HUGSIM_splat |
| IDM/MOBIL background traffic | Implemented |
| UniAD / VAD closed-loop evaluation | Implemented |

The paper's fused CUDA kernels and exact evaluation assets are unpublished. This reimplementation uses a one-call HUGSIM_splat path plus an independently developed static-stream cache and stable dynamic insertion path.

## Quick start

```bash
git clone --recurse-submodules https://github.com/Functionhx/DecoupleGS.git
cd DecoupleGS

# Verify the pinned HUGSIM base and apply the complete DecoupleGS overlay.
bash scripts/bootstrap_hugsim.sh

cd third_party/HUGSIM
bash tools/setup_decouplegs_conda.sh YOUR_EXISTING_ENV decouplegs
conda activate decouplegs

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
python tools/smoke_test_decouplegs_cuda.py
```

For a fresh environment, create `environment-decouplegs.yml` inside `third_party/HUGSIM` first. The setup supports cloning an existing compatible Conda environment so cached packages and compiled dependencies are reused.

## Reproduction coverage

The public replacement protocol was exercised on 20 3DRealCar assets, eight reconstructed nuScenes scenes, 1,296 temporally held-out re-insertion views, 400 strict closed-loop episodes, and dense runtime tests up to 50 dynamic agents. CPU numerical tests plus CUDA forward/backward and raster-equivalence smoke tests pass.

These results show that the released pipeline is executable end to end; they do not claim equivalence to the authors' private clips, HDRIs, vehicle identities, planner seeds, or unpublished fused kernels.

## Repository layout

```text
third_party/HUGSIM/          HUGSIM base, pinned as a Git submodule
patches/decouplegs.patch.gz  complete binary-safe DecoupleGS source overlay
scripts/bootstrap_hugsim.sh  validates the base commit and applies the overlay
artifacts/                   checked-in visual results for this implementation
METRICS.md                   paper-vs-reproduction metric notes
```

The overlay is inspectable before application:

```bash
gzip -cd patches/decouplegs.patch.gz | git -C third_party/HUGSIM apply --stat
```

## Data and dependencies

This repository does not redistribute nuScenes, PandaSet, SAM weights, planner weights, or 3DRealCar checkpoints. HUGSIM, 3DRealCar, nuScenes/PandaSet, SAM, MapTRv2, UniAD, and VAD remain their respective authors' projects and licenses.

<sub>Metric protocol and deliberate non-equivalent comparisons: [METRICS.md](METRICS.md).</sub>
