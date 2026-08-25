<div align="center">

# DecoupleGS

### Interactive 3D Gaussian Splatting for End-to-End Autonomous Driving Testing

**Unofficial, independent reimplementation of the ECCV 2026 paper**

[![Paper](https://img.shields.io/badge/arXiv-2608.01761-b31b1b.svg)](https://arxiv.org/abs/2608.01761)
[![Base](https://img.shields.io/badge/base-HUGSIM-6f42c1.svg)](https://github.com/hyzhou404/HUGSIM)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

> This is an independent reimplementation from the public paper and supplementary material. It is not the authors' code and is not affiliated with them. The official DecoupleGS implementation was not public when this repository was released.

## Why HUGSIM is a submodule

This is a standalone repository, not a GitHub fork. HUGSIM is pinned as a Git submodule at the exact base revision used for this implementation; our complete DecoupleGS overlay is stored as a reproducible binary Git patch. This preserves clear upstream attribution while avoiding a duplicate copy of HUGSIM's large assets and history.

The overlay implements the object-centric canonical decomposition, real-SH rotation, importance pruning and EMA-VQ, map registration and grounding, local-SH/OLS relighting, contact shadow, unified rasterization, IDM/MOBIL background traffic, and UniAD/VAD evaluation described in the paper.

## Quick start

```bash
git clone --recurse-submodules https://github.com/Functionhx/DecoupleGS.git
cd DecoupleGS
bash scripts/bootstrap_hugsim.sh

cd third_party/HUGSIM
bash tools/setup_decouplegs_conda.sh YOUR_EXISTING_ENV decouplegs
conda activate decouplegs
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
python tools/smoke_test_decouplegs_cuda.py
```

For a fresh environment, run `conda env create -f environment-decouplegs.yml` inside `third_party/HUGSIM` before the setup script. Generated data, checkpoints, and `results/` are intentionally excluded.

## Repository layout

```text
third_party/HUGSIM/          pinned upstream base after submodule initialization
patches/decouplegs.patch.gz  complete, binary-safe implementation overlay
scripts/bootstrap_hugsim.sh  verifies the base revision and applies the overlay
METRICS.md                   concise paper-vs-reproduction metric notes
```

The patch is deliberately transparent: `gzip -cd patches/decouplegs.patch.gz | git -C third_party/HUGSIM apply --stat` shows the complete source-level delta before it is applied.

## Dependencies and data

The implementation reuses HUGSIM and uses public 3DRealCar, nuScenes/PandaSet, SAM, MapTRv2, UniAD, and VAD components. It does not redistribute their checkpoints or datasets.

For the reproduction protocol, commands, public replacement scenes, CUDA checks, and implementation coverage, first run `scripts/bootstrap_hugsim.sh` and then read `third_party/HUGSIM/docs/decouplegs_reimplementation.md`.

<sub>Metric protocol and the deliberately non-equivalent paper comparisons are documented in [METRICS.md](METRICS.md).</sub>
