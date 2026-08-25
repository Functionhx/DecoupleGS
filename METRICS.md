# Metric differences from the DecoupleGS paper

This repository is an independent reimplementation. The table below records important numerical differences without treating non-equivalent protocols as a leaderboard comparison.

| Metric | Paper | This repository | Main reason for the difference |
|---|---:|---:|---|
| Open-loop UniAD mADE ↓ | 0.82 m | 1.0246 m | Eight public HUGSIM scenes replace the authors' undisclosed 15 nuScenes clips. |
| Open-loop minTTC ↑ | 3.3 s | 2.4732 s | Public replacement scenes and different traffic clips. |
| Map registration ADE ↓ | 0.05 m | 0.2646 m paper-literal; 0.0255 m with Frenet residual enhancement | The paper cohort is unavailable; the enhanced number adds a non-paper method. |
| Held-out Masked PSNR ↑ | ≈30.0 dB | 18.159 dB | Public temporal holdout, SAM masks, and replacement HDRI calibration differ from the private protocol. |
| Held-out PAE ↓ | 6.8° | 10.902° | Original HDRIs, materials, renderer, and relit supervision are unavailable. |
| 50-agent throughput ↑ | ≈15 FPS | 5.76 FPS | Paper plot uses RTX 4090 and unpublished fused CUDA; local test uses RTX 4070 Ti SUPER and HUGSIM_splat. |
| Closed-loop Driving Score ↑ | 0.884 | Not reported | The paper does not publish its infraction penalty factors. |
| Closed-loop Route Completion ↑ | 0.956 | UniAD 0.568 / 0.566 / 0.521 / 0.319, Easy→Extreme | Different scenes, seeds, and termination details. |
| Closed-loop rendering rate ↑ | 45 FPS | 7.43–15.50 six-camera observations/s, or 44.57–93.03 images/s | The paper does not define whether FPS is an image or a six-camera observation; hardware differs. |

The exact clips, vehicle identities, seeds, HDRIs, planner scoring factors, and fused CUDA rasterizer are not public. The complete local methodology is installed at `third_party/HUGSIM/docs/decouplegs_reimplementation.md` after bootstrap.

Paper source: [DecoupleGS, arXiv:2608.01761](https://arxiv.org/abs/2608.01761).
