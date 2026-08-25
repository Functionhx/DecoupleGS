# Metric differences from the DecoupleGS paper

This repository is an independent reimplementation. The table below records the important numerical differences without treating non-equivalent protocols as a leaderboard comparison.

| Metric | Paper | This repository | Main reason for the difference |
|---|---:|---:|---|
| Open-loop UniAD mADE ↓ | 0.82 m | 1.0246 m | Eight public HUGSIM scenes replace the authors' undisclosed 15 nuScenes clips; the local value is measured against the real-image planner output. |
| Open-loop minTTC ↑ | 3.3 s | 2.4732 s | Same public replacement scenes and paper-literal center-distance formula, but different clips/traffic. |
| Map registration ADE ↓ | 0.05 m | 0.2646 m paper-literal; 0.0255 m with our Frenet residual enhancement | The paper's map/trajectory cohort is unavailable. The enhanced number adds a method not present in the paper and is not labeled paper-exact. |
| Held-out Masked PSNR ↑ | ≈30.0 dB | 18.159 dB, compact + public OLS + shadow | Ten public tracks, temporal holdout views, SAM ViT-H masks, and replacement HDRI calibration differ from the private author protocol. The paper value is read approximately from its plot. |
| Held-out PAE ↓ | 6.8° | 10.902° | The authors' HDRIs, vehicle materials, target renderer, and relit primitive supervision are not public. Our public OLS proxy did not reproduce the paper's relighting gain. |
| 50-agent throughput ↑ | ≈15 FPS | 5.76 FPS | Paper plot on RTX 4090 with unpublished fused CUDA; local full-runtime test on RTX 4070 Ti SUPER with HUGSIM_splat and Python-side composition. |
| Closed-loop Driving Score ↑ | 0.884 | Not reported | The paper does not publish the infraction penalty factors required by its DS equation, so no factors were guessed or fitted. |
| Closed-loop Route Completion ↑ | 0.956 | UniAD 0.568 / 0.566 / 0.521 / 0.319 from Easy to Extreme | Different scenes, seeds and termination details; the local values are the complete 4×50 public replacement suite rather than one author-matched aggregate. |
| Closed-loop rendering rate ↑ | 45 FPS | 7.43–15.50 six-camera observations/s, or 44.57–93.03 images/s | The paper does not state whether FPS counts a single image or a six-camera observation; local hardware is also different. |

Additional context:

- The exact 15 nuScenes clips, 10 PandaSet sequences, 20 vehicle IDs, scenario seeds, Success Rate rule, Driving Score penalty factors, and FPS aggregation are unpublished.
- The local held-out protocol contains 1,296 views, of which 210 have non-empty primary vehicle masks.
- Compression itself is nearly lossless on that public held-out set: compact/no-relight differs from raw assets by −0.051 dB Masked PSNR and +0.024° PAE.
- Local results and paper targets are frozen in <code>configs/benchmark/decouplegs_paper_targets.yaml</code>; the detailed methodology is in [the reproduction guide](decouplegs_reimplementation.md).

Paper source: [DecoupleGS, arXiv:2608.01761](https://arxiv.org/abs/2608.01761).
