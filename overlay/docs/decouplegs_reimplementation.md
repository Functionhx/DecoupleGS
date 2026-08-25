# DecoupleGS：独立可执行复刻

本分支在 HUGSIM 上实现了 DecoupleGS 论文公开的完整算法链，并把静态背景与任意数量的 canonical vehicle Gaussians 合并后交给 **一次** HUGSIM_splat CUDA rasterization。对应论文为 [DecoupleGS: Interactive 3D Gaussian Splatting for End-to-End Autonomous Driving Testing](https://arxiv.org/abs/2608.01761)，底座为 [HUGSIM](https://github.com/hyzhou404/HUGSIM)。

截至 2026-08-25，论文页面没有提供作者代码或其 custom CUDA rasterizer，因此这里是根据正文和 Supplementary 完成的独立实现，不声称与未发布实现 bit-exact。

## 已实现模块

| 论文模块 | 本仓库实现 | 验证方式 |
|---|---|---|
| Object-centric canonical decomposition | `decouplegs/types.py`, `transforms.py` | SE(3) mean/covariance 与 quaternion 数值性质测试 |
| Real-SH Wigner-D rotation | `RealSHRotator` | degree 0–3 球谐旋转等变测试 |
| Importance pruning + covariance/SH VQ | `compression.py` | pruning、K-Means++、EMA、dead-code、存取/解码测试 |
| DTW + SE(2) Procrustes | `registration.py` | lane selection 与刚体变换恢复测试 |
| Opacity vertical grounding | `registration.py` | opacity column accumulation 与斜平面恢复测试 |
| Local SH probe + affine OLS relighting | `relighting.py`, `hdri_protocol.py` | 20 车、12 HDRI 的 train/validation/test 标定与逐环境 gate |
| Superellipse contact shadow | `relighting.py` | mask 衰减与背景 SH attenuation 测试 |
| Unified online composition/rasterization | `rasterizer.py`, `runtime.py`, `gaussian_renderer/__init__.py` | CUDA 前向/反向 smoke test；静态 radix cache 与全量路径逐像素对账 |
| HUGSIM closed loop | `hug_sim.py`, `closed_loop.py` | HUGSIM train/Gym 完整导入测试 |
| IDM/MOBIL 与论文指标 | `behavior.py`, `metrics.py` | 独立参考实现与单元测试 |

在线数据流为：

```text
static background ───────────────────────────────┐
canonical assets → VQ decode → relight → SE(3) ─┼→ one primitive list
HD map/trajectory → DTW → SE(2) → grounding ────┘         │
                                                           v
                                             one CUDA rasterization
                                                           │
                                             UniAD/VAD → state update
```

## 已创建并验证的环境

当前验证环境通过克隆已有兼容 Conda 环境创建，因此复用了 package cache 和原环境中绝大多数包：

```bash
conda activate decouplegs
python --version
# Python 3.11.15

python -c "import torch; print(torch.__version__, torch.version.cuda)"
# 2.4.1+cu118 11.8
```

环境名为 `decouplegs`，实际安装路径由本机 Conda 决定。关键版本与 HUGSIM 官方 `pixi.toml` 对齐：Torch 2.4.1、torchvision 0.19.1、CUDA 11.8、NumPy 1.26.4、Open3D 0.18.0。`simple-knn`、`HUGSIM_splat`、`tiny-cuda-nn` 和 UniDepth KNN 四个 CUDA 扩展会为当前 GPU 架构重编译。

在另一台相同软件栈的机器上可复用已有环境：

```bash
bash tools/setup_decouplegs_conda.sh YOUR_EXISTING_ENV decouplegs
```

脚本使用 GCC/G++ 11 与 `/usr/local/cuda-11.8`，并锁定源码依赖到本次验证的 commit。没有可复用环境时，可先从 `environment-decouplegs.yml` 创建基础环境，再把它作为脚本的第一个参数。

## 快速验收

```bash
conda activate decouplegs

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
python tools/smoke_test_decouplegs_cuda.py
pip check
```

预期结果是全部测试通过、CUDA smoke test 报告 `simple-knn=OK`、`unidepth-knn=forward+backward OK` 和 `unified-rasterizer=forward+backward OK`、`pip check` 无破损依赖。

当前严格测试集为 63 项，另对 UniAD/VAD 外部推理适配器执行 `py_compile` 和 diff whitespace 检查。

下载任一 HUGSIM 官方导出场景和 3DRealCar 资产后，还可做真实 checkpoint 验收：

```bash
python tools/smoke_test_decouplegs_real.py \
  /path/to/extracted/scene-0013 /path/to/3DRealCar/car-id/gs.pth \
  --output artifacts/decouplegs-real-smoke.png
```

该命令加载完整静态场景与车辆，执行自动落地、SE(3)/SH 变换、primitive 合并和一次 CUDA rasterization，并报告 Gaussian 数、有限值、耗时和显存峰值。本机以官方 `scene-0013`（1,814,109 个背景 Gaussian）和 `2024_07_05_10_58_02`（402,492 个车辆 Gaussian）实测通过；一次预热后，400×225 单帧验收为 0.22 s、峰值约 2.81 GiB。该数字包含 Python 组合开销，不作为论文吞吐量对比。

## 实际效果与逐模块消融

不仅可以检查“能否渲染”，还可以在同一相机、同一车辆 pose 和同一 CUDA rasterizer 下逐档开启 DecoupleGS 效果：

```bash
python tools/render_decouplegs_effects.py \
  /path/to/extracted/scene-0013 \
  /path/to/3DRealCar/car-id/gs.pth \
  artifacts/decouplegs-effects-demo \
  --scale 0.5 --distance 14
```

输出包括：

- `00_raw.png`：只做 canonical vehicle SE(3) 插入；
- `01_sh_rotation.png`：再启用 Real-SH Wigner-D rotation；
- `02_grounding.png`：再启用 opacity-column vertical grounding；
- `03_relighting.png`：再启用 local SH probe relighting；
- `04_full_shadow.png`：再启用 superellipse contact shadow；
- `effects_full.png`：含纯背景的全图对照；
- `effects_zoom.png`：根据车辆影响区域自动裁剪并放大；
- `effects_difference.png`：相邻阶段的放大差分热图；
- `metrics.json`：逐阶段 MAE、最大差值、变化像素比例和耗时。

若显式传入 `--calibration relighting.pt`，或资产同目录存在该文件，重光照使用论文的 OLS affine operator；`metrics.json` 会继续读取 sidecar 元数据区分 `paper_exact_ols`、`public_hdri_*_proxy_ols` 和来源未知的 OLS，不再仅凭文件名把标定写成 exact。公开的 HUGSIM/3DRealCar 资产目前不附带作者 HDRI 标定 sidecar；未传 sidecar 时工具明确写入 `adaptive_fallback`，使用 local probe 的 GraphDeco DC RGB 做曝光与色温迁移。

当前 OLS 效果图使用 1,814,109 个背景 Gaussian 和 140,521 个 visibility-gated 车辆 Gaussian。在 400×225、14 m 插入距离下，SH rotation、grounding、public-HDRI covariance-diffuse OLS、contact shadow 四步相对前一步分别有 `1.422% / 1.774% / 1.663% / 1.130%` 的像素变化超过 `1/255`。真实 local descriptor 的 27/27 个维度全部位于标定 fit range 内，最大绝对 z-score 为 `2.183`，因此这张图没有依赖 HDRI 域外的危险线性外推。图像、放大差分和审计元数据在 `results/decouplegs/relighting/public-hdri-cohort-v1/qualitative-scene0013/`。

## 数据和背景重建

HUGSIM 的 nuScenes loader 按上游目录结构从 `data/` 运行：

```bash
conda activate decouplegs
cd data
python nusc/load.py --help
```

生成静态背景 mask 时，传入 SAM checkpoint 即启用 box-prompted SAM；未传入时保留 HUGSIM semantic-box fallback。两条路径都会执行论文指定的 5 pixel morphology dilation：

```bash
python utils/create_dynamic_mask.py \
  --data_path /path/to/processed/scene \
  --data_type nuscenes \
  --sam_checkpoint /path/to/sam_vit_h_4b8939.pth \
  --dilation 5
```

随后沿用 HUGSIM 的 ground/background reconstruction：

```bash
cd ..
python train_ground.py --data_cfg configs/nusc.yaml \
  --source_path /path/to/processed/scene --model_path /path/to/model
python train.py --base_cfg configs/decouplegs_background.yaml --data_cfg configs/nusc.yaml \
  --source_path /path/to/processed/scene --model_path /path/to/model \
  --ignore_dynamic --disable_affine
```

`--ignore_dynamic` 会让 Scene 排除动态实例并把生成的 mask 应用于 RGB loss；`--disable_affine` 关闭 HUGSIM 的 appearance MLP，保持论文所述 vanilla 3DGS 背景，两者都不要省略。`configs/decouplegs_background.yaml` 精确采用 Supplementary Table 5 的 30,000 iterations、degree-3 SH 和 `5e-3` scaling learning rate；HUGSIM 上游 `configs/gs_base.yaml` 的对应值为 `1e-3`，因此没有直接复用。车辆使用 3DRealCar/HUGSIM 发布的 canonical `gs.pth`；若从原始多视图数据重训车辆，论文规定 20,000 iterations 和 `4e-4` asset densification threshold。

训练完成后按 HUGSIM 格式导出，closed loop 的 `model_base` 应指向导出目录的父目录：

```bash
python eval_render/export_scene.py \
  --model_path /path/to/model --output_path /path/to/exported/scene-name \
  --iteration 30000
```

## 压缩 canonical vehicle

把输出写到车辆 `gs.pth` 同一目录，HUGSIM runtime 会自动优先加载 `decouplegs.dgs`：

```bash
python tools/compress_decouplegs_asset.py \
  /path/to/car/gs.pth /path/to/car/decouplegs.dgs
```

默认值严格采用 Supplementary：importance 权重 `0.5 / 0.3 / 0.2`、pruning threshold `0.005`、shape codebook `512`、color codebook `1024`、EMA momentum `0.99`、EMA update `5000` 次。完整 5000 次更新会耗时；测试时可临时减小 `--ema-iterations`，正式资产不要减。

论文没有给出 local color contrast/texture entropy 的精确 neighborhood 代码。本实现提供确定性的 voxel-neighborhood fallback，也允许输入训练视图统计来替换它：

```python
torch.save({
    "visibility": visibility,          # [N]
    "color_contrast": color_contrast, # [N]
    "texture_entropy": texture_entropy, # [N]
}, "importance.pt")
```

```bash
python tools/compress_decouplegs_asset.py car/gs.pth car/decouplegs.dgs \
  --importance-stats importance.pt
```

论文没有公布所用的 20 个 3DRealCar ID。本仓库冻结了 20 辆公开资产的替代 cohort，并对每辆资产用同一组 72 个 orbit view 估计可见度、执行正式 5000 次 EMA VQ、再对 raw checkpoint 做配对渲染：

| 指标 | 20-asset cohort |
|---|---:|
| raw / compact primitives | 12,260,540 / 3,660,565 |
| 总体保留率 | 29.856% |
| raw / compact tensor bytes | 3.874 GB / 77.39 MB |
| 总体内存压缩比 | 50.06× |
| PSNR-All（20 辆均值） | 35.058 dB |
| PSNR-Vehicle（20 辆均值） | 26.043 dB |
| SSIM（20 辆均值） | 0.99123 |
| LPIPS-Alex（20 辆均值） | 0.00858 |

完整 manifest、逐资产 hash 和指标在 `configs/benchmark/decouplegs_assets_20.yaml` 与 `results/decouplegs/asset-cohort/cohort-summary.json`。这些数字复刻的是公开替代 cohort 上的压缩质量，不能伪装成作者未公开的 20 辆资产。

### 运行时可扩展性

`tools/benchmark_decouplegs_scalability.py` 在真实 scene-0013 背景、800×450、moving assets、完整 grounding/relighting/contact-shadow/runtime 路径上测试 1/5/10/20/50 辆车。当前 visibility-gated compact asset 在 RTX 4070 Ti SUPER 上结果如下：

| 动态车数 | FPS ↑ | mean latency ↓ | peak allocated ↓ |
|---:|---:|---:|---:|
| 1 | 41.15 | 24.30 ms | 2863 MiB |
| 5 | 28.22 | 35.44 ms | 3233 MiB |
| 10 | 20.31 | 49.23 ms | 3264 MiB |
| 20 | 12.44 | 80.36 ms | 3520 MiB |
| 50 | 5.76 | 173.72 ms | 4233 MiB |

50 车没有 OOM，显存随实例数平稳增长；未使用 visibility-gated pruning 的同一路径在 50 车时为 `4.00 FPS / 6448 MiB`。这组是单相机完整 runtime FPS，硬件也不是论文的 RTX 4090，因此只报告实测值和趋势，不把它换算成论文的 45 FPS。机器可读结果为 `results/decouplegs/full-runtime-scalability-{visgate,no-lod}-4070ti.json`。

论文 runtime optimization 所述的静态背景排序复用现已落成独立的 exact cache 路径。缓存只保留相机不变时的 background projection、tile coverage、depth radix stream 和 offsets；背景 SH/semantic feature 每帧重新求值，因此车辆移动造成的 local relighting/contact shadow 不会被冻结。动态 tile/depth stream 单独排序后插入静态 stream：稀疏动态流只对 dynamic keys 做 CUDA search，达到静态 intersection 的 `0.5×` 后切换为自研 Triton stable merge-path kernel；当动态可见 primitive 超过背景的 `1.25×` 时自动退回完整 radix。cache miss 直接从当帧完整 radix 结果提取静态子序列，不会为了填 cache 重复排序背景。

scene-0013、800×450 的配对测试（相同进程、相同 checkpoint、完整 grounding/relighting/contact-shadow 路径）如下：

| 动态车数 | full radix | static cache | 提升 | 选择的排序路径 |
|---:|---:|---:|---:|---|
| 1 | 60.45 FPS | 67.85 FPS | +12.2% | dynamic search merge |
| 5 | 39.52 FPS | 42.20 FPS | +6.8% | dynamic search merge |
| 10 | 27.50 FPS | 28.31 FPS | +3.0% | Triton merge-path |
| 20 | 16.34 FPS | 16.44 FPS | +0.6% | full-radix fallback |
| 50 | 6.71 FPS | 6.71 FPS | -0.04% | full-radix fallback |

单相机 cache 占用 `120.69 MiB`。独立验证器把车辆整体移动 `3.1 cm` 后，1/10 车增量路径以及 20 车回退路径的 RGB、alpha 都与禁用 cache 的完整 radix 输出 bit-exact，最大误差为 `0`；更换 camera key 会 miss 并按 one-entry LRU 驱逐旧条目。六相机 batched sensor 回归为 `105.45 dB`，只有 uint8 量化边界上的极少数 `1/255` 差异。对应证据为：

- `results/decouplegs/static-cache-equivalence.json`
- `results/decouplegs/static-cache-full-runtime-{baseline,optimized}-10x.json`
- `results/decouplegs/camera-batch-static-cache-equivalence-scene0013.json`

复现实验：

```bash
python tools/validate_decouplegs_static_cache.py \
  /path/to/scene-0013 /path/to/car/decouplegs-visgate.dgs \
  --output results/decouplegs/static-cache-equivalence.json
```

这一版复用的是**相机姿态完全相同**的静态排序。相机 rig 连续出现两次才获准建 cache；闭环中持续变化的 ego camera 不会填充或抖动这块约 121 MiB 的 LRU，离开固定视角时旧条目也会释放。论文未公开的跨相机小位姿增量更新/custom fused merge 仍是下一阶段系统优化，不能把当前 exact cache 描述成作者 kernel 的完整替代。

## HD map registration

输入 trajectory 为 `.npy [N,2]`；lane archive 可以是 `.npz` 中的 `lanes [L,N,2]`、object-array lanes，或多个 `lane_*` 数组：

```bash
python tools/register_decouplegs_trajectory.py \
  trajectory.npy lane_centerlines.npz registration.npz \
  --heading-weight 2.5 --map-resolution 0.1
```

lane polyline 会先按论文给出的 `0.1 m/pixel` 等效间距重新采样。输出包含 corrected trajectory、SE(2) matrix、DTW path、selected lane 和 normalized cost。无原生 HD map 时，应先用 MapTRv2 生成 lane centerlines；MapTR 的 mmcv/mmdet 依赖与 HUGSIM 冲突较多，建议保留其独立环境，只交换 `.npz`，不要污染 `decouplegs` 环境。

该 CLI 是场景预处理边界：原生 HD map 场景可继续让 HUGSIM planner 读取 map；MapTRv2 路径则应由 scenario/controller 消费输出的 corrected trajectory 或 SE(2) matrix。当前工具不会静默改写 HUGSIM planner state，避免渲染 pose 与碰撞/行为状态产生两套坐标。

scene-0013 的 180-pose 实车 track 定量结果为：无注册到拓扑路线 ADE `0.4635 m`，论文原文的 DTW + global SE(2) Procrustes 为 `0.2646 m`；针对弯曲长轨迹新增的单调 Frenet residual projection 为 `0.0255 m`。后者是明确标注 `H-GEO-01` 的自研增强，不归入 paper-literal 成绩。opacity grounding 在 180 帧、720 个底部 anchor 上全部成功；相对拟合局部平面的 `1 mm` tolerance penetration rate 为 `0.0%`。原始日志资产平均悬空约 `1.60 m`，所以 raw pose 的零 penetration 不是正确接地。结果在 `results/decouplegs/geometry-scene0013-full.json`；正式结论仍需扩展到八场景多 track。

## 重光照标定

准备 synthetic HDRI samples：

```python
torch.save({
    "descriptors": descriptors, # [P,27]，背景前三个 SH bands
    "canonical": canonical_sh,  # [M,C,3] 或 [P,M,C,3]
    "targets": target_sh,       # [P,M,C,3]
}, "hdri_samples.pt")
```

然后做论文给出的 closed-form OLS：

```bash
python tools/calibrate_decouplegs_relighting.py \
  hdri_samples.pt /path/to/car/relighting.pt
```

`relighting.pt` 与 `decouplegs.dgs` 同目录时会自动加载。OLS 直接累积 56×56 normal equations，不会展开随 Gaussian 数增长的巨型设计矩阵。论文固定 local descriptor 为前三个 SH bands（`9 × RGB = 27`）以及 superellipse exponent `4`。`probe_sigma/radius`、shadow strength/decay 和 opacity column radius 等未公开 kernel 细节都暴露在 `configs/decouplegs.yaml`，便于用验证集标定。

作者的 HDRI、车辆 mesh/material、target renderer 和 relit primitive colors 均未发布，因此本仓库冻结了一套 **public replacement protocol**，绝不把它命名为论文 heldout：8 个完整 HDRI 训练、2 个完整 HDRI 选择 identity-prior ridge、2 个完整 HDRI 最终测试；每个环境展开 4 个 yaw、3 个 exposure、3 个 tint，共 `288 / 72 / 72` 个 train/validation/test probes。一个 HDRI 的所有变体严格留在同一 split。每辆车固定抽取 2048 个 decoded primitives，冻结 cohort 为同一批 20 个公开 3DRealCar 资产。

```bash
python tools/benchmark_decouplegs_relighting_cohort.py
```

同时报告两种不同证据，不能混为一谈：

| held-out primitive proxy（20 车均值） | 用途 | supported PAE ↓ | supported p95 ↓ | peak intensity error ↓ | primitive RGB PSNR ↑ | 逐 test HDRI gate |
|---|---|---:|---:|---:|---:|---:|
| no relight → global-affine OLS | Eq. (15) 可解算子 oracle | 7.960° → 0.005° | 36.259° → 0.028° | 0.1951 → 0.000012 | 17.79 → 108.80 dB | 20/20 |
| no relight → covariance-diffuse OLS | covariance normal + cosine irradiance 物理压力测试 | 10.874° → 3.610° | 22.176° → 14.421° | 0.7351 → 0.4875 | 8.73 → 17.18 dB | 20/20 |

oracle 只证明我们的全局 OLS 实现能恢复论文声明的算子类，所以永远不具备 runtime deployment 资格。covariance-diffuse 在两个独立 test HDRI 上逐环境通过 chromaticity、p95、intensity、PSNR 四个 gate，20 辆 sidecar 都标为 `experimental_runtime_eligible`；但它仍缺少作者材质与光追监督，元数据固定为 `paper_exact: false`。sidecar 只写入 `results/decouplegs/relighting/public-hdri-cohort-v1/assets/`，不会复制成车辆目录旁的 `relighting.pt`，必须显式传入才会使用。机器可读汇总、输入内容 hash、432 probe manifest 和 HDRI 接触表分别为：

- `results/decouplegs/relighting/public-hdri-cohort-v1/cohort-summary.json`
- `results/decouplegs/relighting/public-hdri-cohort-v1/probe-manifest.json`
- `results/decouplegs/relighting/public-hdri-cohort-v1/hdri-contact-sheet.jpg`

上述 PAE/PSNR 是 primitive proxy 诊断，而论文的 `6.8°` 是 held-out re-insertion 的 masked image 指标；二者协议不同，禁止横向声称已经超过论文。旧的单车 controlled report `results/decouplegs/relighting/hdri-proxy-report.json` 保留作历史审计，但已由这套 20 车、独立三分割 protocol 取代。

## Held-out vehicle re-insertion 同公式公开替代协议

论文只说明把 held-out vehicle 重插回原场景，并在精确 SAM mask 内计算 Masked PSNR、PIE、PAE；没有发布 vehicle IDs、帧采样、SAM checkpoint/prompt、跨图 peak 聚合方式，以及 PAE 的零 RGB 向量处理。本仓库因此冻结 `configs/benchmark/decouplegs_reinsertion_public.yaml`：从六个有重建车辆的 nuScenes/HUGSIM 场景取 10 条真实 vehicle track，只在 HUGSIM 未用于训练的 temporal test split 上重插。每场景 216 个相机 view，共 1296 views；scene-0062 没有动态车辆，scene-0071 只有行人，明确排除。

主 mask 使用官方 SAM ViT-H、逐车辆投影 3D convex-hull box + positive point、最高 predicted-IoU candidate，**不膨胀**。论文的 metric 段只写 precise SAM masks；5-pixel dilation 出现在背景动态物体清理段，所以另报 sensitivity，不混入主结果。指标严格实现 Supplementary 方程：Masked PSNR 的分母是每像素 RGB 向量 squared L2（相对常规 channel-MSE PSNR 固定低 `10 log10(3)`），PIE 是 Rec.709 luminance 最大绝对误差，PAE 是 RGB cosine angle 的 mask 内最大值。

```bash
conda run -n decouplegs python tools/calibrate_decouplegs_reinsertion_relighting.py
conda run -n decouplegs python tools/run_decouplegs_reinsertion_protocol.py
conda run -n decouplegs python tools/audit_decouplegs_reinsertion_probe_domain.py
conda run -n decouplegs python tools/make_decouplegs_reinsertion_report.py
```

四个分支共享完全相同的 frame IDs、target 和 SAM masks；runner 对不配对输入直接失败。`raw-native` 是未压缩动态 3DGS 上界，`compact-no-relight` 是压缩 + Wigner-D SH rotation，`compact-ols` 显式加载 public-HDRI covariance-diffuse OLS sidecar，`compact-ols-shadow` 再加接触阴影。最后一支只估计 ground plane，不移动数据集 logged pose；为此 runtime 增加 `adjust_grounding_pose=false`，避免 photometric 误差被重复落地污染。

| 主协议（undilated SAM，210 张非空 mask） | Masked PSNR ↑ | PIE ↓ | PAE ↓ | 离线单相机 FPS |
|---|---:|---:|---:|---:|
| raw native | 19.660 dB | 0.3407 | 10.669° | 101.4 |
| compact no relight | 19.610 dB | 0.3420 | 10.694° | 214.6 |
| compact public-HDRI OLS | 18.148 dB | 0.3596 | 10.909° | 179.3 |
| compact OLS + pose-preserving shadow | 18.159 dB | 0.3594 | 10.902° | 104.7 |

压缩相对 raw 只变化 `-0.051 dB / +0.024°`，说明 asset compression/unified rasterization 在这套 held-out 图像上基本无损。但 public covariance-diffuse OLS 相对不重光照为 `-1.462 dB / +0.0176 PIE / +0.215° PAE`，没有复现论文的重光照增益。5px mask sensitivity 仍为 `-1.156 dB / +0.0165 / +0.160°`，结论不依赖 mask 边界。

这个负结果是协议证据，不是隐藏或调参目标。10/10 sidecar 在独立 public HDRI primitive proxy gate 上通过，但真实 scene probe 有 2.77% descriptor 元素超出 fit range、48.15% track-timestamp 至少一个维度越界；同时完全在域内的车辆也会退化。因此核心缺口是作者未公开的 vehicle materials、target renderer 与 relit per-primitive colors，而不是 OLS 线性求解器或单纯 descriptor OOD。完整报告、逐 track CSV、机器可读多聚合结果与车辆 crop 对照图位于：

- `results/decouplegs/reinsertion-paper-protocol/evaluation-v1/REPORT.md`
- `results/decouplegs/reinsertion-paper-protocol/evaluation-v1/protocol-summary.json`
- `results/decouplegs/reinsertion-paper-protocol/evaluation-v1/per-track-primary.csv`
- `results/decouplegs/reinsertion-paper-protocol/evaluation-v1/reinsertion-comparison-contact-sheet.jpg`
- `results/decouplegs/reinsertion-paper-protocol/evaluation-v1/reinsertion-shadow-contact-sheet.jpg`

论文的 `6.8° / 48.5°` 不能与上表直接声称同 cohort 比较。JSON 同时保留 per-image arithmetic mean/median、pooled masked PSNR、dataset peak 和 near-zero-vector PAE sensitivity，避免看到结果后选择隐藏聚合口径。

显式 calibration 始终优先于自适应 fallback。`configs/decouplegs.yaml` 默认打开 fallback，以便没有 `relighting.pt` 的公开车辆仍能获得实际重光照效果；设置 `adaptive_relighting: false` 可关闭。接触阴影的主光强先把 GraphDeco 存储的 DC coefficient 还原为 RGB（`rgb = C0 * sh_dc + 0.5`），再按论文定义取非负主分量；若直接对原始 coefficient 截断，典型阴天场景会错误得到零强度，从而完全没有阴影。

Eq. (7) 中的 `V_g` 是训练视图期望可见度。若保留了该统计，可将 `[full_background_gaussians]` tensor（或 `{"visibility": tensor}`）保存到 scene model 目录，并设置 `background_visibility_path`；未设置时 runtime 明确退化为 opacity 权重。

HUGSIM 的道路平面是 X/Z，物理向上方向为世界 `-Y`；发布的 3DRealCar canonical asset 也沿局部 `-Y` 向上延伸。runtime 因此分别显式设置 `world_up_sign=-1` 和 `asset_up_sign=-1`。车辆落地与接触阴影使用中心/资产元数据定义的物理包围盒，frustum culling 才使用包含 covariance 的保守 3σ 包围盒，避免大 splat 把车体错误抬离路面。

## 2 Hz closed loop

先修改 `configs/sim/nuscenes_base.yaml` 中 scene、vehicle、UniAD/VAD 路径，然后运行：

```bash
python closed_loop.py \
  --scenario_path configs/benchmark/nuscenes/scene-0013-easy-00.yaml \
  --base_path configs/sim/nuscenes_base.yaml \
  --camera_path configs/sim/nuscenes_camera.yaml \
  --kinematic_path configs/sim/decouplegs_kinematic.yaml \
  --decouple_config configs/decouplegs.yaml \
  --ad uniad --ad_cuda 1
```

`decouplegs_kinematic.yaml` 将 state/update 周期设为 0.5 s，即论文的 2 Hz；runtime config 将 episode 上限设为论文的 20 s（40 steps）。UniAD/VAD client 按 HUGSIM 上游建议使用独立环境，通过 FIFO 与 simulator 交换 observation/trajectory。`decouplegs/behavior.py` 的 IDM/MOBIL engine 已接入压力场景生成器和同步环境更新；车辆到达有限路线末端后会被退役，而不是 wrap/overlap 后瞬移回路线内部。每帧同时记录 lane-change 状态与有限差分速度，使 cut-in 的横向相对速度进入 TTC 计算。

## 论文值与工程值的边界

### 2026-08-22 官方 planner 管线审计

scene-0013 的 30 个连续 nuScenes 2 Hz keyframe 使用 UniAD/VAD 作者发布的 temporal info，并分别从官方 dataset/dataloader 和本仓库 FIFO adapter 独立推理。这个检查覆盖图像增强、历史 CAN bus、坐标轴、导航命令和时间顺序，而不是只比较输入张量形状：

| Planner | 官方 dataloader 对 logged GT mADE ↓ | 官方输出与 adapter 输出的逐点均差 ↓ | command match |
|---|---:|---:|---:|
| UniAD | 4.992301 m | 0.001260 m | 30/30 |
| VAD | 2.193601 m | 0.012280 m | 30/30 |

UniAD 的毫米级差异来自浮点/容器路径；VAD 的 `12.3 mm` 差异包含官方 test-time augmentation 容器与 FIFO 序列化路径。两者都远小于模型规划误差，说明 adapter 不是当前 mADE 差距的来源。对应机器可读证据是 `results/decouplegs/protocol-audit/scene0013-{uniad,vad}-official-vs-adapter.json`。

同一官方管线在 scene-0013 的真实图像上已经明显高于论文 Table 2 的 `0.76 m`，说明论文未公开的 15 个 clip/聚合口径对结果有决定性影响，不能用这个单场景值宣称匹配论文。

### 八场景 open-loop 聚合

八个 HUGSIM nuScenes 场景 `0010/0013/0038/0041/0051/0062/0064/0071` 各取 30 个 keyframe、每帧六相机，共 1440 张图。SAM ViT-H 车辆 mask 覆盖 260 张图；聚合器按真实 pair/frame 数加权并检查 manifest：

| 指标 | raw HUGSIM 3DGS | compact base | 变化 |
|---|---:|---:|---:|
| 单相机 rasterizer FPS | 73.560 | 126.229 | `1.716×` |
| peak allocated（跨场景最大） | 1998.7 MiB | 1982.8 MiB | `-15.9 MiB` |
| PSNR-All | 26.29918 dB | 26.29881 dB | `-0.00037 dB` |
| PSNR-Vehicle | 24.58754 dB | 24.54451 dB | `-0.04303 dB` |
| SSIM | 0.8031719 | 0.8031588 | `-0.0000131` |
| LPIPS-Alex ↓ | 0.3019160 | 0.3019427 | `+0.0000267` |
| PAE ↓ | 11.0428° | 11.0256° | `-0.0171°` |
| UniAD paired mADE | 1.02094 m | 1.02458 m | `+0.00364 m` |
| VAD paired mADE | 0.59258 m | 0.59302 m | `+0.00044 m` |

这里的 FPS 是离线单相机 CUDA rasterizer 吞吐，不能直接等同于论文未说明计数单位的 45 FPS。八场景结果证明压缩带来 `1.716×` 吞吐，同时画质和 planner 行为变化很小。真实图像对 logged GT 的聚合 mADE 仍为 UniAD `5.911 m`、VAD `3.353 m`，因此论文 clip 选择/指标聚合仍未恢复。完整结果位于 `results/decouplegs/open-loop-keyframes/strict-multiscene-summary.json`。

公开资产没有论文 HDRI/OLS calibration sidecar，因而 `compact-adaptive-full` 只作为 fallback 诊断，不计入 paper-exact 成绩。日志内已有车辆也不能重复 grounding；grounding 指标只在新插入或人为扰动的 canonical 资产上测量。

### 闭环协议与 oracle 审计

正式闭环运行前，控制器与环境分别做了路线 oracle 反证测试：

- 当前转向符号在八场景、`6 m/s`、40 steps 下达到 RC `0.852–1.000`，反转符号仅 `0.093–0.312`，排除了控制坐标符号错误；
- 完整环境 oracle 的 RC 为 `0.852–0.997`，最大横向误差不超过 `0.981 m`，背景碰撞点最多 1（阈值 100）；
- RC 统一为 metric polyline 上的 `distance_traveled / distance_planned`，`RC >= 0.99` 才终止，删除了 HUGSIM 旧 camera-index `/0.9` 的提前结束；
- 导航命令改为当前位置投影而非固定 `8 m` lookahead，避免开局转弯命令被跳过；
- 动态车到达有限路线末端后退役，位置不再瞬移；自车速度物理下限为 `0 m/s`。

证据位于 `closed-loop-controller-oracle*.json`、`closed-loop-environment-oracle.json`，每个正式 episode 还必须包含 `H-E2E-03/H-PHYSICS-01/02/04/H-BEHAVIOR-03/H-RC-01/H-PLAN-01/H-NAV-01/H-CONTROL-01` 契约，旧产物会被 suite runner 拒绝。

### 八场景 closed-loop 正式矩阵

strict 管线已经完成 `UniAD/VAD × Easy/Medium/Hard/Extreme × 50 episodes`，共 400/400 条。每条记录都重新通过 40-step/20 s、2 Hz 和九项当前 implementation contract 校验；200 个 scenario seed 在两个 planner 间严格配对，八个公开 nuScenes 场景各承担 25 个 scenario。

`SR*` 是无碰撞、无越界、无 planner failure 的安全代理；它没有冒充论文未公开判定规则的 SR。Strict SR 使用 `RC ≥ 0.99` 且无严重失败。

| 难度 | Planner | N | DS | SR* | Strict SR | RC | paper-literal minTTC | 六相机 observation/s | 六相机图像/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Easy | UniAD | 50 | — | 0.980±0.140 | 0.000±0.000 | 0.568±0.110 | 4.311±1.104 s | 15.50 | 93.03 |
| Easy | VAD | 50 | — | 0.000±0.000 | 0.000±0.000 | 0.542±0.159 | 3.955±1.067 s | 14.97 | 89.85 |
| Medium | UniAD | 50 | — | 1.000±0.000 | 0.000±0.000 | 0.566±0.106 | 3.402±1.293 s | 11.35 | 68.12 |
| Medium | VAD | 50 | — | 0.000±0.000 | 0.000±0.000 | 0.523±0.176 | 2.667±1.611 s | 10.64 | 63.86 |
| Hard | UniAD | 50 | — | 0.900±0.300 | 0.000±0.000 | 0.521±0.138 | 2.403±1.127 s | 9.28 | 55.67 |
| Hard | VAD | 50 | — | 0.020±0.140 | 0.000±0.000 | 0.453±0.225 | 1.407±1.094 s | 8.74 | 52.44 |
| Extreme | UniAD | 50 | — | 0.480±0.500 | 0.000±0.000 | 0.319±0.262 | 1.240±0.628 s | 7.43 | 44.57 |
| Extreme | VAD | 50 | — | 0.000±0.000 | 0.000±0.000 | 0.180±0.157 | 0.868±0.489 s | 7.00 | 42.01 |

完整置信区间、逐难度 UniAD−VAD 配对 bootstrap、termination audit、图和 CSV/JSON 位于 `results/decouplegs/closed-loop-8scene/closed-loop-suite/REPORT.md`。strict 的低 RC 和 VAD 大量碰撞/越界保留为真实 planner/controller 结果，没有用 oracle 或 trajectory stabilizer 润色；assisted 结果仍只作为本仓库的非论文诊断，不进入正式表。

论文给出 `DS = RC × ∏ p_i^{C_i}`，但没有公开 `p_i` 数值；因此当前 DS 固定为 `null`，不能反推或手调到论文的 `0.884`。同时输出 paper-literal center TTC、closing-center TTC 和 clearance TTC，最终表只把第一项与论文比较。论文的 FPS 也未说明是单相机图像还是一组六相机 observation，因此两种单位都保留。

### 公开边界

论文明确公开并已按原值实现：

- background 30k、vehicle 20k iterations，vehicle densification threshold `4e-4`；
- SAM dilation 5 pixels；
- pruning threshold `0.005` 和 importance 权重 `0.5/0.3/0.2`；
- covariance/color codebooks `512/1024`、EMA `0.99`、5000 updates、dead-code K-Means++；
- DTW heading weight `2.5`、map rasterization `0.1 m/pixel`、SE(2) Procrustes；
- 27-D local SH descriptor、OLS affine relighting、superellipse exponent `4`；
- 2 Hz planner、20 s episode，以及可独立调用的 IDM/MOBIL 同步状态更新公式。

当前无法从论文唯一恢复的内容：作者选取的精确 clips、scenario seeds 和 20 辆资产 ID，mADE 的精确 reference/aggregation，SR 的终止定义，DS penalty factors，FPS 的计数单位，训练视图 importance neighborhood/visibility traces，HDRI calibration 样本，MapTRv2 中间输出，以及作者 custom CUDA kernel 的调度/缓存策略。本实现对可工程化的项提供显式输入或多口径输出；统一渲染的功能、梯度与逐模块视觉效果已经实测，但不能与未发布 kernel 做逐像素或性能等价声明，也不会把未公开协议猜测包装成 paper-exact 结果。
