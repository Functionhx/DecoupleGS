#!/usr/bin/env python3
"""Exercise the installed CUDA extensions and the unified HUGSIM render path."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simple_knn._C import distCUDA2
from unidepth.ops.knn import knn_points

from decouplegs.hugsim import HUGSIMRuntimeBridge
from decouplegs.kernels import merge_sorted_index_streams, sorted_index_merge_backend
from decouplegs.runtime import DecoupleRuntime
from gaussian_renderer import render


class FakeHUGSIMModel:
    def __init__(self, means: torch.Tensor, *, semantic_classes: int = 20) -> None:
        count = means.shape[0]
        self._means = means
        self._scales = torch.full((count, 3), 0.12, device=means.device)
        self._quats = torch.zeros((count, 4), device=means.device)
        self._quats[:, 0] = 1.0
        self._opacities = torch.full((count, 1), 0.9, device=means.device)
        self._sh = torch.zeros((count, 16, 3), device=means.device)
        self._sh[:, 0, 0] = 1.5
        index = torch.arange(count, device=means.device) % semantic_classes
        self._semantics = torch.nn.functional.one_hot(index, semantic_classes).float()
        self.ground_model = None
        self.affine = False
        self.active_sh_degree = 3

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._means

    @property
    def get_scaling(self) -> torch.Tensor:
        return self._scales

    @property
    def get_rotation(self) -> torch.Tensor:
        return self._quats

    @property
    def get_opacity(self) -> torch.Tensor:
        return self._opacities

    @property
    def get_features(self) -> torch.Tensor:
        return self._sh

    @property
    def get_3D_features(self) -> torch.Tensor:
        return self._semantics


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required")
    device = torch.device("cuda")
    points = torch.rand((128, 3), device=device)
    distances = distCUDA2(points)
    if distances.shape != (128,) or not bool(torch.isfinite(distances).all()):
        raise RuntimeError("simple-knn returned invalid distances")
    knn_query = torch.randn((1, 8, 3), device=device, requires_grad=True)
    knn_reference = torch.randn((1, 11, 3), device=device, requires_grad=True)
    knn = knn_points(knn_query, knn_reference, K=2)
    knn.dists.sum().backward()
    if (
        knn.dists.shape != (1, 8, 2)
        or knn_query.grad is None
        or knn_reference.grad is None
        or not bool(torch.isfinite(knn_query.grad).all())
        or not bool(torch.isfinite(knn_reference.grad).all())
    ):
        raise RuntimeError("UniDepth KNN returned invalid outputs or gradients")

    merge_backends = []
    for dynamic_count in (73, 700):
        static_ids = torch.arange(1024, dtype=torch.int64, device=device) * 3
        dynamic_ids = torch.arange(dynamic_count, dtype=torch.int64, device=device) * 5
        static_values = torch.arange(1024, dtype=torch.int32, device=device)
        dynamic_values = torch.arange(dynamic_count, dtype=torch.int32, device=device)
        merged = merge_sorted_index_streams(
            static_ids,
            static_values,
            dynamic_ids,
            dynamic_values,
            dynamic_index_offset=1024,
        )
        keys = torch.cat((static_ids, dynamic_ids))
        values = torch.cat((static_values, dynamic_values + 1024))
        expected = values[torch.argsort(keys, stable=True)]
        if not torch.equal(merged, expected):
            raise RuntimeError("stable CUDA intersection merge does not match full sort")
        merge_backends.append(
            sorted_index_merge_backend(1024, dynamic_count, device)
        )

    background_means = torch.tensor(
        [[-0.3, 0.0, 3.0], [0.3, 0.0, 3.0]],
        device=device,
        requires_grad=True,
    )
    asset_means = torch.tensor([[0.0, 0.0, 0.0]], device=device, requires_grad=True)
    background = FakeHUGSIMModel(background_means)
    asset = FakeHUGSIMModel(asset_means)
    pose = torch.eye(4, device=device)
    pose[2, 3] = 2.5
    camera = SimpleNamespace(
        c2w=torch.eye(4, device=device),
        K=torch.tensor(
            [[80.0, 0.0, 32.0], [0.0, 80.0, 32.0], [0.0, 0.0, 1.0]],
            device=device,
        ),
        width=64,
        height=64,
        timestamp=0.0,
        dynamics={"car": pose},
    )
    runtime = DecoupleRuntime(
        opacity_grounding=False,
        relighting=False,
        contact_shadows=False,
        frustum_culling=False,
    )
    bridge = HUGSIMRuntimeBridge(runtime)
    # Raw assets take the compose fallback rather than the compact VQ path;
    # register it before the renderer probes the bridge.
    bridge.register_model("car", asset)
    result = render(
        camera,
        None,
        background,
        {"car": asset},
        {},
        torch.zeros(3, device=device),
        decouple_bridge=bridge,
    )
    loss = result["render"].sum()
    loss.backward()
    if result["render"].shape != (3, 64, 64):
        raise RuntimeError(f"unexpected render shape: {tuple(result['render'].shape)}")
    if float(result["alphas"].max()) <= 0:
        raise RuntimeError("unified rasterization produced an empty frame")
    for name, gradient in (("background", background_means.grad), ("asset", asset_means.grad)):
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise RuntimeError(f"{name} gradients are invalid")
    print(
        "DecoupleGS CUDA smoke test passed:",
        "simple-knn=OK,",
        "unidepth-knn=forward+backward OK,",
        f"intersection-merge={'+'.join(merge_backends)} OK,",
        "unified-rasterizer=forward+backward OK,",
        f"alpha_max={float(result['alphas'].max()):.6f}",
    )


if __name__ == "__main__":
    main()
