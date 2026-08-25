from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor

from .types import GaussianSet


@dataclass(frozen=True)
class OrbitVisibilityConfig:
    """Deterministic proxy views for assets whose training cameras are unavailable.

    The released HUGSIM 3DRealCar checkpoints do not include their training
    cameras.  These orbit views make the paper's expected opacity-contribution
    term measurable without pretending that opacity alone is visibility.
    """

    azimuth_views: int = 24
    elevations_degrees: tuple[float, ...] = (0.0, 10.0, 20.0)
    image_width: int = 256
    image_height: int = 256
    horizontal_fov_degrees: float = 55.0
    distance_margin: float = 1.2
    vertical_axis: int = 1
    vertical_sign: int = -1
    horizontal_axes: tuple[int, int] = (0, 2)
    batch_size: int = 1
    packed: bool = False

    def __post_init__(self) -> None:
        if self.azimuth_views <= 0 or not self.elevations_degrees:
            raise ValueError("at least one azimuth and elevation view is required")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        if not 1.0 < self.horizontal_fov_degrees < 179.0:
            raise ValueError("horizontal_fov_degrees must be in (1, 179)")
        if self.distance_margin <= 1.0:
            raise ValueError("distance_margin must be greater than one")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.vertical_sign not in (-1, 1):
            raise ValueError("vertical_sign must be -1 or 1")
        if set(self.horizontal_axes + (self.vertical_axis,)) != {0, 1, 2}:
            raise ValueError("horizontal_axes and vertical_axis must partition 3D axes")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _camera_to_world(position: Tensor, target: Tensor, world_up: Tensor) -> Tensor:
    """Build a right-handed c2w for the +Z-forward, +Y-down HUGSIM camera."""

    forward = torch.nn.functional.normalize(target - position, dim=0)
    right = torch.linalg.cross(forward, world_up)
    if float(torch.linalg.vector_norm(right).item()) < 1e-6:
        raise ValueError("camera direction is parallel to world up")
    right = torch.nn.functional.normalize(right, dim=0)
    down = torch.nn.functional.normalize(torch.linalg.cross(forward, right), dim=0)
    matrix = torch.eye(4, dtype=position.dtype, device=position.device)
    matrix[:3, :3] = torch.stack((right, down, forward), dim=-1)
    matrix[:3, 3] = position
    return matrix


def orbit_view_matrices(
    bounds: tuple[Tensor, Tensor],
    config: OrbitVisibilityConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Return world-to-camera matrices and pinhole intrinsics for proxy views."""

    config = OrbitVisibilityConfig() if config is None else config
    minimum, maximum = bounds
    if minimum.shape != (3,) or maximum.shape != (3,):
        raise ValueError("bounds must contain two three-dimensional vectors")
    if bool((maximum <= minimum).all()):
        raise ValueError("asset bounds must have non-zero extent")
    center = 0.5 * (minimum + maximum)
    bounding_radius = 0.5 * torch.linalg.vector_norm(maximum - minimum)
    half_fov = math.radians(config.horizontal_fov_degrees) * 0.5
    distance = float(bounding_radius.item()) / math.sin(half_fov) * config.distance_margin

    world_up = torch.zeros(3, dtype=center.dtype, device=center.device)
    world_up[config.vertical_axis] = config.vertical_sign
    axis0 = torch.zeros_like(world_up)
    axis1 = torch.zeros_like(world_up)
    axis0[config.horizontal_axes[0]] = 1.0
    axis1[config.horizontal_axes[1]] = 1.0
    matrices = []
    for elevation_degrees in config.elevations_degrees:
        elevation = math.radians(elevation_degrees)
        for index in range(config.azimuth_views):
            azimuth = 2.0 * math.pi * index / config.azimuth_views
            radial = math.cos(azimuth) * axis0 + math.sin(azimuth) * axis1
            position = center + distance * (
                math.cos(elevation) * radial + math.sin(elevation) * world_up
            )
            matrices.append(torch.linalg.inv(_camera_to_world(position, center, world_up)))

    focal = config.image_width / (2.0 * math.tan(half_fov))
    intrinsics = center.new_tensor(
        (
            (focal, 0.0, config.image_width * 0.5),
            (0.0, focal, config.image_height * 0.5),
            (0.0, 0.0, 1.0),
        )
    )
    return torch.stack(matrices), intrinsics[None].expand(len(matrices), -1, -1).clone()


def estimate_opacity_contribution_visibility(
    gaussians: GaussianSet,
    config: OrbitVisibilityConfig | None = None,
) -> Tensor:
    """Estimate each primitive's expected front-to-back alpha contribution.

    For alpha compositing, the derivative of the summed rendered color with
    respect to a per-primitive unit color is exactly that primitive's integrated
    transmittance-weighted opacity.  Autodiff therefore exposes the quantity the
    paper describes without an O(views * pixels * primitives) materialization.
    The result is averaged over views but deliberately not over pixels; a score
    of 0.005 then means a primitive contributes at least half a percent of one
    pixel per view on average.
    """

    if gaussians.device.type != "cuda":
        raise ValueError("opacity-contribution estimation requires CUDA")
    if len(gaussians) == 0:
        return torch.empty(0, dtype=gaussians.dtype, device=gaussians.device)
    config = OrbitVisibilityConfig() if config is None else config
    from gsplat.rendering import rasterization

    viewmats, intrinsics = orbit_view_matrices(gaussians.physical_bounds, config)
    colors = torch.ones(
        (len(gaussians), 1),
        dtype=gaussians.dtype,
        device=gaussians.device,
        requires_grad=True,
    )
    contribution = torch.zeros(len(gaussians), dtype=gaussians.dtype, device=gaussians.device)
    total_views = viewmats.shape[0]
    with torch.enable_grad():
        for start in range(0, total_views, config.batch_size):
            end = min(start + config.batch_size, total_views)
            rendered, _, _ = rasterization(
                means=gaussians.means.detach(),
                quats=gaussians.quats.detach(),
                scales=gaussians.scales.detach(),
                opacities=gaussians.opacities.detach(),
                colors=colors,
                viewmats=viewmats[start:end],
                Ks=intrinsics[start:end],
                width=config.image_width,
                height=config.image_height,
                render_mode="RGB",
                sh_degree=None,
                near_plane=0.01,
                far_plane=max(100.0, float(torch.linalg.vector_norm(gaussians.physical_bounds[1] - gaussians.physical_bounds[0]).item()) * 20.0),
                packed=config.packed,
            )
            gradient = torch.autograd.grad(rendered.sum(), colors, retain_graph=False)[0]
            contribution.add_(gradient[:, 0] / total_views)
    return contribution.clamp_min_(0.0).detach()
