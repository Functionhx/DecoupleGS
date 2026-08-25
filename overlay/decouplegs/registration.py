from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .spatial import BackgroundSpatialIndex
from .types import GaussianSet


@dataclass(frozen=True)
class RegistrationConfig:
    heading_weight: float = 2.5
    map_resolution: float | None = 0.1
    dtw_window: int | None = None
    dtw_window_fraction: float = 0.25
    column_radius: float = 0.35
    opacity_threshold: float = 0.0
    max_vertical_distance: float | None = None
    vertical_axis: int = 2
    vertical_sign: int = 1
    horizontal_axes: tuple[int, int] = (0, 1)
    forward_axis: int = 0
    up_axis: int = 2
    up_sign: int = 1

    def __post_init__(self) -> None:
        axes = set(self.horizontal_axes + (self.vertical_axis,))
        if axes != {0, 1, 2}:
            raise ValueError("horizontal_axes and vertical_axis must partition the three dimensions")
        if self.forward_axis not in (0, 1, 2) or self.up_axis not in (0, 1, 2):
            raise ValueError("forward_axis and up_axis must be valid 3D axes")
        if self.forward_axis == self.up_axis:
            raise ValueError("forward_axis and up_axis must differ")
        if self.vertical_sign not in (-1, 1) or self.up_sign not in (-1, 1):
            raise ValueError("vertical_sign and up_sign must each be -1 or 1")
        if self.heading_weight < 0 or self.dtw_window_fraction < 0 or self.column_radius <= 0:
            raise ValueError("registration weights and column radius must be non-negative/positive")
        if self.map_resolution is not None and self.map_resolution <= 0:
            raise ValueError("map_resolution must be positive when provided")
        if not 0.0 <= self.opacity_threshold <= 1.0:
            raise ValueError("opacity_threshold must be in [0, 1]")
        if self.max_vertical_distance is not None and self.max_vertical_distance <= 0:
            raise ValueError("max_vertical_distance must be positive when provided")


@dataclass(frozen=True)
class MapRegistrationResult:
    transform: Tensor
    lane_index: int
    trajectory_indices: Tensor
    lane_indices: Tensor
    normalized_cost: float

    @property
    def rotation(self) -> Tensor:
        return self.transform[:2, :2]

    @property
    def translation(self) -> Tensor:
        return self.transform[:2, 2]


@dataclass(frozen=True)
class GroundPlane:
    normal: Tensor
    offset: Tensor
    vertical_axis: int = 2
    horizontal_axes: tuple[int, int] = (0, 1)

    def height(self, horizontal: Tensor) -> Tensor:
        h0, h1 = self.horizontal_axes
        numerator = -(
            self.normal[h0] * horizontal[..., 0]
            + self.normal[h1] * horizontal[..., 1]
            + self.offset
        )
        denominator = self.normal[self.vertical_axis]
        epsilon = torch.finfo(denominator.dtype).eps
        denominator = torch.where(
            denominator.abs() < epsilon,
            torch.copysign(denominator.new_tensor(epsilon), denominator),
            denominator,
        )
        return numerator / denominator

    def signed_distance(self, points: Tensor) -> Tensor:
        return points @ self.normal + self.offset


@dataclass(frozen=True)
class GroundingResult:
    transform: Tensor
    plane: GroundPlane
    anchor_heights: Tensor
    valid_anchors: Tensor


@dataclass(frozen=True)
class LaneProjectionResult:
    """Monotone continuous projection of a trajectory onto a lane polyline."""

    points: Tensor
    arc_lengths: Tensor
    segment_indices: Tensor


def wrap_angle(angle: Tensor) -> Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def infer_headings(points: Tensor) -> Tensor:
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise ValueError("at least two 2D points are required to infer headings")
    delta = torch.empty_like(points)
    delta[0] = points[1] - points[0]
    delta[-1] = points[-1] - points[-2]
    if points.shape[0] > 2:
        delta[1:-1] = points[2:] - points[:-2]
    heading = torch.atan2(delta[:, 1], delta[:, 0])
    if bool((torch.linalg.vector_norm(delta, dim=-1) < 1e-8).any()):
        # Fill stationary samples with their nearest non-stationary direction.
        valid = torch.linalg.vector_norm(delta, dim=-1) >= 1e-8
        if not bool(valid.any()):
            return torch.zeros(points.shape[0], dtype=points.dtype, device=points.device)
        valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        all_indices = torch.arange(points.shape[0], device=points.device)
        nearest = (all_indices[:, None] - valid_indices[None]).abs().argmin(dim=-1)
        heading = heading[valid_indices[nearest]]
    return heading


def resample_polyline(points: Tensor, spacing: float = 0.1) -> Tensor:
    """Rasterize a vector lane at the Supplementary's 0.1 m resolution."""

    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise ValueError("a lane must contain at least two 2D points")
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    lengths = torch.linalg.vector_norm(points[1:] - points[:-1], dim=-1)
    keep = torch.cat((torch.ones(1, dtype=torch.bool, device=points.device), lengths > 1e-8))
    points = points[keep]
    if points.shape[0] < 2:
        raise ValueError("a lane must contain two distinct points")
    lengths = torch.linalg.vector_norm(points[1:] - points[:-1], dim=-1)
    cumulative = torch.cat((lengths.new_zeros(1), lengths.cumsum(dim=0)))
    total = float(cumulative[-1].item())
    steps = torch.arange(
        math.floor(total / spacing) + 1,
        dtype=points.dtype,
        device=points.device,
    ) * spacing
    if total - float(steps[-1].item()) > max(spacing * 1e-6, 1e-8):
        steps = torch.cat((steps, steps.new_tensor([total])))
    else:
        steps[-1] = total
    segment = torch.searchsorted(cumulative[1:].contiguous(), steps, right=False)
    segment = segment.clamp_max(points.shape[0] - 2)
    fraction = (steps - cumulative[segment]) / lengths[segment]
    return points[segment] + fraction[:, None] * (points[segment + 1] - points[segment])


def trajectory_lane_cost(
    trajectory: Tensor,
    lane: Tensor,
    *,
    trajectory_headings: Tensor | None = None,
    lane_headings: Tensor | None = None,
    heading_weight: float = 2.5,
) -> Tensor:
    if trajectory_headings is None:
        trajectory_headings = infer_headings(trajectory)
    if lane_headings is None:
        lane_headings = infer_headings(lane)
    distance = torch.cdist(trajectory, lane)
    heading = wrap_angle(trajectory_headings[:, None] - lane_headings[None]).abs()
    return distance + heading_weight * heading


def constrained_dtw(cost: Tensor, *, window: int | None = None) -> tuple[Tensor, Tensor, float]:
    """Constrained DTW with deterministic diagonal/up/left tie breaking."""

    if cost.ndim != 2 or min(cost.shape) == 0:
        raise ValueError("cost must be a non-empty [T, U] matrix")
    n, m = cost.shape
    if window is None:
        window = max(n, m)
    # Different sequence lengths require at least this much slack.
    window = max(window, abs(n - m))
    values = cost.detach().to(dtype=torch.float64, device="cpu").numpy()
    cumulative = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    predecessor = np.full((n, m), -1, dtype=np.int8)
    cumulative[0, 0] = 0.0
    for i in range(1, n + 1):
        expected_j = i * m / n
        start = max(1, math.floor(expected_j - window))
        end = min(m, math.ceil(expected_j + window))
        for j in range(start, end + 1):
            candidates = (cumulative[i - 1, j - 1], cumulative[i - 1, j], cumulative[i, j - 1])
            direction = int(np.argmin(candidates))
            cumulative[i, j] = values[i - 1, j - 1] + candidates[direction]
            predecessor[i - 1, j - 1] = direction
    if not np.isfinite(cumulative[n, m]):
        raise RuntimeError("DTW window excludes every valid path")
    i, j = n - 1, m - 1
    path_i, path_j = [i], [j]
    while i > 0 or j > 0:
        direction = predecessor[i, j]
        if direction == 0:
            i, j = i - 1, j - 1
        elif direction == 1:
            i -= 1
        elif direction == 2:
            j -= 1
        else:
            # This is only reachable at a constrained boundary.
            if i > 0 and j > 0:
                i, j = i - 1, j - 1
            elif i > 0:
                i -= 1
            else:
                j -= 1
        path_i.append(i)
        path_j.append(j)
    path_i.reverse()
    path_j.reverse()
    return (
        torch.tensor(path_i, dtype=torch.int64, device=cost.device),
        torch.tensor(path_j, dtype=torch.int64, device=cost.device),
        float(cumulative[n, m]),
    )


def orthogonal_procrustes_se2(source: Tensor, target: Tensor, weights: Tensor | None = None) -> Tensor:
    """Globally optimal proper SE(2) transform for matched 2D points."""

    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("source and target must have matching [N, 2] shapes")
    if source.shape[0] < 2:
        raise ValueError("at least two correspondences are required")
    if weights is None:
        weights = torch.ones(source.shape[0], dtype=source.dtype, device=source.device)
    weights = weights.to(source).reshape(-1)
    weights = weights / weights.sum().clamp_min(1e-12)
    source_center = (weights[:, None] * source).sum(dim=0)
    target_center = (weights[:, None] * target).sum(dim=0)
    source_zero = source - source_center
    target_zero = target - target_center
    cross = (weights[:, None] * source_zero).transpose(0, 1) @ target_zero
    u, _, vh = torch.linalg.svd(cross)
    rotation = vh.transpose(-1, -2) @ u.transpose(-1, -2)
    if float(torch.linalg.det(rotation).item()) < 0:
        vh = vh.clone()
        vh[-1] *= -1
        rotation = vh.transpose(-1, -2) @ u.transpose(-1, -2)
    translation = target_center - rotation @ source_center
    transform = torch.eye(3, dtype=source.dtype, device=source.device)
    transform[:2, :2] = rotation
    transform[:2, 2] = translation
    return transform


def apply_se2(points: Tensor, transform: Tensor) -> Tensor:
    return points @ transform[:2, :2].transpose(0, 1) + transform[:2, 2]


def project_trajectory_to_polyline(
    trajectory: Tensor,
    polyline: Tensor,
    *,
    enforce_monotonic: bool = True,
) -> LaneProjectionResult:
    """Project ordered 2D samples onto continuous lane segments.

    The paper states that semantic topology precedes DTW/Procrustes but does
    not specify how residual non-rigid lane drift is handled after the global
    SE(2) fit.  This function implements the opt-in R&D hypothesis H-GEO-01:
    use the selected topological lane route as a continuous Frenet reference
    and monotonically remove the residual lateral component.  It is kept
    separate from :func:`register_trajectory_to_lanes` so the literal
    paper-described SE(2)-only path remains independently measurable.

    The polyline is automatically oriented to match the trajectory endpoints.
    Returned arc lengths are measured along that oriented polyline.
    """

    if trajectory.ndim != 2 or trajectory.shape[1] != 2 or trajectory.shape[0] == 0:
        raise ValueError("trajectory must have shape [N, 2] and be non-empty")
    if polyline.ndim != 2 or polyline.shape[1] != 2 or polyline.shape[0] < 2:
        raise ValueError("polyline must have shape [M, 2] with M >= 2")
    polyline = polyline.to(trajectory)
    same_direction_cost = (
        torch.linalg.vector_norm(trajectory[0] - polyline[0])
        + torch.linalg.vector_norm(trajectory[-1] - polyline[-1])
    )
    reverse_direction_cost = (
        torch.linalg.vector_norm(trajectory[0] - polyline[-1])
        + torch.linalg.vector_norm(trajectory[-1] - polyline[0])
    )
    if bool(reverse_direction_cost < same_direction_cost):
        polyline = polyline.flip(0)

    starts = polyline[:-1]
    vectors = polyline[1:] - starts
    squared_lengths = vectors.square().sum(dim=-1).clamp_min(1e-12)
    relative = trajectory[:, None, :] - starts[None, :, :]
    fractions = (relative * vectors[None]).sum(dim=-1) / squared_lengths[None]
    fractions = fractions.clamp(0.0, 1.0)
    candidates = starts[None] + fractions[..., None] * vectors[None]
    squared_distance = (trajectory[:, None] - candidates).square().sum(dim=-1)
    segment_indices = squared_distance.argmin(dim=-1)
    selected_fractions = fractions.gather(1, segment_indices[:, None])[:, 0]

    cumulative = torch.cat(
        (
            trajectory.new_zeros(1),
            torch.sqrt(squared_lengths).cumsum(dim=0),
        )
    )
    arc_lengths = cumulative[segment_indices] + (
        selected_fractions * torch.sqrt(squared_lengths[segment_indices])
    )
    if enforce_monotonic:
        arc_lengths = torch.cummax(arc_lengths, dim=0).values

    # Re-evaluate the continuous polyline at the possibly monotone-adjusted
    # arc lengths instead of reusing the independently selected projections.
    segment_indices = torch.searchsorted(cumulative, arc_lengths, right=True) - 1
    segment_indices = segment_indices.clamp(0, vectors.shape[0] - 1)
    segment_start_lengths = cumulative[segment_indices]
    segment_fractions = (
        (arc_lengths - segment_start_lengths)
        / torch.sqrt(squared_lengths[segment_indices]).clamp_min(1e-12)
    ).clamp(0.0, 1.0)
    points = starts[segment_indices] + segment_fractions[:, None] * vectors[segment_indices]
    return LaneProjectionResult(points, arc_lengths, segment_indices)


def register_trajectory_to_lanes(
    trajectory: Tensor,
    lanes: Sequence[Tensor],
    config: RegistrationConfig | None = None,
    *,
    trajectory_headings: Tensor | None = None,
) -> MapRegistrationResult:
    """DTW lane selection followed by the paper's SE(2) Procrustes correction."""

    config = RegistrationConfig() if config is None else config
    if not lanes:
        raise ValueError("at least one lane centerline is required")
    if trajectory_headings is None:
        trajectory_headings = infer_headings(trajectory)
    best: tuple[float, int, Tensor, Tensor, Tensor] | None = None
    for lane_index, lane in enumerate(lanes):
        lane = lane.to(trajectory)
        if lane.shape[0] < 2:
            continue
        if config.map_resolution is not None:
            try:
                lane = resample_polyline(lane, config.map_resolution)
            except ValueError:
                continue
        cost = trajectory_lane_cost(
            trajectory,
            lane,
            trajectory_headings=trajectory_headings,
            heading_weight=config.heading_weight,
        )
        window = config.dtw_window
        if window is None:
            window = max(abs(trajectory.shape[0] - lane.shape[0]), int(max(cost.shape) * config.dtw_window_fraction))
        path_t, path_l, total = constrained_dtw(cost, window=window)
        normalized = total / len(path_t)
        if best is None or normalized < best[0]:
            best = normalized, lane_index, path_t, path_l, lane
    if best is None:
        raise ValueError("no lane contains enough points for registration")
    normalized, lane_index, path_t, path_l, lane = best
    transform = orthogonal_procrustes_se2(trajectory[path_t], lane[path_l])
    return MapRegistrationResult(transform, lane_index, path_t, path_l, normalized)


def opacity_accumulated_heights(
    horizontal: Tensor,
    background_means: Tensor,
    background_opacities: Tensor,
    config: RegistrationConfig | None = None,
    *,
    reference_height: Tensor | float | None = None,
    chunk_size: int = 262144,
    spatial_index: BackgroundSpatialIndex | None = None,
) -> tuple[Tensor, Tensor]:
    """Opacity-weighted vertical-column accumulation from Eq. (6)."""

    config = RegistrationConfig() if config is None else config
    h0, h1 = config.horizontal_axes
    vertical = config.vertical_axis
    opacity = background_opacities.reshape(-1).to(background_means)
    heights = torch.zeros(horizontal.shape[0], dtype=background_means.dtype, device=background_means.device)
    weights = torch.zeros_like(heights)
    horizontal = horizontal.to(background_means)
    if reference_height is not None:
        reference_height = torch.as_tensor(reference_height, dtype=background_means.dtype, device=background_means.device)
    if spatial_index is not None:
        if len(spatial_index) != background_means.shape[0]:
            raise ValueError("spatial index and background means have different lengths")
        query = spatial_index.query_radius(
            horizontal,
            config.column_radius,
            axes=config.horizontal_axes,
        )
        if query.indices.numel():
            points = background_means[query.indices]
            alpha = opacity[query.indices]
            valid = alpha >= config.opacity_threshold
            if reference_height is not None and config.max_vertical_distance is not None:
                reference = reference_height.reshape(-1)
                if reference.numel() == 1:
                    reference = reference.expand(horizontal.shape[0])
                valid &= (
                    points[:, vertical] - reference[query.owners]
                ).abs() <= config.max_vertical_distance
            contribution = alpha * valid
            heights.index_add_(
                0,
                query.owners,
                contribution * points[:, vertical],
            )
            weights.index_add_(0, query.owners, contribution)
        valid = weights > 1e-8
        heights = torch.where(
            valid,
            heights / weights.clamp_min(1e-8),
            torch.full_like(heights, torch.nan),
        )
        return heights, valid
    for start in range(0, background_means.shape[0], chunk_size):
        points = background_means[start : start + chunk_size]
        alpha = opacity[start : start + chunk_size]
        valid_alpha = alpha >= config.opacity_threshold
        delta = horizontal[:, None] - points[None, :, (h0, h1)]
        squared = delta.square().sum(dim=-1)
        valid = valid_alpha[None] & (squared <= config.column_radius**2)
        if reference_height is not None and config.max_vertical_distance is not None:
            reference = reference_height.reshape(-1)
            if reference.numel() == 1:
                reference = reference.expand(horizontal.shape[0])
            valid &= (points[None, :, vertical] - reference[:, None]).abs() <= config.max_vertical_distance
        # Supplementary Eq. (6): every splat in the vertical column contributes
        # only its opacity alpha_k; the column membership defines N_j.
        weight = alpha[None] * valid
        heights += (weight * points[None, :, vertical]).sum(dim=-1)
        weights += weight.sum(dim=-1)
    valid = weights > 1e-8
    heights = torch.where(valid, heights / weights.clamp_min(1e-8), torch.full_like(heights, torch.nan))
    return heights, valid


def fit_ground_plane(
    horizontal: Tensor,
    heights: Tensor,
    config: RegistrationConfig | None = None,
) -> GroundPlane:
    config = RegistrationConfig() if config is None else config
    valid = torch.isfinite(heights)
    if not bool(valid.any()):
        raise ValueError("no valid ground anchors")
    horizontal_valid, height_valid = horizontal[valid], heights[valid]
    if horizontal_valid.shape[0] >= 3:
        design = torch.cat((horizontal_valid, torch.ones_like(horizontal_valid[:, :1])), dim=-1)
        coefficients = torch.linalg.lstsq(design, height_valid[:, None]).solution[:, 0]
    else:
        coefficients = torch.stack(
            (
                height_valid.new_zeros(()),
                height_valid.new_zeros(()),
                height_valid.median(),
            )
        )
    normal = torch.zeros(3, dtype=horizontal.dtype, device=horizontal.device)
    normal[config.horizontal_axes[0]] = -coefficients[0]
    normal[config.horizontal_axes[1]] = -coefficients[1]
    normal[config.vertical_axis] = 1.0
    normal = torch.nn.functional.normalize(normal, dim=0)
    # GroundPlane.normal always points toward physical world-up, which is -Y
    # in HUGSIM and +Z in the default paper-style coordinate system.
    normal = normal * config.vertical_sign
    point = torch.zeros(3, dtype=horizontal.dtype, device=horizontal.device)
    point[config.horizontal_axes[0]] = horizontal_valid[:, 0].mean()
    point[config.horizontal_axes[1]] = horizontal_valid[:, 1].mean()
    point[config.vertical_axis] = coefficients[0] * point[config.horizontal_axes[0]] + coefficients[1] * point[config.horizontal_axes[1]] + coefficients[2]
    return GroundPlane(normal=normal, offset=-(normal @ point), vertical_axis=config.vertical_axis, horizontal_axes=config.horizontal_axes)


def _yaw_forward(yaw: Tensor, config: RegistrationConfig) -> Tensor:
    forward = torch.zeros(3, dtype=yaw.dtype, device=yaw.device)
    forward[config.horizontal_axes[0]] = torch.cos(yaw)
    forward[config.horizontal_axes[1]] = torch.sin(yaw)
    return forward


def rotation_from_yaw_and_normal(yaw: Tensor, normal: Tensor, config: RegistrationConfig | None = None) -> Tensor:
    """Preserve planar heading while aligning the canonical up axis to terrain."""

    config = RegistrationConfig() if config is None else config
    normal = torch.nn.functional.normalize(normal, dim=0)
    if normal[config.vertical_axis] * config.vertical_sign < 0:
        normal = -normal
    forward = _yaw_forward(yaw, config)
    forward = torch.nn.functional.normalize(forward - (forward @ normal) * normal, dim=0)
    # Map canonical physical up (up_sign * e_up) onto the world ground normal,
    # then complete a right-handed basis for arbitrary axis assignments.
    remaining_axis = ({0, 1, 2} - {config.forward_axis, config.up_axis}).pop()
    columns: list[Tensor | None] = [None, None, None]
    columns[config.forward_axis] = forward
    columns[config.up_axis] = normal * config.up_sign
    columns[remaining_axis] = torch.nn.functional.normalize(
        torch.cross(columns[(remaining_axis + 1) % 3], columns[(remaining_axis + 2) % 3], dim=0),
        dim=0,
    )
    rotation = torch.stack(columns, dim=-1)  # type: ignore[arg-type]
    return rotation


def bottom_anchors(gaussians: GaussianSet, config: RegistrationConfig | None = None) -> Tensor:
    """Return canonical asset anchors on its physical bottom face.

    ``vertical_axis`` and ``horizontal_axes`` describe the *world* frame.
    Canonical assets need not use the same convention (HUGSIM scene assets
    are Y-forward/Z-up, while 3DRealCar assets are X-forward/-Y-up), so the
    local face must instead be selected through ``up_axis``/``up_sign``.
    """

    config = RegistrationConfig() if config is None else config
    minimum, maximum = gaussians.physical_bounds
    h0, h1 = tuple(axis for axis in range(3) if axis != config.up_axis)
    vertical = config.up_axis
    anchors = torch.zeros((4, 3), dtype=gaussians.dtype, device=gaussians.device)
    anchors[:, h0] = torch.stack((minimum[h0], minimum[h0], maximum[h0], maximum[h0]))
    anchors[:, h1] = torch.stack((minimum[h1], maximum[h1], minimum[h1], maximum[h1]))
    anchors[:, vertical] = minimum[vertical] if config.up_sign > 0 else maximum[vertical]
    return anchors


def ground_asset_pose(
    asset: GaussianSet,
    center: Tensor,
    yaw: Tensor | float,
    background_means: Tensor,
    background_opacities: Tensor,
    config: RegistrationConfig | None = None,
    *,
    reference_height: Tensor | float | None = None,
    spatial_index: BackgroundSpatialIndex | None = None,
) -> GroundingResult:
    """Derive 6-DoF pose from four bottom anchors and opacity grounding."""

    config = RegistrationConfig() if config is None else config
    center = center.to(asset.means).reshape(2)
    yaw = torch.as_tensor(yaw, dtype=asset.dtype, device=asset.device)
    flat_normal = torch.zeros(3, dtype=asset.dtype, device=asset.device)
    flat_normal[config.vertical_axis] = config.vertical_sign
    yaw_rotation = rotation_from_yaw_and_normal(yaw, flat_normal, config)
    anchors = bottom_anchors(asset, config)
    local_bottom_center = anchors.mean(dim=0)
    preliminary = anchors @ yaw_rotation.transpose(0, 1)
    translation_xy = center - preliminary[:, config.horizontal_axes].mean(dim=0)
    query = preliminary[:, config.horizontal_axes] + translation_xy
    heights, valid = opacity_accumulated_heights(
        query,
        background_means.to(asset.means),
        background_opacities.to(asset.means),
        config,
        reference_height=reference_height,
        spatial_index=spatial_index,
    )
    plane = fit_ground_plane(query, heights, config)
    rotation = rotation_from_yaw_and_normal(yaw, plane.normal, config)
    rotated_center = rotation @ local_bottom_center
    translation = torch.zeros(3, dtype=asset.dtype, device=asset.device)
    translation[config.horizontal_axes[0]] = center[0] - rotated_center[config.horizontal_axes[0]]
    translation[config.horizontal_axes[1]] = center[1] - rotated_center[config.horizontal_axes[1]]
    ground_height = plane.height(center)
    translation[config.vertical_axis] = ground_height - rotated_center[config.vertical_axis]
    transform = torch.eye(4, dtype=asset.dtype, device=asset.device)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return GroundingResult(transform=transform, plane=plane, anchor_heights=heights, valid_anchors=valid)


def ground_asset_from_pose(
    asset: GaussianSet,
    initial_transform: Tensor,
    background_means: Tensor,
    background_opacities: Tensor,
    config: RegistrationConfig | None = None,
    *,
    spatial_index: BackgroundSpatialIndex | None = None,
) -> GroundingResult:
    """Replace an approximate pose's vertical state with opacity grounding."""

    config = RegistrationConfig() if config is None else config
    initial_transform = initial_transform.to(asset.means)
    anchors = bottom_anchors(asset, config)
    bottom_center_world = initial_transform[:3, :3] @ anchors.mean(dim=0) + initial_transform[:3, 3]
    center = bottom_center_world[list(config.horizontal_axes)]
    forward = initial_transform[:3, config.forward_axis]
    yaw = torch.atan2(forward[config.horizontal_axes[1]], forward[config.horizontal_axes[0]])
    return ground_asset_pose(
        asset,
        center,
        yaw,
        background_means,
        background_opacities,
        config,
        reference_height=bottom_center_world[config.vertical_axis],
        spatial_index=spatial_index,
    )
