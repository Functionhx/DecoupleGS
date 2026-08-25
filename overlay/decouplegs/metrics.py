from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor


def _masked_pixels(image: Tensor, target: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    if image.shape != target.shape or image.shape[-1] != 3:
        raise ValueError("image and target must share an [..., 3] shape")
    mask = mask.to(dtype=torch.bool, device=image.device)
    if mask.shape != image.shape[:-1]:
        raise ValueError("mask must match the image's spatial dimensions")
    if not bool(mask.any()):
        raise ValueError("metric mask is empty")
    return image[mask], target.to(image)[mask]


def masked_psnr(image: Tensor, target: Tensor, mask: Tensor, maximum: float = 1.0) -> Tensor:
    """Paper-equation masked PSNR using per-pixel squared RGB L2 error.

    This differs from conventional channel-MSE PSNR by ``10*log10(3)`` for
    RGB data. It is kept explicit because the DecoupleGS supplement writes the
    metric with an RGB L2 norm while most libraries average over channels.
    """

    image_pixels, target_pixels = _masked_pixels(image, target, mask)
    mse = (image_pixels - target_pixels).square().sum(dim=-1).mean()
    return 10.0 * torch.log10(image.new_tensor(maximum**2) / mse.clamp_min(1e-12))


def channel_mse_psnr(image: Tensor, target: Tensor, maximum: float = 1.0) -> Tensor:
    """Conventional PSNR with MSE averaged across pixels and channels."""

    if image.shape != target.shape:
        raise ValueError("image and target must share a shape")
    mse = (image - target.to(image)).square().mean()
    return 10.0 * torch.log10(image.new_tensor(maximum**2) / mse.clamp_min(1e-12))


def masked_channel_mse_psnr(
    image: Tensor, target: Tensor, mask: Tensor, maximum: float = 1.0
) -> Tensor:
    """Conventional channel-MSE PSNR restricted to a spatial mask."""

    image_pixels, target_pixels = _masked_pixels(image, target, mask)
    mse = (image_pixels - target_pixels).square().mean()
    return 10.0 * torch.log10(image.new_tensor(maximum**2) / mse.clamp_min(1e-12))


def peak_intensity_error(image: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    image_pixels, target_pixels = _masked_pixels(image, target, mask)
    weights = image.new_tensor((0.2126, 0.7152, 0.0722))
    return ((image_pixels - target_pixels) @ weights).abs().max()


def peak_angular_error(image: Tensor, target: Tensor, mask: Tensor, *, degrees: bool = True) -> Tensor:
    image_pixels, target_pixels = _masked_pixels(image, target, mask)
    cosine = (image_pixels * target_pixels).sum(dim=-1) / (
        torch.linalg.vector_norm(image_pixels, dim=-1)
        * torch.linalg.vector_norm(target_pixels, dim=-1)
    ).clamp_min(1e-12)
    angle = torch.acos(cosine.clamp(-1.0, 1.0)).max()
    return torch.rad2deg(angle) if degrees else angle


def trajectory_ade(trajectory: Tensor, lane_points: Tensor) -> Tensor:
    if trajectory.ndim != 2 or lane_points.ndim != 2 or trajectory.shape[1] != 2 or lane_points.shape[1] != 2:
        raise ValueError("trajectory and lane_points must be [N, 2]")
    # torch.cdist may use the quadratic expansion ||x||^2 + ||y||^2 - 2x.y.
    # In float32 map coordinates around 1e3 m this loses enough precision to
    # quantize sub-metre errors (observed as repeated sqrt(1/8) values on
    # nuScenes). Distances are translation invariant, so centring both sets
    # makes the evaluation numerically stable without changing its definition.
    origin = trajectory.mean(dim=0, keepdim=True)
    distances = torch.cdist(trajectory - origin, lane_points.to(trajectory) - origin)
    return distances.min(dim=-1).values.mean()


def ground_penetration_rate(anchor_heights: Tensor, ground_heights: Tensor) -> Tensor:
    if anchor_heights.shape != ground_heights.shape:
        raise ValueError("anchor and ground heights must align")
    return (anchor_heights < ground_heights).to(torch.float32).mean()


def route_completion(distance_traveled: float, distance_planned: float) -> float:
    if distance_planned <= 0:
        raise ValueError("distance_planned must be positive")
    return min(max(distance_traveled / distance_planned, 0.0), 1.0)


def driving_score(completion: float, penalties: Mapping[str, tuple[float, int]]) -> float:
    score = completion
    for penalty, count in penalties.values():
        if not 0 < penalty <= 1 or count < 0:
            raise ValueError("penalties must be (factor in (0,1], non-negative count)")
        score *= penalty**count
    return score


def minimum_ttc(
    ego_positions: Tensor,
    agent_positions: Tensor,
    ego_velocities: Tensor,
    agent_velocities: Tensor,
    *,
    epsilon: float = 1e-6,
) -> Tensor:
    """Minimum closing-direction TTC, with diverging pairs treated as infinity."""

    if ego_positions.shape != ego_velocities.shape:
        raise ValueError("ego positions and velocities must align")
    if agent_positions.shape != agent_velocities.shape:
        raise ValueError("agent positions and velocities must align")
    if agent_positions.ndim == ego_positions.ndim:
        agent_positions = agent_positions.unsqueeze(-2)
        agent_velocities = agent_velocities.unsqueeze(-2)
    relative_position = agent_positions - ego_positions.unsqueeze(-2)
    relative_velocity = agent_velocities - ego_velocities.unsqueeze(-2)
    distance = torch.linalg.vector_norm(relative_position, dim=-1)
    closing_speed = -(relative_position * relative_velocity).sum(dim=-1) / distance.clamp_min(epsilon)
    ttc = torch.where(closing_speed > epsilon, distance / closing_speed, torch.full_like(distance, torch.inf))
    return ttc.amin()
