from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .transforms import (
    covariance_from_scale_quaternion,
    covariance_to_scale_quaternion,
    pack_symmetric,
    unpack_symmetric,
)
from .types import GaussianSet

SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class CompressionConfig:
    """Paper/Supplementary defaults for semantic-aware asset compression."""

    prune_threshold: float = 0.005
    weight_visibility: float = 0.5
    weight_color: float = 0.3
    weight_entropy: float = 0.2
    color_codebook_size: int = 1024
    shape_codebook_size: int = 512
    ema_momentum: float = 0.99
    ema_iterations: int = 5000
    ema_batch_size: int = 16384
    kmeans_iterations: int = 25
    assignment_chunk_size: int = 32768
    dead_code_interval: int = 250
    dead_code_threshold: float = 1e-3
    entropy_bins: int = 16
    voxel_size: float | None = None
    visibility_gate_saliency: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.prune_threshold <= 1:
            raise ValueError("prune_threshold must be in [0, 1]")
        weights = (self.weight_visibility, self.weight_color, self.weight_entropy)
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("importance weights must be non-negative and non-zero")
        if not 0 <= self.ema_momentum < 1:
            raise ValueError("ema_momentum must be in [0, 1)")
        if self.color_codebook_size <= 0 or self.shape_codebook_size <= 0:
            raise ValueError("codebook sizes must be positive")


@dataclass(frozen=True)
class CompressionReport:
    input_primitives: int
    retained_primitives: int
    input_bytes: int
    compact_bytes: int
    shape_rmse: float
    color_rmse: float
    score_min: float
    score_max: float
    score_mean: float

    @property
    def retained_fraction(self) -> float:
        return self.retained_primitives / max(self.input_primitives, 1)

    @property
    def compression_ratio(self) -> float:
        return self.input_bytes / max(self.compact_bytes, 1)


def _default_voxel_size(gaussians: GaussianSet) -> float:
    if len(gaussians) == 0:
        return 1.0
    scale_based = float(gaussians.scales.detach().median().item()) * 8.0
    minimum, maximum = gaussians.bounds
    extent_based = float(torch.linalg.vector_norm(maximum - minimum).item()) / 64.0
    return max(scale_based, extent_based, 1e-4)


def estimate_local_importance_terms(
    gaussians: GaussianSet,
    *,
    voxel_size: float | None = None,
    entropy_bins: int = 16,
) -> tuple[Tensor, Tensor]:
    """Estimate Eq. (4)'s local color contrast and texture entropy in O(N).

    The paper does not publish the exact neighborhood implementation.  We use
    spatial voxels as an explicit, deterministic neighborhood and expose the
    resulting terms so recorded training-view statistics can replace them.
    """

    if entropy_bins < 2:
        raise ValueError("entropy_bins must be at least two")
    if len(gaussians) == 0:
        empty = torch.empty(0, dtype=gaussians.dtype, device=gaussians.device)
        return empty, empty
    size = _default_voxel_size(gaussians) if voxel_size is None else voxel_size
    if size <= 0:
        raise ValueError("voxel_size must be positive")
    coordinates = torch.floor((gaussians.means - gaussians.means.amin(dim=0)) / size).to(torch.int64)
    _, inverse = torch.unique(coordinates, dim=0, return_inverse=True)
    groups = int(inverse.max().item()) + 1

    rgb = (gaussians.sh[:, 0] * SH_C0 + 0.5).clamp(0.0, 1.0)
    sums = torch.zeros((groups, 3), dtype=rgb.dtype, device=rgb.device)
    sums.index_add_(0, inverse, rgb)
    counts = torch.bincount(inverse, minlength=groups).to(rgb.dtype).clamp_min_(1.0)
    means = sums / counts[:, None]
    color_contrast = torch.linalg.vector_norm(rgb - means[inverse], dim=-1) / math.sqrt(3.0)

    luminance = rgb @ rgb.new_tensor((0.2126, 0.7152, 0.0722))
    bins = torch.clamp((luminance * entropy_bins).to(torch.int64), max=entropy_bins - 1)
    histogram_index = inverse * entropy_bins + bins
    histogram = torch.bincount(histogram_index, minlength=groups * entropy_bins).reshape(groups, entropy_bins)
    probability = histogram.to(rgb.dtype) / counts[:, None]
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1) / math.log(entropy_bins)
    return color_contrast.clamp(0.0, 1.0), entropy[inverse].clamp(0.0, 1.0)


def importance_score(
    gaussians: GaussianSet,
    config: CompressionConfig,
    *,
    visibility: Tensor | None = None,
    color_contrast: Tensor | None = None,
    texture_entropy: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the explicit importance score from Eq. (4)."""

    if visibility is None:
        visibility = gaussians.visibility
    if visibility is None:
        # A documented fallback for assets lacking per-view alpha traces.
        visibility = gaussians.opacities
    if visibility.ndim > 1:
        visibility = visibility.reshape(len(gaussians), -1).mean(dim=-1)
    visibility = visibility.to(gaussians.means).clamp(0.0, 1.0)

    if color_contrast is None or texture_entropy is None:
        estimated_color, estimated_entropy = estimate_local_importance_terms(
            gaussians,
            voxel_size=config.voxel_size,
            entropy_bins=config.entropy_bins,
        )
        color_contrast = estimated_color if color_contrast is None else color_contrast
        texture_entropy = estimated_entropy if texture_entropy is None else texture_entropy
    color_contrast = color_contrast.to(gaussians.means).reshape(-1).clamp(0.0, 1.0)
    texture_entropy = texture_entropy.to(gaussians.means).reshape(-1).clamp(0.0, 1.0)
    for name, value in (("visibility", visibility), ("color_contrast", color_contrast), ("texture_entropy", texture_entropy)):
        if value.shape != (len(gaussians),):
            raise ValueError(f"{name} must contain one scalar per Gaussian")

    # R&D hypothesis H-COMP-01. The paper does not disclose the numerical
    # scale of D_i/H_i. If they are raw [0,1] neighbourhood statistics, the
    # published 0.005 threshold prunes almost nothing because entropy alone is
    # commonly >0.1. Visibility-gating makes them expected *rendered*
    # photometric/saliency contributions while preserving Eq. (4)'s additive
    # form. It is opt-in until ground-truth Table-1 validation selects it.
    score_color = color_contrast
    score_entropy = texture_entropy
    if config.visibility_gate_saliency:
        score_color = score_color * visibility
        score_entropy = score_entropy * visibility
    score = (
        config.weight_visibility * visibility
        + config.weight_color * score_color
        + config.weight_entropy * score_entropy
    )
    return score, {
        "visibility": visibility,
        "color_contrast": color_contrast,
        "texture_entropy": texture_entropy,
        "score_color": score_color,
        "score_entropy": score_entropy,
    }


def prune_by_importance(
    gaussians: GaussianSet,
    config: CompressionConfig,
    **terms: Tensor,
) -> tuple[GaussianSet, Tensor, dict[str, Tensor]]:
    score, components = importance_score(gaussians, config, **terms)
    keep = score >= config.prune_threshold
    if len(gaussians) and not bool(keep.any()):
        keep[score.argmax()] = True
    retained = gaussians.select(keep)
    retained.metadata["importance_weights"] = {
        "visibility": config.weight_visibility,
        "color": config.weight_color,
        "entropy": config.weight_entropy,
    }
    retained.metadata["prune_threshold"] = config.prune_threshold
    return retained, score, components


def nearest_code(values: Tensor, codebook: Tensor, chunk_size: int = 32768) -> tuple[Tensor, Tensor]:
    """Memory-bounded squared-L2 nearest-code assignment."""

    if values.ndim != 2 or codebook.ndim != 2 or values.shape[1] != codebook.shape[1]:
        raise ValueError("values and codebook must be rank-two with matching feature dimensions")
    indices, distances = [], []
    code_norm = codebook.square().sum(dim=-1)
    for start in range(0, values.shape[0], chunk_size):
        chunk = values[start : start + chunk_size]
        squared = chunk.square().sum(dim=-1, keepdim=True) + code_norm[None] - 2.0 * (chunk @ codebook.transpose(0, 1))
        distance, index = squared.clamp_min_(0.0).min(dim=-1)
        indices.append(index)
        distances.append(distance)
    if not indices:
        return (
            torch.empty(0, dtype=torch.int64, device=values.device),
            torch.empty(0, dtype=values.dtype, device=values.device),
        )
    return torch.cat(indices), torch.cat(distances)


def kmeans_plus_plus(
    values: Tensor,
    clusters: int,
    *,
    seed: int = 0,
    chunk_size: int = 32768,
) -> Tensor:
    """K-Means++ initialization required by the Supplementary."""

    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("non-empty [N, D] values are required")
    clusters = min(clusters, values.shape[0])
    generator = torch.Generator(device=values.device)
    generator.manual_seed(seed)
    first = torch.randint(values.shape[0], (1,), device=values.device, generator=generator)
    centers = [values[first].squeeze(0)]
    closest = torch.full((values.shape[0],), torch.inf, dtype=values.dtype, device=values.device)
    for _ in range(1, clusters):
        center = centers[-1][None]
        for start in range(0, values.shape[0], chunk_size):
            chunk = values[start : start + chunk_size]
            distance = (chunk - center).square().sum(dim=-1)
            closest[start : start + chunk.shape[0]] = torch.minimum(
                closest[start : start + chunk.shape[0]], distance
            )
        total = closest.sum()
        if not bool(torch.isfinite(total)) or float(total.item()) <= 1e-20:
            index = torch.randint(values.shape[0], (1,), device=values.device, generator=generator)
        else:
            index = torch.multinomial(closest / total, 1, generator=generator)
        centers.append(values[index].squeeze(0))
    return torch.stack(centers)


def run_kmeans(
    values: Tensor,
    clusters: int,
    *,
    iterations: int = 25,
    seed: int = 0,
    chunk_size: int = 32768,
) -> Tensor:
    codebook = kmeans_plus_plus(values, clusters, seed=seed, chunk_size=chunk_size)
    for iteration in range(iterations):
        indices, distances = nearest_code(values, codebook, chunk_size)
        sums = torch.zeros_like(codebook)
        sums.index_add_(0, indices, values)
        counts = torch.bincount(indices, minlength=codebook.shape[0]).to(values.dtype)
        occupied = counts > 0
        codebook = torch.where(occupied[:, None], sums / counts.clamp_min(1.0)[:, None], codebook)
        if bool((~occupied).any()):
            farthest = distances.topk(min(int((~occupied).sum().item()), values.shape[0])).indices
            codebook[~occupied] = values[farthest]
        if iteration > 0 and float(distances.mean().item()) <= 1e-12:
            break
    return codebook


class EMAVectorQuantizer:
    """Eqs. (10)-(13), including periodic dead-code reinitialization."""

    def __init__(
        self,
        clusters: int,
        *,
        momentum: float = 0.99,
        kmeans_iterations: int = 25,
        assignment_chunk_size: int = 32768,
        dead_code_interval: int = 250,
        dead_code_threshold: float = 1e-3,
        seed: int = 0,
    ) -> None:
        self.clusters = clusters
        self.momentum = momentum
        self.kmeans_iterations = kmeans_iterations
        self.assignment_chunk_size = assignment_chunk_size
        self.dead_code_interval = dead_code_interval
        self.dead_code_threshold = dead_code_threshold
        self.seed = seed
        self.codebook: Tensor | None = None
        self.ema_count: Tensor | None = None
        self.ema_sum: Tensor | None = None

    def initialize(self, values: Tensor) -> None:
        self.codebook = run_kmeans(
            values,
            self.clusters,
            iterations=self.kmeans_iterations,
            seed=self.seed,
            chunk_size=self.assignment_chunk_size,
        )
        indices, _ = nearest_code(values, self.codebook, self.assignment_chunk_size)
        self.ema_count = torch.bincount(indices, minlength=self.codebook.shape[0]).to(values.dtype)
        self.ema_sum = torch.zeros_like(self.codebook)
        self.ema_sum.index_add_(0, indices, values)

    def _reinitialize_dead(self, values: Tensor, generator: torch.Generator) -> None:
        assert self.codebook is not None and self.ema_count is not None and self.ema_sum is not None
        dead = self.ema_count < self.dead_code_threshold
        count = int(dead.sum().item())
        if count == 0:
            return
        subset_size = min(values.shape[0], max(count * 16, count))
        subset_index = torch.randperm(values.shape[0], device=values.device, generator=generator)[:subset_size]
        replacements = kmeans_plus_plus(
            values[subset_index],
            count,
            seed=self.seed + count,
            chunk_size=self.assignment_chunk_size,
        )
        self.codebook[dead] = replacements
        self.ema_count[dead] = 1.0
        self.ema_sum[dead] = replacements

    def fit(self, values: Tensor, *, iterations: int = 5000, batch_size: int = 16384) -> tuple[Tensor, Tensor]:
        if self.codebook is None:
            self.initialize(values)
        assert self.codebook is not None and self.ema_count is not None and self.ema_sum is not None
        generator = torch.Generator(device=values.device)
        generator.manual_seed(self.seed + 1)
        for iteration in range(iterations):
            if batch_size >= values.shape[0]:
                batch = values
            else:
                index = torch.randint(values.shape[0], (batch_size,), device=values.device, generator=generator)
                batch = values[index]
            assignments, _ = nearest_code(batch, self.codebook, self.assignment_chunk_size)
            counts = torch.bincount(assignments, minlength=self.codebook.shape[0]).to(values.dtype)
            sums = torch.zeros_like(self.codebook)
            sums.index_add_(0, assignments, batch)
            self.ema_count.mul_(self.momentum).add_(counts, alpha=1.0 - self.momentum)
            self.ema_sum.mul_(self.momentum).add_(sums, alpha=1.0 - self.momentum)
            self.codebook.copy_(self.ema_sum / self.ema_count.clamp_min(1e-8)[:, None])
            if self.dead_code_interval > 0 and (iteration + 1) % self.dead_code_interval == 0:
                self._reinitialize_dead(values, generator)
        indices, distances = nearest_code(values, self.codebook, self.assignment_chunk_size)
        return indices, distances


def _compact_index(index: Tensor, codebook_size: int) -> Tensor:
    if codebook_size <= 256:
        return index.to(torch.uint8)
    if codebook_size <= 32768:
        return index.to(torch.int16)
    return index.to(torch.int32)


@dataclass
class CompactGaussianAsset:
    """Versioned VQ asset whose covariance and SH tensors are index-coded."""

    means: Tensor
    opacities: Tensor
    shape_codebook: Tensor
    shape_indices: Tensor
    color_codebook: Tensor
    color_indices: Tensor
    sh_degree: int
    semantics: Tensor | None = None
    visibility: Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    format_version: int = 2

    def __post_init__(self) -> None:
        n = self.means.shape[0]
        if self.means.shape != (n, 3) or self.opacities.reshape(-1).shape != (n,):
            raise ValueError("invalid compact primitive tensors")
        if self.shape_codebook.ndim != 2 or self.shape_codebook.shape[1] != 6:
            raise ValueError("shape codebook must contain packed 3x3 covariances")
        expected_color = (self.sh_degree + 1) ** 2 * 3
        if self.color_codebook.ndim != 2 or self.color_codebook.shape[1] != expected_color:
            raise ValueError("color codebook does not match sh_degree")
        if self.shape_indices.shape != (n,) or self.color_indices.shape != (n,):
            raise ValueError("compact indices must contain one entry per primitive")
        if self.semantics is not None and self.semantics.shape[0] not in (1, n):
            raise ValueError("semantics must contain one constant row or one row per primitive")

    def __len__(self) -> int:
        return self.means.shape[0]

    @property
    def device(self) -> torch.device:
        return self.means.device

    @property
    def dtype(self) -> torch.dtype:
        return self.means.dtype

    @property
    def memory_bytes(self) -> int:
        values = (
            self.means,
            self.opacities,
            self.shape_codebook,
            self.shape_indices,
            self.color_codebook,
            self.color_indices,
            self.semantics,
            self.visibility,
        )
        return sum(0 if value is None else value.numel() * value.element_size() for value in values)

    @property
    def physical_bounds(self) -> tuple[Tensor, Tensor]:
        value = self.metadata.get("physical_bounds")
        if value is None:
            return self.means.amin(dim=0), self.means.amax(dim=0)
        bounds = torch.as_tensor(value, dtype=self.means.dtype, device=self.means.device)
        if bounds.shape != (2, 3) or bool((bounds[0] > bounds[1]).any()):
            raise ValueError("metadata['physical_bounds'] must contain ordered [min, max] bounds")
        return bounds[0], bounds[1]

    @property
    def bounds(self) -> tuple[Tensor, Tensor]:
        covariance = unpack_symmetric(self.shape_codebook)
        radius = 3.0 * torch.linalg.eigvalsh(covariance).amax().clamp_min(0.0).sqrt()
        return self.means.amin(dim=0) - radius, self.means.amax(dim=0) + radius

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
    ) -> CompactGaussianAsset:
        target = torch.device(device)

        def move_float(value: Tensor | None) -> Tensor | None:
            if value is None:
                return None
            return value.to(device=target, dtype=value.dtype if dtype is None else dtype)

        return CompactGaussianAsset(
            means=move_float(self.means),
            opacities=move_float(self.opacities),
            shape_codebook=move_float(self.shape_codebook),
            shape_indices=self.shape_indices.to(target),
            color_codebook=move_float(self.color_codebook),
            color_indices=self.color_indices.to(target),
            sh_degree=self.sh_degree,
            semantics=move_float(self.semantics),
            visibility=move_float(self.visibility),
            metadata=dict(self.metadata),
            format_version=self.format_version,
        )

    def decode(self, *, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32) -> GaussianSet:
        target = self.means.device if device is None else torch.device(device)
        shape_index = self.shape_indices.to(device=target, dtype=torch.int64)
        color_index = self.color_indices.to(device=target, dtype=torch.int64)
        covariance = unpack_symmetric(self.shape_codebook.to(device=target, dtype=dtype)[shape_index])
        scales, quats = covariance_to_scale_quaternion(covariance)
        coeffs = (self.sh_degree + 1) ** 2
        sh = self.color_codebook.to(device=target, dtype=dtype)[color_index].reshape(-1, coeffs, 3)
        semantics = None
        if self.semantics is not None:
            semantics = self.semantics.to(device=target, dtype=dtype)
            if semantics.shape[0] == 1:
                semantics = semantics.expand(len(self), -1)
        return GaussianSet(
            means=self.means.to(device=target, dtype=dtype),
            scales=scales,
            quats=quats,
            opacities=self.opacities.to(device=target, dtype=dtype).reshape(-1),
            sh=sh,
            semantics=semantics,
            visibility=None if self.visibility is None else self.visibility.to(device=target, dtype=dtype),
            metadata=dict(self.metadata),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": "OpenDecoupleGS-CompactAsset",
            "format_version": self.format_version,
            "means": self.means,
            "opacities": self.opacities,
            "shape_codebook": self.shape_codebook,
            "shape_indices": self.shape_indices,
            "color_codebook": self.color_codebook,
            "color_indices": self.color_indices,
            "sh_degree": self.sh_degree,
            "semantics": self.semantics,
            "visibility": self.visibility,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> CompactGaussianAsset:
        state = torch.load(path, map_location=map_location, weights_only=True)
        if state.get("format") != "OpenDecoupleGS-CompactAsset":
            raise ValueError(f"{path} is not an OpenDecoupleGS compact asset")
        state = dict(state)
        state.pop("format")
        return cls(**state)


def compress_asset(
    gaussians: GaussianSet,
    config: CompressionConfig | None = None,
    *,
    visibility: Tensor | None = None,
    color_contrast: Tensor | None = None,
    texture_entropy: Tensor | None = None,
) -> tuple[CompactGaussianAsset, CompressionReport]:
    """Execute Supplementary Algorithm 1 end to end."""

    config = CompressionConfig() if config is None else config
    retained, score, _ = prune_by_importance(
        gaussians,
        config,
        visibility=visibility,
        color_contrast=color_contrast,
        texture_entropy=texture_entropy,
    )
    if len(retained) == 0:
        raise ValueError("cannot quantize an empty asset")
    physical_minimum, physical_maximum = gaussians.physical_bounds
    shape_values = pack_symmetric(covariance_from_scale_quaternion(retained.scales, retained.quats))
    color_values = retained.sh.reshape(len(retained), -1)
    shape_quantizer = EMAVectorQuantizer(
        config.shape_codebook_size,
        momentum=config.ema_momentum,
        kmeans_iterations=config.kmeans_iterations,
        assignment_chunk_size=config.assignment_chunk_size,
        dead_code_interval=config.dead_code_interval,
        dead_code_threshold=config.dead_code_threshold,
        seed=config.seed,
    )
    color_quantizer = EMAVectorQuantizer(
        config.color_codebook_size,
        momentum=config.ema_momentum,
        kmeans_iterations=config.kmeans_iterations,
        assignment_chunk_size=config.assignment_chunk_size,
        dead_code_interval=config.dead_code_interval,
        dead_code_threshold=config.dead_code_threshold,
        seed=config.seed + 1009,
    )
    shape_index, shape_distance = shape_quantizer.fit(
        shape_values,
        iterations=config.ema_iterations,
        batch_size=config.ema_batch_size,
    )
    color_index, color_distance = color_quantizer.fit(
        color_values,
        iterations=config.ema_iterations,
        batch_size=config.ema_batch_size,
    )
    assert shape_quantizer.codebook is not None and color_quantizer.codebook is not None
    compact_semantics = None if retained.semantics is None else retained.semantics.detach()
    semantic_encoding = "none"
    if compact_semantics is not None:
        semantic_encoding = "dense"
        # HUGSIM vehicle assets carry the same class distribution on every
        # primitive. Keeping N identical 20-D rows costs more than all VQ
        # indices combined, so encode this common case losslessly as one row.
        if torch.equal(compact_semantics, compact_semantics[:1].expand_as(compact_semantics)):
            compact_semantics = compact_semantics[:1].clone()
            semantic_encoding = "constant"
    compact = CompactGaussianAsset(
        means=retained.means.detach(),
        opacities=retained.opacities.detach(),
        shape_codebook=shape_quantizer.codebook.detach(),
        shape_indices=_compact_index(shape_index, shape_quantizer.codebook.shape[0]),
        color_codebook=color_quantizer.codebook.detach(),
        color_indices=_compact_index(color_index, color_quantizer.codebook.shape[0]),
        sh_degree=retained.sh_degree,
        semantics=compact_semantics,
        visibility=None if retained.visibility is None else retained.visibility.detach(),
        metadata={
            **retained.metadata,
            # Pruning boundary primitives must not change grounding geometry.
            "physical_bounds": torch.stack((physical_minimum, physical_maximum))
            .detach()
            .cpu()
            .tolist(),
            "compression_config": asdict(config),
            "source_primitives": len(gaussians),
            "semantic_encoding": semantic_encoding,
        },
    )
    report = CompressionReport(
        input_primitives=len(gaussians),
        retained_primitives=len(retained),
        input_bytes=gaussians.memory_bytes,
        compact_bytes=compact.memory_bytes,
        shape_rmse=float(shape_distance.mean().sqrt().item()),
        color_rmse=float(color_distance.mean().sqrt().item()),
        score_min=float(score.min().item()) if score.numel() else 0.0,
        score_max=float(score.max().item()) if score.numel() else 0.0,
        score_mean=float(score.mean().item()) if score.numel() else 0.0,
    )
    return compact, report
