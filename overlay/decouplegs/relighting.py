from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor

from .registration import GroundPlane, RegistrationConfig
from .spatial import BackgroundSpatialIndex
from .types import GaussianSet

SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class RelightingConfig:
    descriptor_bands: int = 3
    probe_sigma: float = 3.0
    probe_radius: float | None = 9.0
    shadow_strength: float = 0.55
    shadow_exponent: float = 4.0
    shadow_decay: float = 2.0
    shadow_ground_band: float = 0.25
    shadow_mask_epsilon: float | None = None
    adaptive_strength: float = 0.65
    adaptive_reference_intensity: float = 0.5
    adaptive_min_gain: float = 0.25
    adaptive_max_gain: float = 2.0

    def __post_init__(self) -> None:
        if self.descriptor_bands != 3:
            raise ValueError("the paper fixes the local descriptor to the first three SH bands")
        if self.probe_sigma <= 0 or self.shadow_exponent <= 0 or self.shadow_decay <= 0:
            raise ValueError("relighting spatial parameters must be positive")
        if self.probe_radius is not None and self.probe_radius <= 0:
            raise ValueError("probe_radius must be positive when provided")
        if self.shadow_ground_band < 0:
            raise ValueError("shadow_ground_band must be non-negative")
        if self.shadow_mask_epsilon is not None and not 0 < self.shadow_mask_epsilon < 1:
            raise ValueError("shadow_mask_epsilon must be in (0, 1) when provided")
        if not 0 <= self.shadow_strength <= 1:
            raise ValueError("shadow_strength must be in [0, 1]")
        if not 0 <= self.adaptive_strength <= 1:
            raise ValueError("adaptive_strength must be in [0, 1]")
        if self.adaptive_reference_intensity <= 0:
            raise ValueError("adaptive_reference_intensity must be positive")
        if not 0 < self.adaptive_min_gain <= self.adaptive_max_gain:
            raise ValueError("adaptive gain limits must be positive and ordered")


@dataclass
class RelightingNormalEquations:
    """Cached sufficient statistics for Eq. (16) ridge sweeps.

    The normal matrix and right-hand side are independent of the ridge value.
    Keeping them lets an HDRI protocol select regularization on a validation
    split without repeatedly traversing every probe/primitive pair.
    """

    normal: Tensor
    rhs: Tensor
    output_dtype: torch.dtype
    samples: int
    primitives: int

    @classmethod
    def from_samples(
        cls,
        descriptors: Tensor,
        canonical: Tensor,
        targets: Tensor,
    ) -> RelightingNormalEquations:
        if descriptors.ndim != 2 or descriptors.shape[1] != 27:
            raise ValueError("descriptors must have shape [N, 27]")
        if targets.ndim != 3 or targets.shape[0] != descriptors.shape[0]:
            raise ValueError("targets must have shape [N, M, A]")
        if min(targets.shape) == 0:
            raise ValueError(
                "OLS calibration requires non-empty probe, primitive, and attribute dimensions"
            )
        if canonical.ndim == 2 and canonical.shape != targets.shape[1:]:
            raise ValueError("shared canonical values must have shape [M, A]")
        if canonical.ndim == 3 and canonical.shape != targets.shape:
            raise ValueError("canonical and target samples must align")
        if canonical.ndim not in (2, 3):
            raise ValueError("canonical must have shape [M, A] or [N, M, A]")
        output_dtype = (
            targets.dtype
            if targets.dtype in (torch.float32, torch.float64)
            else torch.float32
        )
        descriptors = descriptors.to(dtype=torch.float64)
        canonical = canonical.to(device=descriptors.device, dtype=torch.float64)
        targets = targets.to(device=descriptors.device, dtype=torch.float64)
        if canonical.ndim == 2:
            canonical = canonical[None].expand(targets.shape[0], -1, -1)
        probes, primitives, attributes = targets.shape
        augmented = torch.cat(
            (
                descriptors,
                torch.ones(
                    (probes, 1),
                    dtype=descriptors.dtype,
                    device=descriptors.device,
                ),
            ),
            dim=-1,
        )
        outer = torch.einsum("ni,nj->nij", augmented, augmented)
        canonical_sum = canonical.sum(dim=1)
        canonical_square_sum = canonical.square().sum(dim=1)
        canonical_target_sum = (canonical * targets).sum(dim=1)
        target_sum = targets.sum(dim=1)

        # A row of Eq. (16)'s design matrix is [c_can * [L,1], [L,1]].
        # Accumulating X^T X and X^T y avoids materializing [N*M, 56].
        normal = torch.empty(
            (attributes, 56, 56),
            dtype=torch.float64,
            device=descriptors.device,
        )
        normal[:, :28, :28] = torch.einsum(
            "na,nij->aij", canonical_square_sum, outer
        )
        cross = torch.einsum("na,nij->aij", canonical_sum, outer)
        normal[:, :28, 28:] = cross
        normal[:, 28:, :28] = cross.transpose(-1, -2)
        normal[:, 28:, 28:] = primitives * outer.sum(dim=0)[None]
        rhs = torch.cat(
            (
                torch.einsum("na,ni->ai", canonical_target_sum, augmented),
                torch.einsum("na,ni->ai", target_sum, augmented),
            ),
            dim=-1,
        )
        return cls(
            normal=normal,
            rhs=rhs,
            output_dtype=output_dtype,
            samples=probes,
            primitives=primitives,
        )

    def solve(
        self,
        *,
        ridge: float = 0.0,
        ridge_prior: str = "zero",
    ) -> RelightingCalibration:
        if ridge < 0:
            raise ValueError("ridge must be non-negative")
        if ridge_prior not in {"zero", "identity"}:
            raise ValueError("ridge_prior must be 'zero' or 'identity'")
        attributes = self.normal.shape[0]
        if ridge:
            identity = torch.eye(
                56, dtype=self.normal.dtype, device=self.normal.device
            )
            regularized_rhs = self.rhs
            if ridge_prior == "identity":
                prior = torch.zeros(
                    (attributes, 56),
                    dtype=self.normal.dtype,
                    device=self.normal.device,
                )
                prior[:, 27] = 1.0
                regularized_rhs = self.rhs + ridge * prior
            theta = torch.linalg.solve(
                self.normal + ridge * identity[None],
                regularized_rhs[..., None],
            )[..., 0]
        else:
            theta = (torch.linalg.pinv(self.normal) @ self.rhs[..., None])[..., 0]
        return RelightingCalibration(
            weight_scale=theta[:, :27].to(self.output_dtype),
            bias_scale=theta[:, 27].to(self.output_dtype),
            weight_bias=theta[:, 28:55].to(self.output_dtype),
            bias_bias=theta[:, 55].to(self.output_dtype),
            metadata={
                "samples": self.samples,
                "primitives": self.primitives,
                "ridge": ridge,
                "ridge_prior": ridge_prior,
            },
        )


@dataclass
class RelightingCalibration:
    """Affine local-light operator from Supplementary Eqs. (14)-(16)."""

    weight_scale: Tensor
    bias_scale: Tensor
    weight_bias: Tensor
    bias_bias: Tensor
    metadata: dict[str, Any] | None = None
    format_version: int = 1

    def __post_init__(self) -> None:
        attributes, descriptors = self.weight_scale.shape
        if self.bias_scale.shape != (attributes,):
            raise ValueError("bias_scale has an invalid shape")
        if self.weight_bias.shape != (attributes, descriptors):
            raise ValueError("weight_bias has an invalid shape")
        if self.bias_bias.shape != (attributes,):
            raise ValueError("bias_bias has an invalid shape")
        if descriptors != 27:
            raise ValueError("DecoupleGS local ambient descriptors have 27 values")
        if self.metadata is None:
            self.metadata = {}

    @property
    def attribute_dim(self) -> int:
        return self.weight_scale.shape[0]

    @property
    def descriptor_dim(self) -> int:
        return self.weight_scale.shape[1]

    @classmethod
    def identity(
        cls,
        attribute_dim: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> RelightingCalibration:
        return cls(
            weight_scale=torch.zeros((attribute_dim, 27), dtype=dtype, device=device),
            bias_scale=torch.ones(attribute_dim, dtype=dtype, device=device),
            weight_bias=torch.zeros((attribute_dim, 27), dtype=dtype, device=device),
            bias_bias=torch.zeros(attribute_dim, dtype=dtype, device=device),
            metadata={"kind": "identity"},
        )

    @classmethod
    def fit_ols(
        cls,
        descriptors: Tensor,
        canonical: Tensor,
        targets: Tensor,
        *,
        ridge: float = 0.0,
        ridge_prior: str = "zero",
    ) -> RelightingCalibration:
        """Closed-form global OLS calibration using synthetic HDRI samples.

        ``descriptors`` is ``[N, 27]``.  ``canonical`` can be ``[M, A]`` (one
        canonical asset shared across probes) or ``[N, M, A]``.  Targets are
        always ``[N, M, A]``.
        """

        statistics = RelightingNormalEquations.from_samples(
            descriptors, canonical, targets
        )
        return statistics.solve(ridge=ridge, ridge_prior=ridge_prior)

    def apply(self, canonical_sh: Tensor, descriptor: Tensor) -> Tensor:
        original_shape = canonical_sh.shape
        flattened = canonical_sh.reshape(original_shape[0], -1)
        if flattened.shape[1] != self.attribute_dim:
            raise ValueError(
                f"calibration expects {self.attribute_dim} attributes, got {flattened.shape[1]}"
            )
        descriptor = descriptor.to(device=flattened.device, dtype=flattened.dtype)
        weight_scale = self.weight_scale.to(flattened)
        bias_scale = self.bias_scale.to(flattened)
        weight_bias = self.weight_bias.to(flattened)
        bias_bias = self.bias_bias.to(flattened)
        if descriptor.ndim == 1:
            scale = weight_scale @ descriptor + bias_scale
            bias = weight_bias @ descriptor + bias_bias
            output = flattened * scale[None] + bias[None]
        elif descriptor.ndim == 2 and descriptor.shape[0] == flattened.shape[0]:
            scale = descriptor @ weight_scale.transpose(0, 1) + bias_scale
            bias = descriptor @ weight_bias.transpose(0, 1) + bias_bias
            output = flattened * scale + bias
        else:
            raise ValueError("descriptor must be [27] or one [N, 27] value per Gaussian")
        return output.reshape(original_shape)

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": "OpenDecoupleGS-RelightingCalibration",
            "format_version": self.format_version,
            "weight_scale": self.weight_scale,
            "bias_scale": self.bias_scale,
            "weight_bias": self.weight_bias,
            "bias_bias": self.bias_bias,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> RelightingCalibration:
        state = torch.load(path, map_location=map_location, weights_only=True)
        if state.get("format") != "OpenDecoupleGS-RelightingCalibration":
            raise ValueError(f"{path} is not a DecoupleGS calibration")
        state = dict(state)
        state.pop("format")
        return cls(**state)


def sample_local_probe(
    positions: Tensor,
    background_means: Tensor,
    background_sh: Tensor,
    *,
    visibility: Tensor | None = None,
    config: RelightingConfig | None = None,
    chunk_size: int = 262144,
    spatial_index: BackgroundSpatialIndex | None = None,
) -> Tensor:
    """Aggregate neighboring background SH probes using Eq. (7)."""

    config = RelightingConfig() if config is None else config
    if positions.ndim == 1:
        positions = positions[None]
        squeeze = True
    else:
        squeeze = False
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [B, 3]")
    coeffs = config.descriptor_bands**2
    if background_sh.ndim != 3 or background_sh.shape[1] < coeffs or background_sh.shape[2] != 3:
        raise ValueError("background_sh does not contain the first three SH bands")
    positions = positions.to(background_means)
    descriptors = background_sh[:, :coeffs].reshape(background_sh.shape[0], -1).to(background_means)
    if visibility is None:
        visibility = torch.ones(background_means.shape[0], dtype=background_means.dtype, device=background_means.device)
    visibility = visibility.reshape(-1).to(background_means).clamp_min(0.0)
    if spatial_index is not None and len(spatial_index) != background_means.shape[0]:
        raise ValueError("spatial index and background means have different lengths")
    if spatial_index is not None and config.probe_radius is not None:
        query = spatial_index.query_radius(positions, config.probe_radius)
        numerator = torch.zeros(
            (positions.shape[0], coeffs * 3),
            dtype=background_means.dtype,
            device=background_means.device,
        )
        denominator = torch.zeros(
            positions.shape[0],
            dtype=background_means.dtype,
            device=background_means.device,
        )
        if query.indices.numel():
            candidate_means = background_means[query.indices]
            candidate_descriptors = descriptors[query.indices]
            candidate_visibility = visibility[query.indices]
            squared = (
                positions[query.owners] - candidate_means
            ).square().sum(dim=-1)
            weights = (
                torch.exp(-squared / (2.0 * config.probe_sigma**2))
                * candidate_visibility
            )
            numerator.index_add_(
                0,
                query.owners,
                weights[:, None] * candidate_descriptors,
            )
            denominator.index_add_(0, query.owners, weights)
        nearest_descriptor = descriptors[query.nearest]
        output = numerator / denominator.clamp_min(1e-12)[:, None]
        output = torch.where(
            (denominator > 1e-12)[:, None], output, nearest_descriptor
        )
        return output[0] if squeeze else output
    numerator = torch.zeros((positions.shape[0], coeffs * 3), dtype=background_means.dtype, device=background_means.device)
    denominator = torch.zeros(positions.shape[0], dtype=background_means.dtype, device=background_means.device)
    nearest_distance = torch.full_like(denominator, torch.inf)
    nearest_descriptor = torch.zeros_like(numerator)
    for start in range(0, background_means.shape[0], chunk_size):
        means = background_means[start : start + chunk_size]
        local_descriptor = descriptors[start : start + chunk_size]
        local_visibility = visibility[start : start + chunk_size]
        squared = (positions[:, None] - means[None]).square().sum(dim=-1)
        nearest_value, nearest_index = squared.min(dim=-1)
        update = nearest_value < nearest_distance
        nearest_descriptor[update] = local_descriptor[nearest_index[update]]
        nearest_distance = torch.minimum(nearest_distance, nearest_value)
        weight = torch.exp(-squared / (2.0 * config.probe_sigma**2)) * local_visibility[None]
        if config.probe_radius is not None:
            weight *= squared <= config.probe_radius**2
        numerator += weight @ local_descriptor
        denominator += weight.sum(dim=-1)
    output = numerator / denominator.clamp_min(1e-12)[:, None]
    output = torch.where((denominator > 1e-12)[:, None], output, nearest_descriptor)
    return output[0] if squeeze else output


def descriptor_dc_rgb(descriptor: Tensor) -> Tensor:
    """Convert GraphDeco's stored DC coefficient to linear RGB radiance."""

    if descriptor.shape[-1] < 3:
        raise ValueError("a light descriptor must contain an RGB DC coefficient")
    return descriptor[..., :3] * SH_C0 + 0.5


def dominant_light_intensity(descriptor: Tensor) -> Tensor:
    """Paper definition ``max(0, L_DC)`` in HUGSIM's SH representation."""

    return descriptor_dc_rgb(descriptor).amax(dim=-1).clamp_min(0.0)


def contact_shadow_mask(
    local_ground_coordinates: Tensor,
    footprint: Tensor | tuple[float, float],
    *,
    exponent: float = 4.0,
    decay: float = 2.0,
) -> Tensor:
    """The smoothly decaying super-ellipse in Supplementary Eq. (17)."""

    footprint = torch.as_tensor(footprint, dtype=local_ground_coordinates.dtype, device=local_ground_coordinates.device)
    normalized = local_ground_coordinates.abs() / (footprint.clamp_min(1e-6) / 2.0)
    radius = normalized.pow(exponent).sum(dim=-1).pow(1.0 / exponent)
    return torch.exp(-radius * decay)


def apply_contact_shadow(
    background: GaussianSet,
    asset: GaussianSet,
    asset_transform: Tensor,
    plane: GroundPlane,
    descriptor: Tensor,
    *,
    relighting_config: RelightingConfig | None = None,
    registration_config: RegistrationConfig | None = None,
    ground_mask: Tensor | None = None,
) -> GaussianSet:
    """Apply Eq. (9) to background SH probes under an inserted asset."""

    relighting_config = RelightingConfig() if relighting_config is None else relighting_config
    registration_config = RegistrationConfig() if registration_config is None else registration_config
    asset_transform = asset_transform.to(background.means)
    local = (background.means - asset_transform[:3, 3]) @ asset_transform[:3, :3]
    axes = registration_config.horizontal_axes
    local_ground = local[:, axes]
    minimum, maximum = asset.physical_bounds
    footprint = maximum[list(axes)] - minimum[list(axes)]
    mask = contact_shadow_mask(
        local_ground,
        footprint,
        exponent=relighting_config.shadow_exponent,
        decay=relighting_config.shadow_decay,
    )
    near_plane = plane.signed_distance(background.means).abs() <= relighting_config.shadow_ground_band
    if ground_mask is not None:
        near_plane &= ground_mask.to(device=background.device, dtype=torch.bool)
    mask = mask * near_plane
    if relighting_config.shadow_mask_epsilon is not None:
        mask = torch.where(
            mask >= relighting_config.shadow_mask_epsilon,
            mask,
            torch.zeros_like(mask),
        )
    intensity = dominant_light_intensity(descriptor.to(background.means))
    attenuation = (1.0 - relighting_config.shadow_strength * intensity * mask).clamp_min(0.0)
    sh = background.sh.clone()
    dc_rgb = sh[:, 0] * SH_C0 + 0.5
    sh[:, 0] = (dc_rgb * attenuation[:, None] - 0.5) / SH_C0
    if sh.shape[1] > 1:
        sh[:, 1:] *= attenuation[:, None, None]
    return GaussianSet(
        means=background.means,
        scales=background.scales,
        quats=background.quats,
        opacities=background.opacities,
        sh=sh,
        semantics=background.semantics,
        visibility=background.visibility,
        metadata={**background.metadata, "contact_shadow": True},
    )


def apply_contact_shadows_batched(
    background: GaussianSet,
    assets: Sequence[Any],
    asset_transforms: Sequence[Tensor],
    planes: Sequence[GroundPlane],
    descriptors: Tensor | Sequence[Tensor],
    *,
    relighting_config: RelightingConfig | None = None,
    registration_config: RegistrationConfig | None = None,
    ground_mask: Tensor | None = None,
    chunk_size: int = 65536,
    spatial_index: BackgroundSpatialIndex | None = None,
) -> GaussianSet:
    """Apply multiple contact shadows with one background copy.

    Sequentially applying :func:`apply_contact_shadow` clones and rewrites the
    complete static SH tensor once per vehicle. The multiplicative attenuation
    is associative, so all masks can be evaluated in chunks, multiplied, and
    committed once without changing the analytical shadow model.
    """

    relighting_config = RelightingConfig() if relighting_config is None else relighting_config
    registration_config = RegistrationConfig() if registration_config is None else registration_config
    count = len(assets)
    if count == 0:
        return background
    if len(asset_transforms) != count or len(planes) != count:
        raise ValueError("assets, transforms, and planes must have equal lengths")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if isinstance(descriptors, Tensor):
        descriptor_tensor = descriptors
    else:
        descriptor_tensor = torch.stack(list(descriptors))
    if descriptor_tensor.ndim != 2 or descriptor_tensor.shape[0] != count:
        raise ValueError("descriptors must contain one vector per asset")

    transforms = torch.stack([transform.to(background.means) for transform in asset_transforms])
    normals = torch.stack([plane.normal.to(background.means) for plane in planes])
    offsets = torch.stack([plane.offset.to(background.means) for plane in planes]).reshape(-1)
    axes = registration_config.horizontal_axes
    footprints = []
    for asset in assets:
        minimum, maximum = asset.physical_bounds
        footprints.append((maximum - minimum)[list(axes)].to(background.means))
    footprint_tensor = torch.stack(footprints)
    intensities = dominant_light_intensity(descriptor_tensor.to(background.means))
    if ground_mask is not None:
        ground_mask = ground_mask.to(device=background.device, dtype=torch.bool)
        if ground_mask.shape != (len(background),):
            raise ValueError("ground_mask must contain one value per background Gaussian")

    # Engineering hypothesis H-RUNTIME-03: the paper's parameterized shadow
    # mask is spatially truncated below a negligible attenuation.  The bound is
    # explicit and optional; epsilon=None retains the exact dense equation.
    if (
        spatial_index is not None
        and ground_mask is None
        and relighting_config.shadow_mask_epsilon is not None
    ):
        if len(spatial_index) != len(background):
            raise ValueError("spatial index and background means have different lengths")
        epsilon = relighting_config.shadow_mask_epsilon
        normalized_radius = -torch.log(
            background.means.new_tensor(epsilon)
        ) / relighting_config.shadow_decay
        half_footprints = footprint_tensor / 2.0
        horizontal_bound = normalized_radius * torch.linalg.vector_norm(
            half_footprints, dim=-1
        )
        local_normals = torch.einsum(
            "bij,bj->bi", transforms[:, :3, :3].transpose(1, 2), normals
        )
        vertical = registration_config.vertical_axis
        plane_at_origin = (
            transforms[:, :3, 3] * normals
        ).sum(dim=-1) + offsets
        horizontal_normal_bound = normalized_radius * (
            local_normals[:, list(axes)].abs() * half_footprints
        ).sum(dim=-1)
        vertical_normal = local_normals[:, vertical].abs()
        if bool((vertical_normal < 1e-4).any()):
            spatial_index = None
        else:
            local_vertical_bound = (
                plane_at_origin.abs()
                + horizontal_normal_bound
                + relighting_config.shadow_ground_band
            ) / vertical_normal
            query_radius = torch.sqrt(
                horizontal_bound.square() + local_vertical_bound.square()
            ).clamp_min(1e-6)
            query = spatial_index.query_radius(
                transforms[:, :3, 3], query_radius
            )
            attenuation = torch.ones(
                len(background),
                dtype=background.means.dtype,
                device=background.device,
            )
            if query.indices.numel():
                owners = query.owners
                means = background.means[query.indices]
                delta = means - transforms[owners, :3, 3]
                local = torch.einsum(
                    "ni,nij->nj", delta, transforms[owners, :3, :3]
                )
                local_ground = local[:, list(axes)]
                normalized = local_ground.abs() / (
                    footprint_tensor[owners].clamp_min(1e-6) / 2.0
                )
                radius = normalized.pow(
                    relighting_config.shadow_exponent
                ).sum(dim=-1).pow(1.0 / relighting_config.shadow_exponent)
                masks = torch.exp(-radius * relighting_config.shadow_decay)
                near_plane = (
                    (means * normals[owners]).sum(dim=-1) + offsets[owners]
                ).abs() <= relighting_config.shadow_ground_band
                if ground_mask is not None:
                    near_plane &= ground_mask[query.indices]
                masks = torch.where(
                    near_plane & (masks >= epsilon), masks, torch.zeros_like(masks)
                )
                factors = (
                    1.0
                    - relighting_config.shadow_strength
                    * intensities[owners]
                    * masks
                ).clamp_min(0.0)
                attenuation.scatter_reduce_(
                    0,
                    query.indices,
                    factors,
                    reduce="prod",
                    include_self=True,
                )

            sh = background.sh.clone()
            dc_rgb = sh[:, 0] * SH_C0 + 0.5
            sh[:, 0] = (dc_rgb * attenuation[:, None] - 0.5) / SH_C0
            if sh.shape[1] > 1:
                sh[:, 1:] *= attenuation[:, None, None]
            return GaussianSet(
                means=background.means,
                scales=background.scales,
                quats=background.quats,
                opacities=background.opacities,
                sh=sh,
                semantics=background.semantics,
                visibility=background.visibility,
                metadata={
                    **background.metadata,
                    "contact_shadow": True,
                    "contact_shadow_batch_size": count,
                    "contact_shadow_indexed": True,
                    "contact_shadow_mask_epsilon": epsilon,
                    "contact_shadow_candidates": int(query.indices.numel()),
                },
            )

    # Contact shadows are defined on the estimated road plane. When semantic
    # ground labels are available, evaluate the analytical mask only for those
    # Gaussians instead of allocating K x N intermediates for buildings, sky,
    # vegetation, and vehicles that are guaranteed to remain unchanged.
    shadow_indices = (
        None
        if ground_mask is None
        else torch.nonzero(ground_mask, as_tuple=False).squeeze(-1)
    )
    shadow_means = (
        background.means
        if shadow_indices is None
        else background.means[shadow_indices]
    )
    attenuation = torch.empty(
        shadow_means.shape[0],
        dtype=background.means.dtype,
        device=background.device,
    )
    for start in range(0, shadow_means.shape[0], chunk_size):
        stop = min(start + chunk_size, shadow_means.shape[0])
        means = shadow_means[start:stop]
        delta = means[None] - transforms[:, None, :3, 3]
        local = torch.einsum("bni,bij->bnj", delta, transforms[:, :3, :3])
        local_ground = local[..., list(axes)]
        normalized = local_ground.abs() / (footprint_tensor[:, None].clamp_min(1e-6) / 2.0)
        radius = normalized.pow(relighting_config.shadow_exponent).sum(dim=-1).pow(
            1.0 / relighting_config.shadow_exponent
        )
        masks = torch.exp(-radius * relighting_config.shadow_decay)
        near_plane = (
            torch.einsum("ni,bi->bn", means, normals) + offsets[:, None]
        ).abs() <= relighting_config.shadow_ground_band
        masks *= near_plane
        if relighting_config.shadow_mask_epsilon is not None:
            masks = torch.where(
                masks >= relighting_config.shadow_mask_epsilon,
                masks,
                torch.zeros_like(masks),
            )
        factors = (
            1.0
            - relighting_config.shadow_strength * intensities[:, None] * masks
        ).clamp_min(0.0)
        attenuation[start:stop] = factors.prod(dim=0)

    sh = background.sh.clone()
    if shadow_indices is None:
        dc_rgb = sh[:, 0] * SH_C0 + 0.5
        sh[:, 0] = (dc_rgb * attenuation[:, None] - 0.5) / SH_C0
        if sh.shape[1] > 1:
            sh[:, 1:] *= attenuation[:, None, None]
    else:
        dc_rgb = sh[shadow_indices, 0] * SH_C0 + 0.5
        sh[shadow_indices, 0] = (
            dc_rgb * attenuation[:, None] - 0.5
        ) / SH_C0
        if sh.shape[1] > 1:
            sh[shadow_indices, 1:] = (
                sh[shadow_indices, 1:] * attenuation[:, None, None]
            )
    return GaussianSet(
        means=background.means,
        scales=background.scales,
        quats=background.quats,
        opacities=background.opacities,
        sh=sh,
        semantics=background.semantics,
        visibility=background.visibility,
        metadata={
            **background.metadata,
            "contact_shadow": True,
            "contact_shadow_batch_size": count,
            "contact_shadow_semantic_candidates": int(shadow_means.shape[0]),
        },
    )


def relight_asset(asset: GaussianSet, descriptor: Tensor, calibration: RelightingCalibration) -> GaussianSet:
    return GaussianSet(
        means=asset.means,
        scales=asset.scales,
        quats=asset.quats,
        opacities=asset.opacities,
        sh=calibration.apply(asset.sh, descriptor),
        semantics=asset.semantics,
        visibility=asset.visibility,
        metadata={**asset.metadata, "relit": True},
    )


def adaptive_relight_asset(
    asset: GaussianSet,
    descriptor: Tensor,
    config: RelightingConfig | None = None,
) -> GaussianSet:
    """Practical scene-adaptive fallback when author HDRI calibration is absent.

    The paper-exact path is :func:`relight_asset` with an OLS calibration
    sidecar. Released third-party assets do not contain that sidecar, so this
    representation-aware fallback transfers local exposure and RGB tint while
    preserving each primitive's canonical contrast and directional SH detail.
    """

    config = RelightingConfig() if config is None else config
    if descriptor.ndim != 1 or descriptor.shape[0] < 3:
        raise ValueError("adaptive relighting expects one flattened local light descriptor")
    ambient_rgb = descriptor_dc_rgb(descriptor.to(asset.means)).clamp_min(0.0)
    raw_gain = ambient_rgb / config.adaptive_reference_intensity
    gain = 1.0 + config.adaptive_strength * (raw_gain - 1.0)
    gain = gain.clamp(config.adaptive_min_gain, config.adaptive_max_gain)

    sh = asset.sh.clone()
    dc_rgb = sh[:, 0] * SH_C0 + 0.5
    sh[:, 0] = (dc_rgb * gain[None] - 0.5) / SH_C0
    if sh.shape[1] > 1:
        sh[:, 1:] *= gain[None, None]
    return GaussianSet(
        means=asset.means,
        scales=asset.scales,
        quats=asset.quats,
        opacities=asset.opacities,
        sh=sh,
        semantics=asset.semantics,
        visibility=asset.visibility,
        metadata={
            **asset.metadata,
            "relit": True,
            "relighting_kind": "adaptive_fallback",
            "adaptive_gain": gain.detach().cpu().tolist(),
        },
    )


def relight_compact_codebook(
    asset: Any,
    descriptor: Tensor,
    *,
    calibration: RelightingCalibration | None = None,
    config: RelightingConfig | None = None,
) -> Any:
    """Relight one VQ codebook instead of expanding per-primitive SH tensors."""

    from dataclasses import replace

    from .compression import CompactGaussianAsset

    if not isinstance(asset, CompactGaussianAsset):
        raise TypeError("asset must be a CompactGaussianAsset")
    coefficients = asset.color_codebook.reshape(
        -1, (asset.sh_degree + 1) ** 2, 3
    )
    if calibration is not None:
        relit = calibration.apply(coefficients, descriptor)
        kind = "ols"
    else:
        config = RelightingConfig() if config is None else config
        if descriptor.ndim != 1 or descriptor.shape[0] < 3:
            raise ValueError("adaptive relighting expects one flattened local light descriptor")
        ambient_rgb = descriptor_dc_rgb(descriptor.to(coefficients)).clamp_min(0.0)
        raw_gain = ambient_rgb / config.adaptive_reference_intensity
        gain = 1.0 + config.adaptive_strength * (raw_gain - 1.0)
        gain = gain.clamp(config.adaptive_min_gain, config.adaptive_max_gain)
        relit = coefficients.clone()
        dc_rgb = relit[:, 0] * SH_C0 + 0.5
        relit[:, 0] = (dc_rgb * gain[None] - 0.5) / SH_C0
        if relit.shape[1] > 1:
            relit[:, 1:] *= gain[None, None]
        kind = "adaptive_fallback"
    metadata = {**asset.metadata, "relit": True, "relighting_kind": kind}
    return replace(
        asset,
        color_codebook=relit.reshape_as(asset.color_codebook),
        metadata=metadata,
    )
