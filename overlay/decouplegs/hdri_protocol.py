"""Reproducible public-HDRI supervision for DecoupleGS relighting R&D.

The DecoupleGS supplement specifies the global affine OLS operator, but does
not release the HDRI set, relighting renderer, materials, or per-primitive
targets used to fit it.  This module keeps replacement supervision explicit:

* ``global_affine`` is an exactly representable operator-oracle test;
* ``covariance_diffuse`` is a physically motivated stress test using the
  smallest Gaussian covariance axis as a surface-normal proxy.

Neither target is the unpublished author supervision.  Callers must preserve
that distinction in reports and deployment metadata.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# OpenCV ships its EXR codec behind this runtime opt-in in several wheels.  It
# must be set before importing cv2; otherwise a valid EXR raises error -213.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch
from torch import Tensor

from .relighting import RelightingCalibration, SH_C0
from .transforms import quaternion_to_matrix, real_sh_basis


TARGET_MODELS = ("global_affine", "covariance_diffuse")
SPLITS = ("train", "validation", "test")
TARGET_DEFINITION_VERSION = 2


@dataclass(frozen=True)
class HDRIEnvironment:
    """One immutable environment entry in a calibration protocol."""

    identifier: str
    path: Path
    split: str

    def __post_init__(self) -> None:
        if not self.identifier:
            raise ValueError("HDRI identifiers must be non-empty")
        if self.split not in SPLITS:
            raise ValueError(f"HDRI split must be one of {SPLITS}, got {self.split!r}")


@dataclass
class HDRIProbeSet:
    """Projected degree-2 GraphDeco SH descriptors and their provenance."""

    descriptors: Tensor
    metadata: list[dict[str, Any]]
    environments: list[dict[str, Any]]

    def __post_init__(self) -> None:
        if self.descriptors.ndim != 2 or self.descriptors.shape[1] != 27:
            raise ValueError("HDRI descriptors must have shape [N, 27]")
        if len(self.metadata) != self.descriptors.shape[0]:
            raise ValueError("probe metadata and descriptor counts differ")
        for split in SPLITS:
            if not self.mask(split).any():
                raise ValueError(f"HDRI probe set has no {split} probes")

    def mask(self, split: str) -> Tensor:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}")
        return torch.tensor(
            [entry["split"] == split for entry in self.metadata], dtype=torch.bool
        )

    @property
    def descriptor_sha256(self) -> str:
        payload = self.descriptors.detach().cpu().contiguous().numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hdr(path: str | Path, width: int) -> np.ndarray:
    """Load an HDR/EXR panorama as finite non-negative linear RGB."""

    path = Path(path)
    if width <= 0:
        raise ValueError("environment width must be positive")
    try:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    except cv2.error as error:
        raise ValueError(f"OpenCV could not decode HDRI {path}: {error}") from error
    if image is None:
        raise ValueError(f"OpenCV could not decode {path}")
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"HDRI must have at least three channels: {path}")
    image = image[..., :3][..., ::-1].astype(np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    image = np.maximum(image, 0.0)
    height = max(1, width // 2)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def tonemap(image: np.ndarray, exposure: float) -> np.ndarray:
    """Normalize an HDR panorama and apply a bounded photographic response."""

    if exposure <= 0:
        raise ValueError("probe exposure must be positive")
    luminance = image @ np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
    positive = luminance[luminance > 0]
    scale = float(np.percentile(positive, 90)) if positive.size else 1.0
    linear = image * (exposure / max(scale, 1e-6))
    return np.clip(1.0 - np.exp(-linear), 0.0, 1.0)


def panorama_geometry(height: int, width: int) -> tuple[Tensor, Tensor]:
    theta = (torch.arange(height, dtype=torch.float64) + 0.5) * (math.pi / height)
    phi = (torch.arange(width, dtype=torch.float64) + 0.5) * (2.0 * math.pi / width) - math.pi
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing="ij")
    directions = torch.stack(
        (
            torch.sin(theta_grid) * torch.cos(phi_grid),
            torch.cos(theta_grid),
            torch.sin(theta_grid) * torch.sin(phi_grid),
        ),
        dim=-1,
    )
    solid_angle = torch.sin(theta_grid) * (math.pi / height) * (2.0 * math.pi / width)
    return directions.reshape(-1, 3), solid_angle.reshape(-1)


def project_graphdeco_sh(image: np.ndarray) -> Tensor:
    """Project equirectangular RGB into GraphDeco's centered degree-2 SH."""

    height, width = image.shape[:2]
    directions, solid_angle = panorama_geometry(height, width)
    basis = real_sh_basis(directions, degree=2).to(torch.float64)
    centered = torch.from_numpy(image.reshape(-1, 3)).to(torch.float64) - 0.5
    coefficients = torch.einsum("pc,pk,p->kc", centered, basis, solid_angle)
    return coefficients.to(torch.float32)


def build_probe_set(
    environments: Sequence[HDRIEnvironment],
    *,
    environment_width: int = 192,
    yaw_rotations: int = 4,
    exposures: Sequence[float] = (0.45, 0.8, 1.2),
    tints: Sequence[Sequence[float]] = (
        (1.0, 1.0, 1.0),
        (1.30, 0.90, 0.68),
        (0.68, 0.92, 1.30),
    ),
) -> HDRIProbeSet:
    """Build probes while keeping every panorama wholly inside one split."""

    if yaw_rotations <= 0:
        raise ValueError("yaw_rotations must be positive")
    if len(environments) != len({entry.identifier for entry in environments}):
        raise ValueError("HDRI identifiers must be unique")
    missing = [str(entry.path) for entry in environments if not entry.path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    tint_arrays = [np.asarray(tint, dtype=np.float32) for tint in tints]
    if not tint_arrays or any(tint.shape != (3,) or np.any(tint <= 0) for tint in tint_arrays):
        raise ValueError("tints must be positive RGB triples")
    if not exposures or any(exposure <= 0 for exposure in exposures):
        raise ValueError("exposures must be non-empty and positive")

    descriptors: list[Tensor] = []
    probe_metadata: list[dict[str, Any]] = []
    environment_metadata: list[dict[str, Any]] = []
    for hdri_index, environment in enumerate(environments):
        image = load_hdr(environment.path, environment_width)
        luminance = image @ np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
        environment_metadata.append(
            {
                "index": hdri_index,
                "id": environment.identifier,
                "split": environment.split,
                "path": str(environment.path.resolve()),
                "sha256": sha256_file(environment.path),
                "file_bytes": environment.path.stat().st_size,
                "projected_shape": list(image.shape),
                "linear_luminance_p50": float(np.percentile(luminance, 50)),
                "linear_luminance_p99": float(np.percentile(luminance, 99)),
                "linear_luminance_max": float(luminance.max()),
            }
        )
        for rotation_index in range(yaw_rotations):
            shifted = np.roll(
                image,
                rotation_index * image.shape[1] // yaw_rotations,
                axis=1,
            )
            for exposure in exposures:
                base_mapped = tonemap(shifted, float(exposure))
                for tint_index, tint in enumerate(tint_arrays):
                    mapped = np.clip(base_mapped * tint[None, None], 0.0, 1.0)
                    descriptors.append(project_graphdeco_sh(mapped).reshape(-1))
                    probe_metadata.append(
                        {
                            "hdri_index": hdri_index,
                            "hdri_id": environment.identifier,
                            "split": environment.split,
                            "path": str(environment.path.resolve()),
                            "yaw_degrees": rotation_index * 360.0 / yaw_rotations,
                            "exposure": float(exposure),
                            "tint_index": tint_index,
                            "tint_rgb": tint.tolist(),
                        }
                    )
    return HDRIProbeSet(
        descriptors=torch.stack(descriptors),
        metadata=probe_metadata,
        environments=environment_metadata,
    )


def estimate_normals(asset: Any) -> Tensor:
    """Use each splat's thinnest covariance axis as an outward normal proxy."""

    rotations = quaternion_to_matrix(asset.quats)
    smallest_axis = asset.scales.argmin(dim=-1)
    gather = smallest_axis[:, None, None].expand(-1, 3, 1)
    normals = torch.gather(rotations, 2, gather).squeeze(-1)
    center = asset.means.median(dim=0).values
    outward = asset.means - center
    sign = torch.where((normals * outward).sum(dim=-1, keepdim=True) < 0, -1.0, 1.0)
    return torch.nn.functional.normalize(normals * sign, dim=-1)


def covariance_diffuse_targets(canonical: Tensor, normals: Tensor, descriptors: Tensor) -> Tensor:
    """Cosine-convolved irradiance target without unpublished mesh/materials."""

    if canonical.ndim != 3 or canonical.shape[-1] != 3:
        raise ValueError("canonical SH must have shape [M, C, 3]")
    if normals.shape != (canonical.shape[0], 3):
        raise ValueError("normal proxy count must match canonical primitives")
    if descriptors.ndim != 2 or descriptors.shape[1] != 27:
        raise ValueError("descriptors must have shape [N, 27]")
    normal_basis = real_sh_basis(normals, degree=2).to(canonical)
    band_kernel = canonical.new_tensor(
        (
            math.pi,
            2 * math.pi / 3,
            2 * math.pi / 3,
            2 * math.pi / 3,
            math.pi / 4,
            math.pi / 4,
            math.pi / 4,
            math.pi / 4,
            math.pi / 4,
        )
    )
    probes = descriptors.reshape(-1, 9, 3)
    irradiance = 0.5 + torch.einsum(
        "mk,nkc->nmc", normal_basis * band_kernel[None], probes
    ) / math.pi
    irradiance = irradiance.clamp(0.04, 1.5)
    gain = (irradiance / 0.5).clamp(0.1, 3.0)
    targets = canonical[None] * gain[:, :, None, :]
    canonical_rgb = (canonical[:, 0] * SH_C0 + 0.5).clamp(0.0, 1.0)
    target_rgb = (canonical_rgb[None] * gain).clamp(0.0, 1.0)
    targets[:, :, 0] = (target_rgb - 0.5) / SH_C0
    return targets


def global_affine_targets(canonical: Tensor, descriptors: Tensor) -> Tensor:
    """An exactly representable oracle for Supplementary Eqs. (15)-(16)."""

    if canonical.ndim != 3 or canonical.shape[-1] != 3:
        raise ValueError("canonical SH must have shape [M, C, 3]")
    if descriptors.ndim != 2 or descriptors.shape[1] != 27:
        raise ValueError("descriptors must have shape [N, 27]")
    probes = descriptors.reshape(-1, 9, 3)
    ambient = (probes[:, 0] * SH_C0 + 0.5).clamp(0.0, 1.0)
    gain = (0.35 + 1.30 * ambient).clamp(0.2, 1.8)
    targets = canonical[None] * gain[:, None, None, :]
    targets[:, :, :9] += 0.08 * probes[:, None]
    # Do not clamp/re-encode DC here: any display-domain clamp would add a
    # nonlinearity and invalidate this target's role as an Eq. (15) oracle.
    return targets


def make_targets(
    target_model: str,
    canonical: Tensor,
    descriptors: Tensor,
    *,
    normals: Tensor | None = None,
) -> Tensor:
    if target_model == "global_affine":
        return global_affine_targets(canonical, descriptors)
    if target_model == "covariance_diffuse":
        if normals is None:
            raise ValueError("covariance_diffuse targets require normal proxies")
        return covariance_diffuse_targets(canonical, normals, descriptors)
    raise ValueError(f"target_model must be one of {TARGET_MODELS}, got {target_model!r}")


def angular_degrees(predicted: Tensor, target: Tensor) -> Tensor:
    predicted_rgb = (predicted[..., 0, :] * SH_C0 + 0.5).clamp(1e-6, 1.0)
    target_rgb = (target[..., 0, :] * SH_C0 + 0.5).clamp(1e-6, 1.0)
    cosine = torch.nn.functional.cosine_similarity(
        predicted_rgb, target_rgb, dim=-1
    ).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def error_metrics(predicted: Tensor, target: Tensor) -> dict[str, float]:
    """Primitive proxy diagnostics; these are not masked image metrics."""

    if predicted.shape != target.shape or predicted.ndim != 4:
        raise ValueError("predicted and target must share shape [N, M, C, 3]")
    angle = angular_degrees(predicted, target).reshape(-1)
    predicted_rgb = (predicted[..., 0, :] * SH_C0 + 0.5).clamp(0.0, 1.0)
    target_rgb = (target[..., 0, :] * SH_C0 + 0.5).clamp(0.0, 1.0)
    luminance_weights = predicted_rgb.new_tensor((0.2126, 0.7152, 0.0722))
    intensity_error = ((predicted_rgb - target_rgb) * luminance_weights).sum(dim=-1).abs()
    mse = (predicted_rgb - target_rgb).square().sum(dim=-1).mean()
    target_luminance = target_rgb @ luminance_weights
    support = target_luminance.reshape(-1) >= 0.05
    supported_angle = angle[support]
    if not supported_angle.numel():
        supported_angle = angle
    return {
        "angular_mean_deg": float(angle.mean()),
        "angular_p95_deg": float(torch.quantile(angle, 0.95)),
        "peak_angular_error_deg": float(angle.max()),
        "photometric_support_fraction": float(support.float().mean()),
        "supported_angular_mean_deg": float(supported_angle.mean()),
        "supported_angular_p95_deg": float(torch.quantile(supported_angle, 0.95)),
        "supported_peak_angular_error_deg": float(supported_angle.max()),
        "peak_intensity_error": float(intensity_error.max()),
        "primitive_rgb_psnr_l2_db": float(-10.0 * torch.log10(mse.clamp_min(1e-12))),
    }


def apply_probe_batch(
    calibration: RelightingCalibration,
    canonical: Tensor,
    descriptors: Tensor,
) -> Tensor:
    """Apply one global calibration to every descriptor without Python loops."""

    flattened = canonical.reshape(canonical.shape[0], -1)
    if flattened.shape[1] != calibration.attribute_dim:
        raise ValueError("canonical attribute count and calibration differ")
    descriptors = descriptors.to(device=flattened.device, dtype=flattened.dtype)
    weight_scale = calibration.weight_scale.to(flattened)
    bias_scale = calibration.bias_scale.to(flattened)
    weight_bias = calibration.weight_bias.to(flattened)
    bias_bias = calibration.bias_bias.to(flattened)
    scale = descriptors @ weight_scale.transpose(0, 1) + bias_scale
    bias = descriptors @ weight_bias.transpose(0, 1) + bias_bias
    output = flattened[None] * scale[:, None] + bias[:, None]
    return output.reshape(descriptors.shape[0], *canonical.shape)


def validation_score(metrics: dict[str, float]) -> tuple[float, float, float]:
    """Deterministic ridge-selection objective, independent of test probes."""

    return (
        metrics["supported_angular_mean_deg"],
        metrics["supported_angular_p95_deg"],
        -metrics["primitive_rgb_psnr_l2_db"],
    )


def improvement_gate(
    baseline: dict[str, float],
    candidate: dict[str, float],
    *,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Require simultaneous chromaticity, intensity, and RGB improvement."""

    comparisons = {
        "supported_angular_mean_nonworse": candidate["supported_angular_mean_deg"]
        <= baseline["supported_angular_mean_deg"] + tolerance,
        "supported_angular_p95_nonworse": candidate["supported_angular_p95_deg"]
        <= baseline["supported_angular_p95_deg"] + tolerance,
        "peak_intensity_nonworse": candidate["peak_intensity_error"]
        <= baseline["peak_intensity_error"] + tolerance,
        "primitive_psnr_nonworse": candidate["primitive_rgb_psnr_l2_db"] + tolerance
        >= baseline["primitive_rgb_psnr_l2_db"],
    }
    finite = all(math.isfinite(float(value)) for value in candidate.values())
    return {
        "passed": finite and all(comparisons.values()),
        "finite": finite,
        "criteria": comparisons,
        "delta": {
            key: candidate[key] - baseline[key]
            for key in (
                "supported_angular_mean_deg",
                "supported_angular_p95_deg",
                "peak_intensity_error",
                "primitive_rgb_psnr_l2_db",
            )
        },
    }


def descriptor_domain(descriptors: Tensor) -> dict[str, Any]:
    """Persist enough training-domain information for runtime OOD audits."""

    values = descriptors.to(torch.float32)
    return {
        "minimum": values.amin(dim=0).tolist(),
        "maximum": values.amax(dim=0).tolist(),
        "mean": values.mean(dim=0).tolist(),
        "std": values.std(dim=0, unbiased=False).tolist(),
    }
