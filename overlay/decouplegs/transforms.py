from __future__ import annotations

import math
from functools import lru_cache

import torch
from torch import Tensor

from .types import GaussianSet

_C0 = 0.28209479177387814
_C1 = 0.4886025119029199
_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def normalize_quaternion(quaternion: Tensor) -> Tensor:
    return torch.nn.functional.normalize(quaternion, dim=-1)


def quaternion_multiply(left: Tensor, right: Tensor) -> Tensor:
    """Hamilton product for ``(w, x, y, z)`` quaternions."""

    aw, ax, ay, az = left.unbind(-1)
    bw, bx, by, bz = right.unbind(-1)
    result = torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )
    result = normalize_quaternion(result)
    return torch.where(result[..., :1] < 0, -result, result)


def quaternion_to_matrix(quaternion: Tensor) -> Tensor:
    q = normalize_quaternion(quaternion)
    w, x, y, z = q.unbind(-1)
    two = 2.0
    return torch.stack(
        (
            1 - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1 - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1 - two * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def _sqrt_positive_part(value: Tensor) -> Tensor:
    return torch.sqrt(torch.clamp(value, min=0.0))


def matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """Convert rotation matrices to stable ``(w, x, y, z)`` quaternions."""

    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"matrix must end in [3, 3], got {tuple(matrix.shape)}")
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    q_abs = _sqrt_positive_part(
        torch.stack(
            (
                1 + m00 + m11 + m22,
                1 + m00 - m11 - m22,
                1 - m00 + m11 - m22,
                1 - m00 - m11 + m22,
            ),
            dim=-1,
        )
    )
    candidates = torch.stack(
        (
            torch.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21), dim=-1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2), dim=-1),
        ),
        dim=-2,
    )
    candidates = candidates / (2.0 * q_abs[..., None].clamp_min(0.1))
    choice = q_abs.argmax(dim=-1)
    gather_index = choice[..., None, None].expand(choice.shape + (1, 4))
    result = torch.gather(candidates, -2, gather_index).squeeze(-2)
    result = normalize_quaternion(result)
    return torch.where(result[..., :1] < 0, -result, result)


def real_sh_basis(directions: Tensor, degree: int) -> Tensor:
    """GraphDeco/gsplat real SH basis through degree three."""

    if degree < 0 or degree > 3:
        raise ValueError("DecoupleGS uses SH degrees in [0, 3]")
    directions = torch.nn.functional.normalize(directions, dim=-1)
    x, y, z = directions.unbind(-1)
    values = [torch.full_like(x, _C0)]
    if degree >= 1:
        values.extend((-_C1 * y, _C1 * z, -_C1 * x))
    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        values.extend(
            (
                _C2[0] * x * y,
                _C2[1] * y * z,
                _C2[2] * (2 * zz - xx - yy),
                _C2[3] * x * z,
                _C2[4] * (xx - yy),
            )
        )
    if degree >= 3:
        xx, yy, zz = x * x, y * y, z * z
        values.extend(
            (
                _C3[0] * y * (3 * xx - yy),
                _C3[1] * x * y * z,
                _C3[2] * y * (4 * zz - xx - yy),
                _C3[3] * z * (2 * zz - 3 * xx - 3 * yy),
                _C3[4] * x * (4 * zz - xx - yy),
                _C3[5] * z * (xx - yy),
                _C3[6] * x * (xx - 3 * yy),
            )
        )
    return torch.stack(values, dim=-1)


@lru_cache(maxsize=4)
def _fibonacci_directions(count: int) -> Tensor:
    index = torch.arange(count, dtype=torch.float64)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    y = 1.0 - 2.0 * (index + 0.5) / count
    radius = torch.sqrt(torch.clamp(1.0 - y * y, min=0.0))
    theta = golden_angle * index
    return torch.stack((radius * torch.cos(theta), y, radius * torch.sin(theta)), dim=-1)


class RealSHRotator:
    """Numerically construct the real-valued Wigner-D action used by Eq. (3).

    Each band is solved independently on a fixed, overdetermined spherical
    collocation grid.  This is algebraically equivalent to applying a real
    Wigner-D matrix while exactly matching the basis signs/order used by gsplat.
    """

    def __init__(self, samples_per_band: int = 32) -> None:
        if samples_per_band < 16:
            raise ValueError("samples_per_band must be at least 16")
        self.samples_per_band = samples_per_band
        self._basis_cache: dict[tuple[int, str, torch.dtype], tuple[Tensor, Tensor]] = {}

    def _collocation(self, degree: int, reference: Tensor) -> tuple[Tensor, Tensor]:
        key = (degree, str(reference.device), reference.dtype)
        cached = self._basis_cache.get(key)
        if cached is not None:
            return cached
        count = max(self.samples_per_band, 4 * (2 * degree + 1))
        directions = _fibonacci_directions(count).to(device=reference.device, dtype=reference.dtype)
        basis = real_sh_basis(directions, degree)[..., degree * degree : (degree + 1) ** 2]
        inverse = torch.linalg.pinv(basis)
        self._basis_cache[key] = (directions, inverse)
        return directions, inverse

    def matrix(self, rotation: Tensor, degree: int) -> Tensor:
        if rotation.shape != (3, 3):
            raise ValueError("one 3x3 rotation is required")
        if degree < 0 or degree > 3:
            raise ValueError("degree must be in [0, 3]")
        coeffs = (degree + 1) ** 2
        output = torch.zeros((coeffs, coeffs), dtype=rotation.dtype, device=rotation.device)
        for band in range(degree + 1):
            directions, inverse = self._collocation(band, rotation)
            # Row-vector form of R^T d: d_local = d_world @ R.
            local_directions = directions @ rotation
            rotated_basis = real_sh_basis(local_directions, band)[..., band * band : (band + 1) ** 2]
            block = inverse @ rotated_basis
            start, end = band * band, (band + 1) ** 2
            output[start:end, start:end] = block
        return output

    def __call__(self, coefficients: Tensor, rotation: Tensor) -> Tensor:
        if coefficients.ndim != 3 or coefficients.shape[-1] != 3:
            raise ValueError("coefficients must have shape [N, C, 3]")
        degree = int(coefficients.shape[1] ** 0.5) - 1
        d_matrix = self.matrix(rotation.to(coefficients), degree)
        return torch.einsum("ij,njc->nic", d_matrix, coefficients)


_DEFAULT_SH_ROTATOR = RealSHRotator()


def transform_gaussians(
    gaussians: GaussianSet,
    transform: Tensor,
    *,
    rotate_sh: bool = True,
    sh_rotator: RealSHRotator | None = None,
) -> GaussianSet:
    """Apply the object LCS-to-WCS transform from Eqs. (2)-(3)."""

    if transform.shape != (4, 4):
        raise ValueError(f"transform must be [4, 4], got {tuple(transform.shape)}")
    transform = transform.to(device=gaussians.device, dtype=gaussians.dtype)
    rotation, translation = transform[:3, :3], transform[:3, 3]
    means = gaussians.means @ rotation.transpose(0, 1) + translation
    world_rotation = matrix_to_quaternion(rotation)
    quats = quaternion_multiply(world_rotation.expand_as(gaussians.quats), gaussians.quats)
    rotator = _DEFAULT_SH_ROTATOR if sh_rotator is None else sh_rotator
    sh = rotator(gaussians.sh, rotation) if rotate_sh and gaussians.sh_degree > 0 else gaussians.sh
    metadata = dict(gaussians.metadata)
    metadata["lcs_to_wcs"] = transform.detach().cpu()
    return GaussianSet(
        means=means,
        scales=gaussians.scales,
        quats=quats,
        opacities=gaussians.opacities,
        sh=sh,
        semantics=gaussians.semantics,
        visibility=gaussians.visibility,
        metadata=metadata,
    )


def covariance_from_scale_quaternion(scales: Tensor, quaternions: Tensor) -> Tensor:
    rotation = quaternion_to_matrix(quaternions)
    diagonal = torch.diag_embed(scales.square())
    return rotation @ diagonal @ rotation.transpose(-1, -2)


def covariance_to_scale_quaternion(covariance: Tensor, epsilon: float = 1e-10) -> tuple[Tensor, Tensor]:
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(epsilon)
    # eigh returns eigenvectors in columns.  Enforce a proper SO(3) frame.
    determinant = torch.linalg.det(eigenvectors)
    eigenvectors = eigenvectors.clone()
    eigenvectors[..., :, 0] *= torch.where(determinant < 0, -1.0, 1.0)[..., None]
    return torch.sqrt(eigenvalues), matrix_to_quaternion(eigenvectors)


def pack_symmetric(covariance: Tensor) -> Tensor:
    return covariance[..., (0, 0, 0, 1, 1, 2), (0, 1, 2, 1, 2, 2)]


def unpack_symmetric(packed: Tensor) -> Tensor:
    if packed.shape[-1] != 6:
        raise ValueError("packed covariance must have six values")
    output = torch.zeros(packed.shape[:-1] + (3, 3), dtype=packed.dtype, device=packed.device)
    output[..., 0, 0] = packed[..., 0]
    output[..., 0, 1] = output[..., 1, 0] = packed[..., 1]
    output[..., 0, 2] = output[..., 2, 0] = packed[..., 2]
    output[..., 1, 1] = packed[..., 3]
    output[..., 1, 2] = output[..., 2, 1] = packed[..., 4]
    output[..., 2, 2] = packed[..., 5]
    return output


def points_in_camera_frustum(
    points_world: Tensor,
    world_to_camera: Tensor,
    intrinsics: Tensor,
    width: int,
    height: int,
    *,
    near: float = 0.01,
    far: float = 500.0,
    margin_pixels: float = 32.0,
) -> Tensor:
    """Conservative point/frustum culling used before unified rasterization."""

    camera = points_world @ world_to_camera[:3, :3].transpose(0, 1) + world_to_camera[:3, 3]
    depth = camera[:, 2]
    projected = camera @ intrinsics.transpose(0, 1)
    uv = projected[:, :2] / projected[:, 2:3].clamp_min(1e-8)
    return (
        (depth >= near)
        & (depth <= far)
        & (uv[:, 0] >= -margin_pixels)
        & (uv[:, 0] < width + margin_pixels)
        & (uv[:, 1] >= -margin_pixels)
        & (uv[:, 1] < height + margin_pixels)
    )


def aabb_visible(
    bounds: tuple[Tensor, Tensor],
    transform: Tensor,
    world_to_camera: Tensor,
    intrinsics: Tensor,
    width: int,
    height: int,
    *,
    near: float = 0.01,
    far: float = 500.0,
) -> bool:
    """Bounding-box-level conservative frustum test for one canonical asset."""

    minimum, maximum = bounds
    selector = torch.tensor(
        ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)),
        dtype=torch.bool,
        device=minimum.device,
    )
    corners = torch.where(selector, maximum, minimum)
    corners = corners @ transform[:3, :3].transpose(0, 1) + transform[:3, 3]
    camera = corners @ world_to_camera[:3, :3].transpose(0, 1) + world_to_camera[:3, 3]
    # If all corners lie behind one clipping plane, the box is invisible.  A
    # projected-corner test alone can reject a box enclosing the camera, so that
    # case is explicitly retained.
    if camera[:, 2].max() < near or camera[:, 2].min() > far:
        return False
    if camera[:, 2].min() <= near:
        return True
    projected = camera @ intrinsics.transpose(0, 1)
    uv = projected[:, :2] / projected[:, 2:3]
    if uv[:, 0].max() < 0 or uv[:, 0].min() >= width:
        return False
    return not (uv[:, 1].max() < 0 or uv[:, 1].min() >= height)
