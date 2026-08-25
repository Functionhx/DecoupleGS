from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch import Tensor


def _tensor_bytes(value: Tensor | None) -> int:
    if value is None:
        return 0
    return value.numel() * value.element_size()


@dataclass
class GaussianSet:
    """Renderer-ready 3D Gaussian primitives.

    Quaternions use the HUGSIM/GraphDeco ``(w, x, y, z)`` convention.  Scales
    and opacities are activated values (positive scales and probabilities), not
    their log/logit training parameters.  SH coefficients follow gsplat's
    ``[N, (degree + 1)^2, 3]`` layout.
    """

    means: Tensor
    scales: Tensor
    quats: Tensor
    opacities: Tensor
    sh: Tensor
    semantics: Tensor | None = None
    visibility: Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.opacities.ndim == 2 and self.opacities.shape[1] == 1:
            self.opacities = self.opacities[:, 0]
        n = self.means.shape[0]
        expected = {
            "means": (n, 3),
            "scales": (n, 3),
            "quats": (n, 4),
            "opacities": (n,),
        }
        tensors = {
            "means": self.means,
            "scales": self.scales,
            "quats": self.quats,
            "opacities": self.opacities,
        }
        for name, shape in expected.items():
            if tuple(tensors[name].shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(tensors[name].shape)}")
        if self.sh.ndim != 3 or self.sh.shape[0] != n or self.sh.shape[2] != 3:
            raise ValueError(f"sh must have shape [N, C, 3], got {tuple(self.sh.shape)}")
        coeffs = self.sh.shape[1]
        degree = int(coeffs**0.5) - 1
        if (degree + 1) ** 2 != coeffs:
            raise ValueError(f"SH coefficient count {coeffs} is not a squared band count")
        if self.semantics is not None and self.semantics.shape[0] != n:
            raise ValueError("semantics and means must have the same first dimension")
        if self.visibility is not None and self.visibility.shape[0] != n:
            raise ValueError("visibility and means must have the same first dimension")
        devices = {value.device for value in tensors.values()}
        devices.add(self.sh.device)
        if self.semantics is not None:
            devices.add(self.semantics.device)
        if self.visibility is not None:
            devices.add(self.visibility.device)
        if len(devices) != 1:
            raise ValueError(f"all Gaussian tensors must share a device, got {devices}")

    def __len__(self) -> int:
        return self.means.shape[0]

    @property
    def device(self) -> torch.device:
        return self.means.device

    @property
    def dtype(self) -> torch.dtype:
        return self.means.dtype

    @property
    def sh_degree(self) -> int:
        return int(self.sh.shape[1] ** 0.5) - 1

    @property
    def memory_bytes(self) -> int:
        return sum(
            _tensor_bytes(value)
            for value in (
                self.means,
                self.scales,
                self.quats,
                self.opacities,
                self.sh,
                self.semantics,
                self.visibility,
            )
        )

    @property
    def bounds(self) -> tuple[Tensor, Tensor]:
        """Conservative 3-sigma bounds used for render-time culling."""

        if len(self) == 0:
            empty = torch.zeros(3, dtype=self.dtype, device=self.device)
            return empty, empty
        radius = self.scales.max(dim=-1).values[:, None] * 3.0
        return (self.means - radius).amin(dim=0), (self.means + radius).amax(dim=0)

    @property
    def physical_bounds(self) -> tuple[Tensor, Tensor]:
        """Canonical object bounds used for grounding and contact shadows.

        Gaussian covariance support is deliberately excluded: a large splat is
        useful for conservative frustum culling but must not move the physical
        bottom of a vehicle. Importers may provide an authoritative
        ``metadata["physical_bounds"]`` pair; otherwise center extrema are the
        least-assumptive fallback.
        """

        if len(self) == 0:
            empty = torch.zeros(3, dtype=self.dtype, device=self.device)
            return empty, empty
        value = self.metadata.get("physical_bounds")
        if value is None:
            return self.means.amin(dim=0), self.means.amax(dim=0)
        if not isinstance(value, (list, tuple, Tensor)) or len(value) != 2:
            raise ValueError("metadata['physical_bounds'] must have shape [2, 3]")
        if isinstance(value, Tensor):
            bounds = value.to(device=self.device, dtype=self.dtype)
        else:
            bounds = torch.stack(
                [torch.as_tensor(bound, device=self.device, dtype=self.dtype) for bound in value]
            )
        if bounds.shape != (2, 3) or bool((bounds[0] > bounds[1]).any()):
            raise ValueError("metadata['physical_bounds'] must contain ordered [min, max] 3D bounds")
        return bounds[0], bounds[1]

    def select(self, index: Tensor | slice) -> GaussianSet:
        return GaussianSet(
            means=self.means[index],
            scales=self.scales[index],
            quats=self.quats[index],
            opacities=self.opacities[index],
            sh=self.sh[index],
            semantics=None if self.semantics is None else self.semantics[index],
            visibility=None if self.visibility is None else self.visibility[index],
            metadata=dict(self.metadata),
        )

    def detach(self) -> GaussianSet:
        return replace(
            self,
            means=self.means.detach(),
            scales=self.scales.detach(),
            quats=self.quats.detach(),
            opacities=self.opacities.detach(),
            sh=self.sh.detach(),
            semantics=None if self.semantics is None else self.semantics.detach(),
            visibility=None if self.visibility is None else self.visibility.detach(),
            metadata=dict(self.metadata),
        )

    def to(self, *args: Any, **kwargs: Any) -> GaussianSet:
        def move(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.to(*args, **kwargs)

        return GaussianSet(
            means=move(self.means),
            scales=move(self.scales),
            quats=move(self.quats),
            opacities=move(self.opacities),
            sh=move(self.sh),
            semantics=move(self.semantics),
            visibility=move(self.visibility),
            metadata=dict(self.metadata),
        )

    @classmethod
    def concatenate(cls, sets: Iterable[GaussianSet]) -> GaussianSet:
        items = list(sets)
        if not items:
            raise ValueError("at least one GaussianSet is required")
        degrees = {item.sh_degree for item in items}
        if len(degrees) != 1:
            raise ValueError(f"all sets must use the same SH degree, got {degrees}")
        semantic_dims = {None if item.semantics is None else item.semantics.shape[1:] for item in items}
        if len(semantic_dims) != 1:
            raise ValueError("all sets must use matching semantic features")
        visibility_dims = {item.visibility.shape[1:] for item in items if item.visibility is not None}
        if len(visibility_dims) > 1:
            raise ValueError("all sets must use matching visibility features")
        retain_visibility = all(item.visibility is not None for item in items)

        def cat(name: str) -> Tensor | None:
            values = [getattr(item, name) for item in items]
            if values[0] is None:
                return None
            return torch.cat(values, dim=0)

        return cls(
            means=cat("means"),
            scales=cat("scales"),
            quats=cat("quats"),
            opacities=cat("opacities"),
            sh=cat("sh"),
            semantics=cat("semantics"),
            visibility=cat("visibility") if retain_visibility else None,
            metadata={"parts": [dict(item.metadata) for item in items]},
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "means": self.means,
            "scales": self.scales,
            "quats": self.quats,
            "opacities": self.opacities,
            "sh": self.sh,
            "semantics": self.semantics,
            "visibility": self.visibility,
            "metadata": self.metadata,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> GaussianSet:
        return cls(**dict(state))
