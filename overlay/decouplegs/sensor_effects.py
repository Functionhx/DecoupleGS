from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class SensorLightingConfig:
    kind: str = "identity"
    exposure: float = 1.0
    gamma: float = 1.0
    glare_strength: float = 0.0
    vignette_strength: float = 0.0

    def __post_init__(self) -> None:
        if self.exposure < 0 or self.gamma <= 0:
            raise ValueError("exposure must be non-negative and gamma must be positive")
        if not 0 <= self.glare_strength <= 1 or not 0 <= self.vignette_strength <= 1:
            raise ValueError("glare/vignette strengths must lie in [0, 1]")


@lru_cache(maxsize=32)
def _radial_fields(height: int, width: int, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    x /= max(width - 1, 1)
    y /= max(height - 1, 1)
    if camera_name == "CAM_FRONT_LEFT":
        center_x = 0.72
    elif camera_name == "CAM_FRONT_RIGHT":
        center_x = 0.28
    else:
        center_x = 0.5
    glare = np.exp(
        -0.5 * (((x - center_x) / 0.16) ** 2 + ((y - 0.28) / 0.12) ** 2)
    ).astype(np.float32)
    radial = np.sqrt(((x - 0.5) / 0.72) ** 2 + ((y - 0.5) / 0.72) ** 2)
    vignette = np.clip(1.0 - radial, 0.0, 1.0).astype(np.float32)
    return glare, vignette


def apply_sensor_lighting(
    image: np.ndarray,
    camera_name: str,
    config: SensorLightingConfig,
) -> np.ndarray:
    """Deterministic H-SCENARIO-01 exposure/glare stress transform.

    Natural night/dusk sequences used by the paper are not identified. This
    image-space fallback is intentionally isolated and labelled so it cannot be
    confused with a claimed reconstruction or proxy-relighting result.
    """

    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must be uint8 HxWx3")
    identity = (
        config.exposure == 1.0
        and config.gamma == 1.0
        and config.glare_strength == 0.0
        and config.vignette_strength == 0.0
    )
    if identity:
        return image
    output = image.astype(np.float32) / 255.0
    output = np.clip(output * config.exposure, 0.0, 1.0)
    output = np.power(output, 1.0 / config.gamma)
    glare, vignette = _radial_fields(image.shape[0], image.shape[1], camera_name)
    if config.vignette_strength:
        gain = (1.0 - config.vignette_strength) + config.vignette_strength * vignette
        output *= gain[..., None]
    if config.glare_strength and camera_name.startswith("CAM_FRONT"):
        amount = config.glare_strength * glare[..., None]
        output = output * (1.0 - amount) + amount
    return np.rint(np.clip(output, 0.0, 1.0) * 255.0).astype(np.uint8)
