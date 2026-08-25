from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from .compression import CompactGaussianAsset
from .registration import GroundPlane, RegistrationConfig, ground_asset_from_pose
from .relighting import (
    RelightingCalibration,
    RelightingConfig,
    adaptive_relight_asset,
    apply_contact_shadows_batched,
    relight_asset,
    sample_local_probe,
)
from .spatial import BackgroundSpatialIndex
from .transforms import RealSHRotator, aabb_visible, transform_gaussians
from .types import GaussianSet


@dataclass(frozen=True)
class CameraFrustum:
    world_to_camera: Tensor
    intrinsics: Tensor
    width: int
    height: int
    near: float = 0.01
    far: float = 500.0


@dataclass(frozen=True)
class AgentInstance:
    instance_id: str
    asset_id: str
    transform: Tensor
    ground_plane: GroundPlane | None = None
    enabled: bool = True


@dataclass
class ComposedScene:
    background: GaussianSet
    assets: dict[str, GaussianSet]
    merged: GaussianSet
    descriptors: dict[str, Tensor] = field(default_factory=dict)
    transforms: dict[str, Tensor] = field(default_factory=dict)
    ground_planes: dict[str, GroundPlane] = field(default_factory=dict)


class AssetLibrary:
    def __init__(self) -> None:
        self.assets: dict[str, GaussianSet | CompactGaussianAsset] = {}
        self.calibrations: dict[str, RelightingCalibration] = {}
        self._decoded: dict[tuple[str, str, torch.dtype], GaussianSet] = {}
        self._compact_moved: dict[tuple[int, str, torch.dtype], CompactGaussianAsset] = {}

    def add(
        self,
        asset_id: str,
        asset: GaussianSet | CompactGaussianAsset,
        calibration: RelightingCalibration | None = None,
    ) -> None:
        self.assets[asset_id] = asset
        if calibration is not None:
            self.calibrations[asset_id] = calibration
        for key in [key for key in self._decoded if key[0] == asset_id]:
            del self._decoded[key]
        self._compact_moved.clear()

    def is_compact(self, asset_id: str) -> bool:
        return isinstance(self.assets.get(asset_id), CompactGaussianAsset)

    def get_compact(self, asset_id: str, reference: Tensor) -> CompactGaussianAsset:
        if asset_id not in self.assets:
            raise KeyError(f"unknown DecoupleGS asset: {asset_id}")
        asset = self.assets[asset_id]
        if not isinstance(asset, CompactGaussianAsset):
            raise TypeError(f"asset {asset_id!r} is not compact")
        key = (id(asset), str(reference.device), reference.dtype)
        cached = self._compact_moved.get(key)
        if cached is None:
            cached = asset.to(reference.device, dtype=reference.dtype)
            self._compact_moved[key] = cached
        return cached

    def get(self, asset_id: str, reference: Tensor) -> GaussianSet:
        if asset_id not in self.assets:
            raise KeyError(f"unknown DecoupleGS asset: {asset_id}")
        key = (asset_id, str(reference.device), reference.dtype)
        cached = self._decoded.get(key)
        if cached is not None:
            return cached
        asset = self.assets[asset_id]
        if isinstance(asset, CompactGaussianAsset):
            decoded = asset.decode(device=reference.device, dtype=reference.dtype)
        else:
            decoded = asset.to(device=reference.device, dtype=reference.dtype)
        self._decoded[key] = decoded
        return decoded


class DecoupleRuntime:
    """Online Algorithm 2: register, relight, transform, merge, rasterize once."""

    def __init__(
        self,
        library: AssetLibrary | None = None,
        *,
        registration_config: RegistrationConfig | None = None,
        relighting_config: RelightingConfig | None = None,
        rotate_sh: bool = True,
        relighting: bool = True,
        adaptive_relighting: bool = False,
        contact_shadows: bool = True,
        opacity_grounding: bool = True,
        adjust_grounding_pose: bool = True,
        semantic_grounding: bool = True,
        frustum_culling: bool = True,
    ) -> None:
        self.library = AssetLibrary() if library is None else library
        self.registration_config = RegistrationConfig() if registration_config is None else registration_config
        self.relighting_config = RelightingConfig() if relighting_config is None else relighting_config
        self.rotate_sh = rotate_sh
        self.relighting = relighting
        self.adaptive_relighting = adaptive_relighting
        self.contact_shadows = contact_shadows
        self.opacity_grounding = opacity_grounding
        self.adjust_grounding_pose = adjust_grounding_pose
        self.semantic_grounding = semantic_grounding
        self.frustum_culling = frustum_culling
        self.sh_rotator = RealSHRotator()
        self._spatial_index_key: tuple[int, int, int] | None = None
        self._spatial_index: BackgroundSpatialIndex | None = None

    def _background_spatial_index(self, background: GaussianSet) -> BackgroundSpatialIndex:
        try:
            version = background.means._version
        except RuntimeError:
            # Tensors materialized inside inference_mode deliberately omit a
            # version counter. Their storage identity is still stable for the
            # lifetime of one composed background, so use an explicit sentinel.
            version = -1
        key = (background.means.data_ptr(), version, len(background))
        if self._spatial_index_key != key or self._spatial_index is None:
            self._spatial_index = BackgroundSpatialIndex(background.means)
            self._spatial_index_key = key
        return self._spatial_index

    def compose(
        self,
        background: GaussianSet,
        instances: list[AgentInstance],
        *,
        camera: CameraFrustum | None = None,
        ground_mask: Tensor | None = None,
    ) -> ComposedScene:
        probe_background = background
        shaded_background = background
        world_assets: dict[str, GaussianSet] = {}
        descriptors: dict[str, Tensor] = {}
        transforms: dict[str, Tensor] = {}
        planes: dict[str, GroundPlane] = {}
        local_assets: dict[str, GaussianSet] = {}
        ground_background = (
            background.select(ground_mask)
            if self.semantic_grounding and ground_mask is not None
            else background
        )
        spatial_index = (
            self._background_spatial_index(ground_background)
            if instances
            and self.opacity_grounding
            else None
        )

        for instance in instances:
            if not instance.enabled:
                continue
            asset = self.library.get(instance.asset_id, background.means)
            transform = instance.transform.to(background.means)
            if camera is not None and self.frustum_culling:
                visible = aabb_visible(
                    asset.bounds,
                    transform,
                    camera.world_to_camera.to(background.means),
                    camera.intrinsics.to(background.means),
                    camera.width,
                    camera.height,
                    near=camera.near,
                    far=camera.far,
                )
                if not visible:
                    continue
            plane = instance.ground_plane
            if self.opacity_grounding and plane is None:
                try:
                    grounding = ground_asset_from_pose(
                        asset,
                        transform,
                        ground_background.means,
                        ground_background.opacities,
                        self.registration_config,
                        spatial_index=spatial_index,
                    )
                    if self.adjust_grounding_pose:
                        transform = grounding.transform
                    plane = grounding.plane
                except ValueError:
                    # Sparse/absent ground samples keep the scenario-provided pose.
                    plane = None
            calibration = self.library.calibrations.get(instance.asset_id)
            descriptor = None
            if self.contact_shadows or (
                self.relighting and (calibration is not None or self.adaptive_relighting)
            ):
                descriptor = sample_local_probe(
                    transform[:3, 3],
                    probe_background.means,
                    probe_background.sh,
                    visibility=probe_background.visibility
                    if probe_background.visibility is not None
                    else probe_background.opacities,
                    config=self.relighting_config,
                )
            local_asset = asset
            if self.relighting and calibration is not None:
                assert descriptor is not None
                local_asset = relight_asset(local_asset, descriptor, calibration)
            elif self.relighting and self.adaptive_relighting:
                assert descriptor is not None
                local_asset = adaptive_relight_asset(
                    local_asset,
                    descriptor,
                    self.relighting_config,
                )
            world_asset = transform_gaussians(
                local_asset,
                transform,
                rotate_sh=self.rotate_sh,
                sh_rotator=self.sh_rotator,
            )
            local_assets[instance.instance_id] = asset
            world_assets[instance.instance_id] = world_asset
            if descriptor is not None:
                descriptors[instance.instance_id] = descriptor
            transforms[instance.instance_id] = transform
            if plane is not None:
                planes[instance.instance_id] = plane

        if self.contact_shadows:
            shadow_ids = [
                instance_id
                for instance_id in world_assets
                if instance_id in planes and instance_id in descriptors
            ]
            shaded_background = apply_contact_shadows_batched(
                shaded_background,
                [local_assets[instance_id] for instance_id in shadow_ids],
                [transforms[instance_id] for instance_id in shadow_ids],
                [planes[instance_id] for instance_id in shadow_ids],
                [descriptors[instance_id] for instance_id in shadow_ids],
                relighting_config=self.relighting_config,
                registration_config=self.registration_config,
                ground_mask=ground_mask,
            )
        parts = [shaded_background, *world_assets.values()]
        merged = GaussianSet.concatenate(parts) if len(parts) > 1 else shaded_background
        return ComposedScene(
            background=shaded_background,
            assets=world_assets,
            merged=merged,
            descriptors=descriptors,
            transforms=transforms,
            ground_planes=planes,
        )
