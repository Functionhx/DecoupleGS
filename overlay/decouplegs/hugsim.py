from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .compression import CompactGaussianAsset
from .rasterizer import (
    CompactRenderInstance,
    StaticBackgroundRasterCache,
    rasterize_compact_scene,
)
from .registration import GroundPlane, ground_asset_from_pose
from .relighting import (
    RelightingCalibration,
    apply_contact_shadows_batched,
    relight_compact_codebook,
    sample_local_probe,
)
from .spatial import BackgroundSpatialIndex
from .transforms import aabb_visible
from .runtime import (
    AgentInstance,
    CameraFrustum,
    ComposedScene,
    DecoupleRuntime,
)
from .types import GaussianSet


@dataclass
class PreparedCompactScene:
    background: GaussianSet
    instances: list[CompactRenderInstance]
    descriptors: dict[str, Tensor]
    transforms: dict[str, Tensor]
    ground_planes: dict[str, GroundPlane]


def gaussian_set_from_hugsim_model(model: Any, *, include_ground: bool = False) -> GaussianSet:
    """Zero-copy adapter for HUGSIM ``GaussianModel``/``ObjModel`` objects."""

    if include_ground and getattr(model, "ground_model", None) is not None:
        means = model.get_full_xyz
        scales = model.get_full_scaling
        quats = model.get_full_rotation
        opacities = model.get_full_opacity
        sh = model.get_full_features
        semantics = model.get_full_3D_features
    else:
        means = model.get_xyz
        scales = model.get_scaling
        quats = model.get_rotation
        opacities = model.get_opacity
        sh = model.get_features
        semantics = model.get_3D_features
    return GaussianSet(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh=sh,
        semantics=semantics,
        metadata={"source": type(model).__name__},
    )


def gaussian_set_from_hugsim_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> GaussianSet:
    """Load exported ``gs.pth`` without importing HUGSIM CUDA extensions."""

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    parameters = checkpoint[0] if isinstance(checkpoint, (tuple, list)) and len(checkpoint) == 2 else checkpoint
    if len(parameters) < 8:
        raise ValueError(f"unrecognized HUGSIM checkpoint at {path}")
    _, means, features_dc, features_rest, semantics, log_scales, quats, opacity_logits = parameters[:8]
    return GaussianSet(
        means=means.detach(),
        scales=log_scales.detach().exp(),
        quats=torch.nn.functional.normalize(quats.detach(), dim=-1),
        opacities=opacity_logits.detach().sigmoid(),
        sh=torch.cat((features_dc.detach(), features_rest.detach()), dim=1),
        semantics=torch.softmax(semantics.detach(), dim=-1),
        metadata={"source": str(path)},
    )


class HUGSIMRuntimeBridge:
    """Keep HUGSIM's API while replacing its dynamic composition path."""

    def __init__(
        self,
        runtime: DecoupleRuntime | None = None,
        *,
        background_visibility: Tensor | None = None,
        compact_asset_batch_size: int = 8,
        compact_radius_clip: float = 0.0,
        compact_lod_radius_clip: float | None = None,
        compact_lod_start_distance: float = 20.0,
        static_background_cache: bool = True,
        static_background_cache_entries: int = 1,
        static_background_cache_min_observations: int = 2,
        incremental_merge_max_dynamic_ratio: float = 1.25,
    ) -> None:
        self.runtime = DecoupleRuntime() if runtime is None else runtime
        self.background_visibility = background_visibility
        self.compact_asset_batch_size = compact_asset_batch_size
        self.compact_radius_clip = compact_radius_clip
        self.compact_lod_radius_clip = compact_lod_radius_clip
        self.compact_lod_start_distance = compact_lod_start_distance
        self.incremental_merge_max_dynamic_ratio = incremental_merge_max_dynamic_ratio
        self.static_background_cache = (
            StaticBackgroundRasterCache(static_background_cache_entries)
            if static_background_cache
            else None
        )
        self.static_background_cache_min_observations = (
            static_background_cache_min_observations
        )
        self._last_camera_cache_key: Hashable | None = None
        self._camera_cache_observations = 0
        self._model_identity: dict[str, int] = {}
        self._compact_files: dict[str, CompactGaussianAsset] = {}
        self._background_sets: dict[int, GaussianSet] = {}
        self._ground_masks: dict[int, Tensor] = {}
        self._ground_sets: dict[int, GaussianSet] = {}
        self._compact_bounds: dict[tuple[int, int], tuple[Tensor, Tensor]] = {}
        self._spatial_indices: dict[int, BackgroundSpatialIndex] = {}
        self._prepared_key: tuple[Any, ...] | None = None
        self._prepared_compact: PreparedCompactScene | None = None
        if compact_asset_batch_size <= 0:
            raise ValueError("compact_asset_batch_size must be positive")
        if compact_lod_start_distance <= 0:
            raise ValueError("compact_lod_start_distance must be positive")
        if incremental_merge_max_dynamic_ratio < 0:
            raise ValueError("incremental_merge_max_dynamic_ratio must be non-negative")
        if static_background_cache_min_observations <= 0:
            raise ValueError("static_background_cache_min_observations must be positive")

    @staticmethod
    def _camera_tensor_signature(value: Tensor) -> tuple[tuple[int, ...], tuple[float, ...]]:
        # Camera tensors in HUGSIM are short-lived and CUDA's allocator can
        # reuse their addresses on the next frame. A pointer-only key can then
        # return a projection for a different ego pose. Copying 25 scalar
        # values per camera gives an exact, allocation-independent key.
        return tuple(value.shape), tuple(value.detach().reshape(-1).cpu().tolist())

    def _single_camera_cache_key(self, viewpoint: Any) -> tuple[Any, ...]:
        return (
            "single",
            self._camera_tensor_signature(viewpoint.c2w),
            self._camera_tensor_signature(viewpoint.K),
        )

    def _camera_batch_cache_key(self, viewpoints: list[Any]) -> tuple[Any, ...]:
        return (
            "batch",
            tuple(
                (
                    self._camera_tensor_signature(viewpoint.c2w),
                    self._camera_tensor_signature(viewpoint.K),
                )
                for viewpoint in viewpoints
            ),
        )

    def _static_cache_request(
        self,
        camera_key: Hashable,
    ) -> tuple[StaticBackgroundRasterCache | None, Hashable | None]:
        """Admit only camera rigs that repeat instead of churning the LRU.

        Closed-loop ego cameras normally change every state update. Retaining
        each one-entry projection until the next frame would add allocation
        and extraction work without a possible hit. A fixed interactive view
        becomes eligible on its second observation and then remains hot.
        """

        if self.static_background_cache is None:
            return None, None
        if camera_key == self._last_camera_cache_key:
            self._camera_cache_observations += 1
        else:
            if (
                self._camera_cache_observations
                >= self.static_background_cache_min_observations
            ):
                self.static_background_cache.clear()
            self._last_camera_cache_key = camera_key
            self._camera_cache_observations = 1
        if self._camera_cache_observations < self.static_background_cache_min_observations:
            return None, None
        return self.static_background_cache, camera_key

    def _background(self, background_model: Any) -> GaussianSet:
        key = id(background_model)
        cached = self._background_sets.get(key)
        if cached is not None:
            return cached
        background = gaussian_set_from_hugsim_model(background_model, include_ground=True)
        if self.background_visibility is not None:
            visibility = self.background_visibility.to(background.means).reshape(-1)
            if visibility.shape != (len(background),):
                raise ValueError(
                    "background visibility sidecar must contain one value per full background Gaussian "
                    f"({len(background)} expected, {visibility.numel()} found)"
                )
            background.visibility = visibility
        self._background_sets[key] = background
        return background

    def _background_spatial_index(
        self,
        background_model: Any,
        background: GaussianSet,
    ) -> BackgroundSpatialIndex:
        key = id(background_model)
        index = self._spatial_indices.get(key)
        if index is None or len(index) != len(background):
            index = BackgroundSpatialIndex(background.means)
            self._spatial_indices[key] = index
        return index

    def _ground_background(
        self,
        background_model: Any,
        background: GaussianSet,
        ground_mask: Tensor | None,
    ) -> GaussianSet:
        if not self.runtime.semantic_grounding or ground_mask is None:
            return background
        key = id(background_model)
        ground = self._ground_sets.get(key)
        if ground is None:
            ground = background.select(ground_mask)
            self._ground_sets[key] = ground
        return ground

    def register_model(
        self,
        asset_id: str,
        model: Any,
        calibration_path: str | Path | None = None,
    ) -> None:
        identity = id(model)
        if self._model_identity.get(asset_id) == identity:
            return
        calibration = None if calibration_path is None else RelightingCalibration.load(calibration_path)
        self.runtime.library.add(
            asset_id,
            gaussian_set_from_hugsim_model(model),
            calibration,
        )
        self._model_identity[asset_id] = identity

    def register_compact_asset(
        self,
        asset_id: str,
        asset_path: str | Path,
        calibration_path: str | Path | None = None,
    ) -> None:
        calibration = None if calibration_path is None else RelightingCalibration.load(calibration_path)
        resolved = str(Path(asset_path).resolve())
        asset = self._compact_files.get(resolved)
        if asset is None:
            asset = CompactGaussianAsset.load(resolved)
            self._compact_files[resolved] = asset
        self.runtime.library.add(asset_id, asset, calibration)
        self._prepared_key = None
        self._prepared_compact = None

    def _compact_cache_key(
        self,
        viewpoint: Any,
        background_model: Any,
        transforms: dict[str, Tensor],
    ) -> tuple[Any, ...]:
        def tensor_identity(transform: Tensor) -> tuple[int, int | None]:
            try:
                version = transform._version
            except RuntimeError:
                # Tensors created inside torch.inference_mode deliberately do
                # not own a version counter; their allocation identity still
                # distinguishes the immutable per-frame pose objects.
                version = None
            return transform.data_ptr(), version

        transform_identity = tuple(
            (track_id, *tensor_identity(transform))
            for track_id, transform in sorted(transforms.items())
            if track_id in self.runtime.library.assets
        )
        return (
            id(background_model),
            float(viewpoint.timestamp),
            transform_identity,
            # Prepared data includes grounded transforms, relit codebooks,
            # and shadowed background appearance. These switches are usually
            # static in production, but effect ablations intentionally toggle
            # them at a fixed timestamp/pose and must not reuse stale output.
            self.runtime.opacity_grounding,
            self.runtime.adjust_grounding_pose,
            self.runtime.semantic_grounding,
            self.runtime.relighting,
            self.runtime.adaptive_relighting,
            self.runtime.contact_shadows,
        )

    def prepare_compact(
        self,
        viewpoint: Any,
        background_model: Any,
        transforms: dict[str, Tensor],
    ) -> PreparedCompactScene | None:
        active = [
            (track_id, transform)
            for track_id, transform in transforms.items()
            if track_id in self.runtime.library.assets
        ]
        if any(not self.runtime.library.is_compact(track_id) for track_id, _ in active):
            return None
        key = self._compact_cache_key(viewpoint, background_model, transforms)
        if key == self._prepared_key and self._prepared_compact is not None:
            return self._prepared_compact

        background = self._background(background_model)
        ground_mask = self._ground_masks.get(id(background_model))
        if ground_mask is None and background.semantics is not None:
            ground_mask = background.semantics.argmax(dim=-1) <= 1
            self._ground_masks[id(background_model)] = ground_mask
        ground_background = self._ground_background(
            background_model, background, ground_mask
        )
        spatial_index = (
            self._background_spatial_index(background_model, ground_background)
            if active
            and self.runtime.opacity_grounding
            else None
        )
        prepared: list[tuple[str, CompactGaussianAsset, Tensor, GroundPlane | None]] = []
        transforms_out: dict[str, Tensor] = {}
        planes: dict[str, GroundPlane] = {}
        for track_id, initial_transform in active:
            asset = self.runtime.library.get_compact(track_id, background.means)
            transform = initial_transform.to(background.means)
            plane = None
            if self.runtime.opacity_grounding:
                try:
                    grounding = ground_asset_from_pose(
                        asset,  # Compact assets expose the physical-bounds protocol used here.
                        transform,
                        ground_background.means,
                        ground_background.opacities,
                        self.runtime.registration_config,
                        spatial_index=spatial_index,
                    )
                    if self.runtime.adjust_grounding_pose:
                        transform = grounding.transform
                    plane = grounding.plane
                except ValueError:
                    pass
            prepared.append((track_id, asset, transform, plane))
            transforms_out[track_id] = transform
            if plane is not None:
                planes[track_id] = plane

        need_descriptors = self.runtime.contact_shadows or self.runtime.relighting
        descriptors: dict[str, Tensor] = {}
        if prepared and need_descriptors:
            positions = torch.stack([transform[:3, 3] for _, _, transform, _ in prepared])
            sampled = sample_local_probe(
                positions,
                background.means,
                background.sh,
                visibility=(
                    background.visibility
                    if background.visibility is not None
                    else background.opacities
                ),
                config=self.runtime.relighting_config,
            )
            descriptors = {
                track_id: sampled[index]
                for index, (track_id, _, _, _) in enumerate(prepared)
            }

        compact_instances: list[CompactRenderInstance] = []
        shadow_assets: list[CompactGaussianAsset] = []
        shadow_transforms: list[Tensor] = []
        shadow_planes: list[GroundPlane] = []
        shadow_descriptors: list[Tensor] = []
        for track_id, asset, transform, plane in prepared:
            descriptor = descriptors.get(track_id)
            rendered_asset = asset
            if self.runtime.relighting and descriptor is not None:
                calibration = self.runtime.library.calibrations.get(track_id)
                if calibration is not None:
                    rendered_asset = relight_compact_codebook(
                        asset,
                        descriptor,
                        calibration=calibration,
                    )
                elif self.runtime.adaptive_relighting:
                    rendered_asset = relight_compact_codebook(
                        asset,
                        descriptor,
                        config=self.runtime.relighting_config,
                    )
            compact_instances.append(CompactRenderInstance(track_id, rendered_asset, transform))
            if (
                self.runtime.contact_shadows
                and descriptor is not None
                and plane is not None
            ):
                shadow_assets.append(asset)
                shadow_transforms.append(transform)
                shadow_planes.append(plane)
                shadow_descriptors.append(descriptor)
        shaded_background = apply_contact_shadows_batched(
            background,
            shadow_assets,
            shadow_transforms,
            shadow_planes,
            shadow_descriptors,
            relighting_config=self.runtime.relighting_config,
            registration_config=self.runtime.registration_config,
            ground_mask=ground_mask,
        )
        result = PreparedCompactScene(
            background=shaded_background,
            instances=compact_instances,
            descriptors=descriptors,
            transforms=transforms_out,
            ground_planes=planes,
        )
        self._prepared_key = key
        self._prepared_compact = result
        return result

    def render_compact(
        self,
        viewpoint: Any,
        background_model: Any,
        transforms: dict[str, Tensor],
        background_color: Tensor,
        *,
        auxiliary: bool,
    ) -> dict[str, Any] | None:
        prepared = self.prepare_compact(viewpoint, background_model, transforms)
        if prepared is None:
            return None
        camera_position = viewpoint.c2w[:3, 3].to(prepared.background.means)
        instances = prepared.instances
        if self.runtime.frustum_culling:
            world_to_camera = torch.linalg.inv(viewpoint.c2w).to(
                prepared.background.means
            )
            intrinsics = viewpoint.K[:3, :3].to(prepared.background.means)
            visible_instances = []
            for instance in instances:
                bounds_key = (
                    instance.asset.means.data_ptr(),
                    instance.asset.shape_codebook.data_ptr(),
                )
                bounds = self._compact_bounds.get(bounds_key)
                if bounds is None:
                    bounds = instance.asset.bounds
                    self._compact_bounds[bounds_key] = bounds
                if aabb_visible(
                    bounds,
                    instance.transform,
                    world_to_camera,
                    intrinsics,
                    viewpoint.width,
                    viewpoint.height,
                ):
                    visible_instances.append(instance)
            instances = visible_instances
        if self.compact_lod_radius_clip is not None:
            instances = [
                replace(
                    instance,
                    radius_clip=(
                        self.compact_lod_radius_clip
                        if float(
                            torch.linalg.vector_norm(
                                instance.transform[:3, 3] - camera_position
                            ).item()
                        )
                        >= self.compact_lod_start_distance
                        else None
                    ),
                )
                for instance in instances
            ]
        if self.static_background_cache is None:
            static_cache, static_cache_key = None, None
        else:
            static_cache, static_cache_key = self._static_cache_request(
                self._single_camera_cache_key(viewpoint)
            )
        rasterized = rasterize_compact_scene(
            prepared.background,
            instances,
            torch.linalg.inv(viewpoint.c2w)[None],
            viewpoint.K[None, :3, :3],
            viewpoint.width,
            viewpoint.height,
            sh_degree=background_model.active_sh_degree,
            backgrounds=background_color[None],
            auxiliary=auxiliary,
            radius_clip=self.compact_radius_clip,
            asset_batch_size=self.compact_asset_batch_size,
            rotate_sh=self.runtime.rotate_sh,
            background_cache=static_cache,
            background_cache_key=static_cache_key,
            incremental_merge_max_dynamic_ratio=self.incremental_merge_max_dynamic_ratio,
        )
        rasterized.info["prepared_asset_instances"] = len(prepared.instances)
        rasterized.info["visible_asset_instances"] = len(instances)
        return {
            "render": rasterized.render[0].permute(2, 0, 1),
            "alphas": rasterized.alpha,
            "depth": (
                None
                if rasterized.depth is None
                else rasterized.depth[0].permute(2, 0, 1)
            ),
            "feats": (
                None
                if rasterized.semantics is None
                else rasterized.semantics[0].permute(2, 0, 1)
            ),
            "viewspace_points": rasterized.info["means2d"],
            "info": rasterized.info,
            "prepared": prepared,
        }

    def render_compact_batch(
        self,
        viewpoints: list[Any],
        background_model: Any,
        transforms: dict[str, Tensor],
        background_color: Tensor,
        *,
        auxiliary: bool,
    ) -> dict[str, Any] | None:
        """Rasterize synchronized equal-resolution cameras in one dispatch.

        HUGSIM's six nuScenes cameras observe one shared world state.  Batching
        them retains exact per-camera projection/sorting while reusing scene
        preparation and avoiding six Python/CUDA launch pipelines.
        """

        if not viewpoints:
            raise ValueError("at least one viewpoint is required")
        width, height = viewpoints[0].width, viewpoints[0].height
        if any(view.width != width or view.height != height for view in viewpoints):
            raise ValueError("batched compact rendering requires equal camera resolutions")
        prepared = self.prepare_compact(viewpoints[0], background_model, transforms)
        if prepared is None:
            return None
        viewmats = torch.stack(
            [torch.linalg.inv(view.c2w).to(prepared.background.means) for view in viewpoints]
        )
        intrinsics = torch.stack(
            [view.K[:3, :3].to(prepared.background.means) for view in viewpoints]
        )
        instances = prepared.instances
        if self.runtime.frustum_culling:
            union_visible = []
            for instance in instances:
                bounds_key = (
                    instance.asset.means.data_ptr(),
                    instance.asset.shape_codebook.data_ptr(),
                )
                bounds = self._compact_bounds.get(bounds_key)
                if bounds is None:
                    bounds = instance.asset.bounds
                    self._compact_bounds[bounds_key] = bounds
                if any(
                    aabb_visible(
                        bounds,
                        instance.transform,
                        viewmat,
                        intrinsic,
                        width,
                        height,
                    )
                    for viewmat, intrinsic in zip(viewmats, intrinsics, strict=True)
                ):
                    union_visible.append(instance)
            instances = union_visible
        backgrounds = background_color.to(prepared.background.means)[None].expand(
            len(viewpoints), -1
        )
        if self.static_background_cache is None:
            static_cache, static_cache_key = None, None
        else:
            static_cache, static_cache_key = self._static_cache_request(
                self._camera_batch_cache_key(viewpoints)
            )
        rasterized = rasterize_compact_scene(
            prepared.background,
            instances,
            viewmats,
            intrinsics,
            width,
            height,
            sh_degree=background_model.active_sh_degree,
            backgrounds=backgrounds,
            auxiliary=auxiliary,
            radius_clip=self.compact_radius_clip,
            asset_batch_size=self.compact_asset_batch_size,
            rotate_sh=self.runtime.rotate_sh,
            background_cache=static_cache,
            background_cache_key=static_cache_key,
            incremental_merge_max_dynamic_ratio=self.incremental_merge_max_dynamic_ratio,
        )
        rasterized.info["prepared_asset_instances"] = len(prepared.instances)
        rasterized.info["visible_asset_instances"] = len(instances)
        return {
            "render": rasterized.render.permute(0, 3, 1, 2),
            "alphas": rasterized.alpha,
            "depth": (
                None
                if rasterized.depth is None
                else rasterized.depth.permute(0, 3, 1, 2)
            ),
            "feats": (
                None
                if rasterized.semantics is None
                else rasterized.semantics.permute(0, 3, 1, 2)
            ),
            "info": rasterized.info,
            "prepared": prepared,
        }

    def compose(
        self,
        viewpoint: Any,
        background_model: Any,
        dynamic_models: dict[str, Any],
        transforms: dict[str, Tensor],
    ) -> ComposedScene:
        for asset_id, model in dynamic_models.items():
            if asset_id not in self.runtime.library.assets:
                self.register_model(asset_id, model)
        background = self._background(background_model)
        camera = CameraFrustum(
            world_to_camera=torch.linalg.inv(viewpoint.c2w),
            intrinsics=viewpoint.K[:3, :3],
            width=viewpoint.width,
            height=viewpoint.height,
        )
        instances = [
            AgentInstance(instance_id=track_id, asset_id=track_id, transform=transform)
            for track_id, transform in transforms.items()
            if track_id in self.runtime.library.assets
        ]
        ground_mask = None
        if background.semantics is not None:
            ground_mask = background.semantics.argmax(dim=-1) <= 1
        return self.runtime.compose(background, instances, camera=camera, ground_mask=ground_mask)
