from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Hashable
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .compression import CompactGaussianAsset
from .kernels import (
    indexed_spherical_harmonics,
    merge_sorted_index_streams,
    sorted_index_merge_backend,
)
from .transforms import RealSHRotator, pack_symmetric, unpack_symmetric
from .types import GaussianSet

_SH_ROTATOR = RealSHRotator()


@dataclass(frozen=True)
class CompactRenderInstance:
    instance_id: str
    asset: CompactGaussianAsset
    transform: Tensor
    radius_clip: float | None = None


@dataclass
class CompactRasterizationResult:
    render: Tensor
    alpha: Tensor
    depth: Tensor | None = None
    semantics: Tensor | None = None
    info: dict[str, Tensor | int | float | bool | str] = field(default_factory=dict)


@dataclass
class StaticBackgroundRaster:
    """Camera-space background data that is invariant while agents move.

    Appearance is deliberately not stored here. Contact shadows can update the
    background SH coefficients every frame without changing projection, tile
    coverage, or depth order, so RGB/auxiliary features are gathered from the
    current :class:`GaussianSet` after a cache lookup.
    """

    camera_ids: Tensor
    gaussian_ids: Tensor
    radii: Tensor
    means2d: Tensor
    depths: Tensor
    conics: Tensor
    tiles_per_gaussian: Tensor
    intersection_ids: Tensor
    flatten_ids: Tensor
    offsets: Tensor

    @property
    def visible_gaussians(self) -> int:
        return int(self.means2d.shape[0])

    @property
    def intersections(self) -> int:
        return int(self.flatten_ids.shape[0])

    @property
    def memory_bytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in (
                self.camera_ids,
                self.gaussian_ids,
                self.radii,
                self.means2d,
                self.depths,
                self.conics,
                self.tiles_per_gaussian,
                self.intersection_ids,
                self.flatten_ids,
                self.offsets,
            )
        )


class StaticBackgroundRasterCache:
    """Small LRU for exact static-background projection and radix results.

    One full driving-scene entry can occupy hundreds of MiB. The default keeps
    only the current single- or multi-camera rig, which is enough for repeated
    renders with moving agents and avoids silently consuming several GiB.
    """

    def __init__(self, max_entries: int = 1) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._entries: OrderedDict[Hashable, StaticBackgroundRaster] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: Hashable) -> StaticBackgroundRaster | None:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return entry

    def put(self, key: Hashable, entry: StaticBackgroundRaster) -> None:
        previous = self._entries.pop(key, None)
        self._entries[key] = entry
        if previous is None and len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self.evictions += 1

    def clear(self) -> None:
        self._entries.clear()

    @property
    def entries(self) -> int:
        return len(self._entries)

    @property
    def memory_bytes(self) -> int:
        return sum(entry.memory_bytes for entry in self._entries.values())


def _tensor_identity(
    value: Tensor,
) -> tuple[int, int | None, tuple[int, ...], torch.dtype, torch.device]:
    try:
        version = value._version
    except RuntimeError:
        # inference-mode tensors intentionally have no version counter
        version = None
    return value.data_ptr(), version, tuple(value.shape), value.dtype, value.device


def _project(
    means: Tensor,
    viewmats: Tensor,
    intrinsics: Tensor,
    width: int,
    height: int,
    *,
    quats: Tensor | None = None,
    scales: Tensor | None = None,
    covariances: Tensor | None = None,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
) -> tuple[Tensor, ...]:
    from gsplat.cuda._wrapper import fully_fused_projection

    return fully_fused_projection(
        means,
        covariances,
        quats,
        scales,
        viewmats,
        intrinsics,
        width,
        height,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        packed=True,
    )


def _project_background(
    background: GaussianSet,
    viewmats: Tensor,
    intrinsics: Tensor,
    width: int,
    height: int,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    camera_ids, gaussian_ids, radii, means2d, depths, conics, _ = _project(
        background.means,
        viewmats,
        intrinsics,
        width,
        height,
        quats=background.quats,
        scales=background.scales,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
    )
    return camera_ids, gaussian_ids, radii, means2d, depths, conics


def _background_part_from_projection(
    background: GaussianSet,
    projection: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    sh_degree: int,
    auxiliary: bool,
    camera_centers: Tensor,
) -> tuple[Tensor, ...]:
    from gsplat.cuda._wrapper import spherical_harmonics

    camera_ids, gaussian_ids, radii, means2d, depths, conics = projection
    directions = background.means[gaussian_ids] - camera_centers[camera_ids]
    rgb = spherical_harmonics(sh_degree, directions, background.sh[gaussian_ids])
    rgb = torch.clamp_min(rgb + 0.5, 0.0)
    features = rgb
    if auxiliary:
        if background.semantics is None:
            raise ValueError("auxiliary rendering requires background semantics")
        features = torch.cat((rgb, depths[:, None], background.semantics[gaussian_ids]), dim=-1)
    return camera_ids, radii, means2d, depths, conics, background.opacities[gaussian_ids], features


def _background_part(
    background: GaussianSet,
    viewmats: Tensor,
    intrinsics: Tensor,
    width: int,
    height: int,
    sh_degree: int,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    auxiliary: bool,
    camera_centers: Tensor,
) -> tuple[Tensor, ...]:
    projection = _project_background(
        background,
        viewmats,
        intrinsics,
        width,
        height,
        near_plane,
        far_plane,
        radius_clip,
    )
    return _background_part_from_projection(
        background,
        projection,
        sh_degree,
        auxiliary,
        camera_centers,
    )


def _compact_part(
    instance: CompactRenderInstance,
    viewmats: Tensor,
    intrinsics: Tensor,
    width: int,
    height: int,
    sh_degree: int,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    auxiliary: bool,
    sh_rotator: RealSHRotator,
    rotate_sh: bool,
    camera_centers: Tensor,
) -> tuple[Tensor, ...]:
    asset = instance.asset
    transform = instance.transform.to(device=asset.means.device, dtype=asset.means.dtype)
    rotation, translation = transform[:3, :3], transform[:3, 3]
    means = asset.means @ rotation.transpose(0, 1) + translation

    canonical_covariance = unpack_symmetric(asset.shape_codebook)
    rotated_covariance = rotation @ canonical_covariance @ rotation.transpose(0, 1)
    covariance_codebook = pack_symmetric(rotated_covariance)
    covariances = covariance_codebook[asset.shape_indices.to(torch.int64)]
    camera_ids, gaussian_ids, radii, means2d, depths, conics, _ = _project(
        means,
        viewmats,
        intrinsics,
        width,
        height,
        covariances=covariances,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip if instance.radius_clip is None else instance.radius_clip,
    )
    directions = means[gaussian_ids] - camera_centers[camera_ids]
    color_codebook = asset.color_codebook.reshape(-1, (asset.sh_degree + 1) ** 2, 3)
    rotate_asset_sh = rotate_sh and asset.metadata.get("sh_frame", "local") == "local"
    if rotate_asset_sh and asset.sh_degree > 0:
        color_codebook = sh_rotator(color_codebook, rotation)
    rgb = indexed_spherical_harmonics(
        directions,
        color_codebook,
        asset.color_indices[gaussian_ids],
        sh_degree,
    )
    rgb = torch.clamp_min(rgb + 0.5, 0.0)
    features = rgb
    if auxiliary:
        if asset.semantics is None:
            raise ValueError("auxiliary rendering requires compact-asset semantics")
        if asset.semantics.shape[0] == 1:
            semantics = asset.semantics.expand(gaussian_ids.shape[0], -1)
        else:
            semantics = asset.semantics[gaussian_ids]
        features = torch.cat((rgb, depths[:, None], semantics), dim=-1)
    return camera_ids, radii, means2d, depths, conics, asset.opacities[gaussian_ids], features


def _compact_batch_part(
    instances: list[CompactRenderInstance],
    viewmats: Tensor,
    intrinsics: Tensor,
    width: int,
    height: int,
    sh_degree: int,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    auxiliary: bool,
    sh_rotator: RealSHRotator,
    rotate_sh: bool,
    camera_centers: Tensor,
) -> tuple[Tensor, ...]:
    """Project several instances of one asset in one CUDA dispatch."""

    if not instances:
        raise ValueError("at least one compact instance is required")
    asset = instances[0].asset
    if any(instance.asset is not asset for instance in instances):
        raise ValueError("batched compact projection requires one shared asset object")
    transforms = torch.stack(
        [instance.transform.to(device=asset.means.device, dtype=asset.means.dtype) for instance in instances]
    )
    rotations = transforms[:, :3, :3]
    translations = transforms[:, :3, 3]
    means = torch.matmul(asset.means[None], rotations.transpose(1, 2)) + translations[:, None]

    canonical_covariance = unpack_symmetric(asset.shape_codebook)
    rotated_covariance = (
        rotations[:, None] @ canonical_covariance[None] @ rotations[:, None].transpose(-1, -2)
    )
    covariance_codebooks = pack_symmetric(rotated_covariance)
    shape_indices = asset.shape_indices.to(torch.int64)
    covariances = covariance_codebooks[:, shape_indices]
    flat_means = means.flatten(0, 1)
    camera_ids, gaussian_ids, radii, means2d, depths, conics, _ = _project(
        flat_means,
        viewmats,
        intrinsics,
        width,
        height,
        covariances=covariances.flatten(0, 1),
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
    )
    primitive_count = len(asset)
    instance_indices = torch.div(gaussian_ids, primitive_count, rounding_mode="floor")
    local_indices = gaussian_ids.remainder(primitive_count)
    directions = flat_means[gaussian_ids] - camera_centers[camera_ids]

    canonical_color_codebook = asset.color_codebook.reshape(
        -1, (asset.sh_degree + 1) ** 2, 3
    )
    rotate_asset_sh = rotate_sh and asset.metadata.get("sh_frame", "local") == "local"
    rotated_color_codebooks = torch.stack(
        [
            sh_rotator(canonical_color_codebook, rotation)
            if rotate_asset_sh and asset.sh_degree > 0
            else canonical_color_codebook
            for rotation in rotations
        ]
    )
    codes_per_instance = rotated_color_codebooks.shape[1]
    global_color_indices = (
        instance_indices * codes_per_instance
        + asset.color_indices[local_indices].to(torch.int64)
    )
    rgb = indexed_spherical_harmonics(
        directions,
        rotated_color_codebooks.flatten(0, 1),
        global_color_indices,
        sh_degree,
    )
    rgb = torch.clamp_min(rgb + 0.5, 0.0)
    features = rgb
    if auxiliary:
        if asset.semantics is None:
            raise ValueError("auxiliary rendering requires compact-asset semantics")
        if asset.semantics.shape[0] == 1:
            semantics = asset.semantics.expand(local_indices.shape[0], -1)
        else:
            semantics = asset.semantics[local_indices]
        features = torch.cat((rgb, depths[:, None], semantics), dim=-1)
    return (
        camera_ids,
        radii,
        means2d,
        depths,
        conics,
        asset.opacities[local_indices],
        features,
    )


def _static_background_key(
    background: GaussianSet,
    viewmats: Tensor,
    intrinsics: Tensor,
    width: int,
    height: int,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    tile_size: int,
    camera_key: Hashable | None,
) -> Hashable:
    # SH, opacity, and semantics do not participate: they affect per-frame
    # features but not projection or tile/depth ordering.
    geometry = (
        _tensor_identity(background.means),
        _tensor_identity(background.scales),
        _tensor_identity(background.quats),
    )
    camera = (
        (_tensor_identity(viewmats), _tensor_identity(intrinsics))
        if camera_key is None
        else camera_key
    )
    key = (
        geometry,
        camera,
        viewmats.shape[0],
        width,
        height,
        float(near_plane),
        float(far_plane),
        float(radius_clip),
        tile_size,
    )
    hash(key)
    return key


def _sort_projected_part(
    camera_ids: Tensor,
    radii: Tensor,
    means2d: Tensor,
    depths: Tensor,
    *,
    n_cameras: int,
    tile_size: int,
    tile_width: int,
    tile_height: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles

    local_ids = torch.arange(means2d.shape[0], dtype=torch.int64, device=means2d.device)
    tiles_per_gaussian, intersection_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=True,
        n_cameras=n_cameras,
        camera_ids=camera_ids,
        gaussian_ids=local_ids,
    )
    offsets = isect_offset_encode(intersection_ids, n_cameras, tile_width, tile_height)
    return tiles_per_gaussian, intersection_ids, flatten_ids, offsets


def _build_static_background_raster(
    background: GaussianSet,
    viewmats: Tensor,
    intrinsics: Tensor,
    width: int,
    height: int,
    *,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    tile_size: int,
    tile_width: int,
    tile_height: int,
) -> StaticBackgroundRaster:
    projection = _project_background(
        background,
        viewmats,
        intrinsics,
        width,
        height,
        near_plane,
        far_plane,
        radius_clip,
    )
    camera_ids, gaussian_ids, radii, means2d, depths, conics = projection
    tiles_per_gaussian, intersection_ids, flatten_ids, offsets = _sort_projected_part(
        camera_ids,
        radii,
        means2d,
        depths,
        n_cameras=viewmats.shape[0],
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
    )
    return StaticBackgroundRaster(
        camera_ids=camera_ids,
        gaussian_ids=gaussian_ids,
        radii=radii,
        means2d=means2d,
        depths=depths,
        conics=conics,
        tiles_per_gaussian=tiles_per_gaussian,
        intersection_ids=intersection_ids,
        flatten_ids=flatten_ids,
        offsets=offsets,
    )


def merge_sorted_intersections(
    static_intersection_ids: Tensor,
    static_flatten_ids: Tensor,
    static_offsets: Tensor,
    dynamic_intersection_ids: Tensor,
    dynamic_flatten_ids: Tensor,
    dynamic_offsets: Tensor,
    *,
    dynamic_index_offset: int,
) -> tuple[Tensor, Tensor]:
    """Merge cached-static and freshly sorted dynamic tile/depth streams.

    ``isect_tiles`` encodes camera, tile, and positive depth in one sortable
    int64. Searching only the smaller dynamic stream into the cached static
    stream avoids another full-scene radix pass. Equal-depth ties keep static
    entries first, providing deterministic output across cache hits.
    """

    if static_intersection_ids.ndim != 1 or dynamic_intersection_ids.ndim != 1:
        raise ValueError("intersection ids must be one-dimensional")
    if static_flatten_ids.shape != static_intersection_ids.shape:
        raise ValueError("static ids and flatten ids must have matching shapes")
    if dynamic_flatten_ids.shape != dynamic_intersection_ids.shape:
        raise ValueError("dynamic ids and flatten ids must have matching shapes")
    if static_offsets.shape != dynamic_offsets.shape:
        raise ValueError("static and dynamic offsets must have matching shapes")
    if dynamic_index_offset < 0:
        raise ValueError("dynamic_index_offset must be non-negative")
    if static_intersection_ids.device != dynamic_intersection_ids.device:
        raise ValueError("static and dynamic intersections must share a device")

    static_count = static_intersection_ids.numel()
    dynamic_count = dynamic_intersection_ids.numel()
    if dynamic_count == 0:
        return static_flatten_ids, static_offsets
    if static_count == 0:
        return dynamic_flatten_ids + dynamic_index_offset, dynamic_offsets

    merged_flatten_ids = merge_sorted_index_streams(
        static_intersection_ids,
        static_flatten_ids,
        dynamic_intersection_ids,
        dynamic_flatten_ids,
        dynamic_index_offset=dynamic_index_offset,
    )

    def tile_counts(offsets: Tensor, total: int) -> Tensor:
        starts = offsets.reshape(-1)
        final = starts.new_tensor([total])
        ends = torch.cat((starts[1:], final))
        return ends - starts

    combined_counts = tile_counts(static_offsets, static_count) + tile_counts(
        dynamic_offsets, dynamic_count
    )
    cumulative = torch.cumsum(combined_counts, dim=0)
    merged_offsets = (cumulative - combined_counts).reshape_as(static_offsets).to(torch.int32)
    return merged_flatten_ids, merged_offsets


def rasterize_compact_scene(
    background: GaussianSet,
    instances: list[CompactRenderInstance],
    viewmats: Tensor,
    intrinsics: Tensor,
    width: int,
    height: int,
    *,
    sh_degree: int | None = None,
    backgrounds: Tensor | None = None,
    auxiliary: bool = False,
    near_plane: float = 0.01,
    far_plane: float = 500.0,
    radius_clip: float = 0.0,
    background_radius_clip: float = 0.0,
    tile_size: int = 16,
    asset_batch_size: int = 1,
    rotate_sh: bool = True,
    background_cache: StaticBackgroundRasterCache | None = None,
    background_cache_key: Hashable | None = None,
    incremental_merge_max_dynamic_ratio: float = 1.25,
) -> CompactRasterizationResult:
    """Rasterize background plus VQ assets without decoding or 3D concatenation.

    During inference, an optional background cache retains projection and the
    static radix result. Dynamic intersections are sorted independently and
    inserted into that stream while their screen-space population remains
    small. Dense dynamic scenes automatically use the faster full-scene radix
    path. Autograd renders always bypass persistent caching.
    """

    from gsplat.cuda._wrapper import rasterize_to_pixels

    if background.device.type != "cuda":
        raise ValueError("compact unified rasterization requires CUDA")
    if viewmats.ndim != 3 or viewmats.shape[1:] != (4, 4):
        raise ValueError("viewmats must have shape [C, 4, 4]")
    if intrinsics.shape != (viewmats.shape[0], 3, 3):
        raise ValueError("intrinsics must have shape [C, 3, 3]")
    if asset_batch_size <= 0:
        raise ValueError("asset_batch_size must be positive")
    if incremental_merge_max_dynamic_ratio < 0:
        raise ValueError("incremental_merge_max_dynamic_ratio must be non-negative")
    degree = background.sh_degree if sh_degree is None else sh_degree
    if any(instance.asset.sh_degree < degree for instance in instances):
        raise ValueError("every compact asset must support the requested SH degree")
    camera_centers = torch.linalg.inv(viewmats)[:, :3, 3]
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)
    cache_enabled = background_cache is not None and not torch.is_grad_enabled()
    cache_hit = False
    static_raster = None
    cache_fill_key: Hashable | None = None
    cache_fill_projection: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor] | None = None
    if cache_enabled:
        assert background_cache is not None
        static_key = _static_background_key(
            background,
            viewmats,
            intrinsics,
            width,
            height,
            near_plane,
            far_plane,
            background_radius_clip,
            tile_size,
            background_cache_key,
        )
        static_raster = background_cache.get(static_key)
        cache_hit = static_raster is not None
        if static_raster is None:
            # On a miss, run the ordinary full-scene radix pass below and
            # extract its already-sorted static subsequence. Sorting the
            # background separately here would penalize every moving-camera
            # frame merely to populate a cache that may never hit.
            cache_fill_key = static_key
            cache_fill_projection = _project_background(
                background,
                viewmats,
                intrinsics,
                width,
                height,
                near_plane,
                far_plane,
                background_radius_clip,
            )
            projection = cache_fill_projection
        else:
            projection = (
                static_raster.camera_ids,
                static_raster.gaussian_ids,
                static_raster.radii,
                static_raster.means2d,
                static_raster.depths,
                static_raster.conics,
            )
        background_part = _background_part_from_projection(
            background,
            projection,
            degree,
            auxiliary,
            camera_centers,
        )
    else:
        background_part = _background_part(
            background,
            viewmats,
            intrinsics,
            width,
            height,
            degree,
            near_plane,
            far_plane,
            background_radius_clip,
            auxiliary,
            camera_centers,
        )

    dynamic_parts: list[tuple[Tensor, ...]] = []
    rotator = _SH_ROTATOR
    grouped_instances: dict[tuple[int, float], list[CompactRenderInstance]] = {}
    for instance in instances:
        effective_clip = radius_clip if instance.radius_clip is None else instance.radius_clip
        grouped_instances.setdefault((id(instance.asset), effective_clip), []).append(instance)
    for (_, effective_clip), group in grouped_instances.items():
        for start in range(0, len(group), asset_batch_size):
            chunk = group[start : start + asset_batch_size]
            if len(chunk) == 1:
                dynamic_parts.append(
                    _compact_part(
                        chunk[0],
                        viewmats,
                        intrinsics,
                        width,
                        height,
                        degree,
                        near_plane,
                        far_plane,
                        effective_clip,
                        auxiliary,
                        rotator,
                        rotate_sh,
                        camera_centers,
                    )
                )
            else:
                dynamic_parts.append(
                    _compact_batch_part(
                        chunk,
                        viewmats,
                        intrinsics,
                        width,
                        height,
                        degree,
                        near_plane,
                        far_plane,
                        effective_clip,
                        auxiliary,
                        rotator,
                        rotate_sh,
                        camera_centers,
                    )
                )
    parts = [background_part, *dynamic_parts]
    if len(parts) == 1:
        camera_ids, radii, means2d, depths, conics, opacities, features = background_part
    else:
        camera_ids, radii, means2d, depths, conics, opacities, features = (
            torch.cat(values, dim=0) for values in zip(*parts)
        )

    sort_mode = "full_radix"
    merge_backend = "none"
    dynamic_visible = sum(part[2].shape[0] for part in dynamic_parts)
    if static_raster is None:
        tiles_per_gaussian, intersection_ids, flatten_ids, offsets = _sort_projected_part(
            camera_ids,
            radii,
            means2d,
            depths,
            n_cameras=viewmats.shape[0],
            tile_size=tile_size,
            tile_width=tile_width,
            tile_height=tile_height,
        )
        if cache_fill_projection is not None:
            assert background_cache is not None and cache_fill_key is not None
            static_visible = background_part[2].shape[0]
            if dynamic_parts:
                static_mask = flatten_ids < static_visible
                static_intersection_ids = intersection_ids[static_mask]
                static_flatten_ids = flatten_ids[static_mask]
                from gsplat.cuda._wrapper import isect_offset_encode

                static_offsets = isect_offset_encode(
                    static_intersection_ids,
                    viewmats.shape[0],
                    tile_width,
                    tile_height,
                )
            else:
                static_intersection_ids = intersection_ids
                static_flatten_ids = flatten_ids
                static_offsets = offsets
            (
                static_camera_ids,
                static_gaussian_ids,
                static_radii,
                static_means2d,
                static_depths,
                static_conics,
            ) = cache_fill_projection
            background_cache.put(
                cache_fill_key,
                StaticBackgroundRaster(
                    camera_ids=static_camera_ids,
                    gaussian_ids=static_gaussian_ids,
                    radii=static_radii,
                    means2d=static_means2d,
                    depths=static_depths,
                    conics=static_conics,
                    tiles_per_gaussian=tiles_per_gaussian[:static_visible],
                    intersection_ids=static_intersection_ids,
                    flatten_ids=static_flatten_ids,
                    offsets=static_offsets,
                ),
            )
            sort_mode = "full_radix_cache_fill"
    elif not dynamic_parts:
        tiles_per_gaussian = static_raster.tiles_per_gaussian
        flatten_ids = static_raster.flatten_ids
        offsets = static_raster.offsets
        sort_mode = "cached_static"
    elif dynamic_visible <= static_raster.visible_gaussians * incremental_merge_max_dynamic_ratio:
        dynamic_camera_ids, dynamic_radii, dynamic_means2d, dynamic_depths = (
            torch.cat([part[index] for part in dynamic_parts], dim=0)
            for index in range(4)
        )
        (
            dynamic_tiles_per_gaussian,
            dynamic_intersection_ids,
            dynamic_flatten_ids,
            dynamic_offsets,
        ) = _sort_projected_part(
            dynamic_camera_ids,
            dynamic_radii,
            dynamic_means2d,
            dynamic_depths,
            n_cameras=viewmats.shape[0],
            tile_size=tile_size,
            tile_width=tile_width,
            tile_height=tile_height,
        )
        flatten_ids, offsets = merge_sorted_intersections(
            static_raster.intersection_ids,
            static_raster.flatten_ids,
            static_raster.offsets,
            dynamic_intersection_ids,
            dynamic_flatten_ids,
            dynamic_offsets,
            dynamic_index_offset=static_raster.visible_gaussians,
        )
        merge_backend = sorted_index_merge_backend(
            static_raster.intersections,
            dynamic_intersection_ids.numel(),
            means2d.device,
        )
        tiles_per_gaussian = torch.cat(
            (static_raster.tiles_per_gaussian, dynamic_tiles_per_gaussian)
        )
        sort_mode = "incremental_merge"
    else:
        tiles_per_gaussian, _, flatten_ids, offsets = _sort_projected_part(
            camera_ids,
            radii,
            means2d,
            depths,
            n_cameras=viewmats.shape[0],
            tile_size=tile_size,
            tile_width=tile_width,
            tile_height=tile_height,
        )
        sort_mode = "full_radix_dynamic_fallback"
    if backgrounds is None:
        backgrounds = torch.zeros(
            (viewmats.shape[0], features.shape[-1]),
            dtype=features.dtype,
            device=features.device,
        )
    elif auxiliary:
        backgrounds = torch.cat(
            (
                backgrounds,
                torch.zeros(
                    (backgrounds.shape[0], features.shape[-1] - backgrounds.shape[-1]),
                    dtype=backgrounds.dtype,
                    device=backgrounds.device,
                ),
            ),
            dim=-1,
        )
    rendered, alpha = rasterize_to_pixels(
        means2d,
        conics,
        features,
        opacities,
        width,
        height,
        tile_size,
        offsets,
        flatten_ids,
        backgrounds=backgrounds,
        packed=True,
    )
    depth = semantics = None
    if auxiliary:
        depth = rendered[..., 3:4] / alpha.clamp_min(1e-10)
        semantics = rendered[..., 4:]
    return CompactRasterizationResult(
        render=rendered[..., :3],
        alpha=alpha,
        depth=depth,
        semantics=semantics,
        info={
            "visible_gaussians": int(means2d.shape[0]),
            "intersections": int(flatten_ids.shape[0]),
            "static_visible_gaussians": int(background_part[2].shape[0]),
            "dynamic_visible_gaussians": int(dynamic_visible),
            "background_cache_enabled": cache_enabled,
            "background_cache_hit": cache_hit,
            "background_cache_entries": (
                0 if background_cache is None else background_cache.entries
            ),
            "background_cache_memory_bytes": (
                0 if background_cache is None else background_cache.memory_bytes
            ),
            "intersection_sort_mode": sort_mode,
            "intersection_merge_backend": merge_backend,
            "tiles_per_gaussian": tiles_per_gaussian,
            "means2d": means2d,
        },
    )
