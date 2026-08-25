from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from decouplegs.behavior import (
    IDMParameters,
    InteractiveTrafficState,
    LaneVehicleState,
    PolylineIDMMOBILEngine,
    PolylineLaneNetwork,
    idm_acceleration,
    mobil_incentive,
)
from decouplegs.closed_loop_metrics import (
    aggregate_episodes,
    evaluate_episode,
    frame_ttc,
)
from decouplegs.compression import (
    CompactGaussianAsset,
    CompressionConfig,
    compress_asset,
    importance_score,
)
from decouplegs.hugsim import HUGSIMRuntimeBridge
from decouplegs.hdri_protocol import (
    apply_probe_batch,
    error_metrics,
    global_affine_targets,
    improvement_gate,
)
from decouplegs.kernels import indexed_spherical_harmonics
from decouplegs.metrics import (
    channel_mse_psnr,
    masked_channel_mse_psnr,
    masked_psnr,
    minimum_ttc,
    route_completion,
    trajectory_ade,
)
from decouplegs.nuscenes_gt import NUSCENES_CAMERAS, build_hugsim_12hz_manifest
from decouplegs.nuscenes_protocol import (
    future_lidar_trajectory,
    navigation_command,
    official_uniad_planner_state,
    past_lidar_offsets,
    pose_matrix,
    quaternion_matrix,
)
from decouplegs.registration import (
    GroundPlane,
    RegistrationConfig,
    apply_se2,
    bottom_anchors,
    fit_ground_plane,
    opacity_accumulated_heights,
    orthogonal_procrustes_se2,
    project_trajectory_to_polyline,
    register_trajectory_to_lanes,
    resample_polyline,
    rotation_from_yaw_and_normal,
)
from decouplegs.rasterizer import merge_sorted_intersections
from decouplegs.relighting import (
    RelightingCalibration,
    RelightingConfig,
    RelightingNormalEquations,
    adaptive_relight_asset,
    apply_contact_shadow,
    apply_contact_shadows_batched,
    contact_shadow_mask,
    dominant_light_intensity,
    relight_compact_codebook,
    sample_local_probe,
)
from decouplegs.runtime import AgentInstance, AssetLibrary, DecoupleRuntime
from decouplegs.sensor_effects import SensorLightingConfig, apply_sensor_lighting
from decouplegs.spatial import BackgroundSpatialIndex
from decouplegs.transforms import (
    RealSHRotator,
    covariance_from_scale_quaternion,
    covariance_to_scale_quaternion,
    matrix_to_quaternion,
    quaternion_to_matrix,
    real_sh_basis,
    transform_gaussians,
)
from decouplegs.types import GaussianSet
from decouplegs.visibility import OrbitVisibilityConfig, orbit_view_matrices
from sim.utils.sim_utils import trajectory_to_ilqr_reference


def synthetic_gaussians(count: int, *, degree: int = 3, semantics: int | None = 4) -> GaussianSet:
    generator = torch.Generator().manual_seed(7 + count + degree)
    means = torch.randn((count, 3), generator=generator)
    scales = torch.rand((count, 3), generator=generator) * 0.2 + 0.02
    quats = torch.nn.functional.normalize(torch.randn((count, 4), generator=generator), dim=-1)
    opacities = torch.rand(count, generator=generator) * 0.8 + 0.1
    sh = torch.randn((count, (degree + 1) ** 2, 3), generator=generator) * 0.1
    semantic_values = None
    if semantics is not None:
        semantic_values = torch.softmax(torch.randn((count, semantics), generator=generator), dim=-1)
    return GaussianSet(means, scales, quats, opacities, sh, semantic_values)


class TransformTests(unittest.TestCase):
    def test_indexed_sh_cpu_fallback_matches_explicit_gather(self) -> None:
        generator = torch.Generator().manual_seed(19)
        directions = torch.randn((17, 3), generator=generator)
        codebook = torch.randn((5, 16, 3), generator=generator)
        indices = torch.randint(5, (17,), generator=generator, dtype=torch.int16)
        expected = torch.einsum(
            "nb,nbc->nc",
            real_sh_basis(directions, 3),
            codebook[indices.to(torch.int64)],
        )
        actual = indexed_spherical_harmonics(directions, codebook, indices, 3)
        torch.testing.assert_close(actual, expected)

    def test_quaternion_matrix_roundtrip(self) -> None:
        quaternions = torch.nn.functional.normalize(torch.randn(64, 4), dim=-1)
        rebuilt = quaternion_to_matrix(matrix_to_quaternion(quaternion_to_matrix(quaternions)))
        torch.testing.assert_close(rebuilt, quaternion_to_matrix(quaternions), atol=2e-6, rtol=2e-6)

    def test_covariance_roundtrip(self) -> None:
        asset = synthetic_gaussians(32)
        covariance = covariance_from_scale_quaternion(asset.scales, asset.quats)
        scales, quats = covariance_to_scale_quaternion(covariance)
        rebuilt = covariance_from_scale_quaternion(scales, quats)
        torch.testing.assert_close(rebuilt, covariance, atol=2e-6, rtol=2e-5)

    def test_real_sh_rotation_equivariance(self) -> None:
        asset = synthetic_gaussians(5)
        angle = torch.tensor(0.73)
        rotation = torch.tensor([
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        transformed = RealSHRotator()(asset.sh, rotation)
        world_directions = torch.nn.functional.normalize(torch.randn(31, 3), dim=-1)
        expected = torch.einsum(
            "dc,nck->ndk",
            real_sh_basis(world_directions @ rotation, 3),
            asset.sh,
        )
        actual = torch.einsum("dc,nck->ndk", real_sh_basis(world_directions, 3), transformed)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    def test_se3_transforms_mean_and_covariance(self) -> None:
        asset = synthetic_gaussians(8)
        angle = torch.tensor(-0.4)
        rotation = torch.tensor([
            [torch.cos(angle), 0.0, torch.sin(angle)],
            [0.0, 1.0, 0.0],
            [-torch.sin(angle), 0.0, torch.cos(angle)],
        ])
        pose = torch.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = torch.tensor([3.0, -1.0, 7.0])
        world = transform_gaussians(asset, pose)
        torch.testing.assert_close(world.means, asset.means @ rotation.T + pose[:3, 3])
        expected_covariance = rotation @ covariance_from_scale_quaternion(asset.scales, asset.quats) @ rotation.T
        actual_covariance = covariance_from_scale_quaternion(world.scales, world.quats)
        torch.testing.assert_close(actual_covariance, expected_covariance, atol=2e-6, rtol=2e-5)


class CompressionTests(unittest.TestCase):
    def test_importance_is_literal_weighted_sum(self) -> None:
        asset = synthetic_gaussians(2, degree=0)
        config = CompressionConfig(weight_visibility=2.0, weight_color=3.0, weight_entropy=4.0)
        visibility = torch.tensor([0.2, 0.8])
        color = torch.tensor([0.3, 0.1])
        entropy = torch.tensor([0.4, 0.2])
        score, _ = importance_score(
            asset,
            config,
            visibility=visibility,
            color_contrast=color,
            texture_entropy=entropy,
        )
        torch.testing.assert_close(score, 2.0 * visibility + 3.0 * color + 4.0 * entropy)

    def test_visibility_gated_saliency_hypothesis_is_explicit(self) -> None:
        asset = synthetic_gaussians(2, degree=0)
        visibility = torch.tensor([0.2, 0.8])
        color = torch.tensor([0.3, 0.1])
        entropy = torch.tensor([0.4, 0.2])
        config = CompressionConfig(
            weight_visibility=0.5,
            weight_color=0.3,
            weight_entropy=0.2,
            visibility_gate_saliency=True,
        )
        score, components = importance_score(
            asset,
            config,
            visibility=visibility,
            color_contrast=color,
            texture_entropy=entropy,
        )
        expected = visibility * (0.5 + 0.3 * color + 0.2 * entropy)
        torch.testing.assert_close(score, expected)
        torch.testing.assert_close(components["score_color"], visibility * color)

    def test_prune_vq_save_decode(self) -> None:
        asset = synthetic_gaussians(48, degree=2)
        visibility = torch.linspace(0.0, 1.0, len(asset))
        config = CompressionConfig(
            prune_threshold=0.2,
            shape_codebook_size=8,
            color_codebook_size=12,
            ema_iterations=3,
            ema_batch_size=24,
            kmeans_iterations=3,
            dead_code_interval=2,
        )
        compact, report = compress_asset(asset, config, visibility=visibility)
        self.assertLess(report.retained_primitives, report.input_primitives)
        self.assertGreater(report.compression_ratio, 0.0)
        decoded = compact.decode()
        self.assertEqual(len(decoded), report.retained_primitives)
        self.assertEqual(decoded.sh.shape[1:], asset.sh.shape[1:])
        for actual, expected in zip(decoded.physical_bounds, asset.physical_bounds):
            torch.testing.assert_close(actual, expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.dgs"
            compact.save(path)
            restored = CompactGaussianAsset.load(path)
            torch.testing.assert_close(restored.decode().means, decoded.means)

    def test_constant_semantics_are_stored_once_and_decode_losslessly(self) -> None:
        asset = synthetic_gaussians(32, degree=1)
        asset.semantics[:] = asset.semantics[:1]
        compact, _ = compress_asset(
            asset,
            CompressionConfig(
                prune_threshold=0.0,
                shape_codebook_size=4,
                color_codebook_size=4,
                ema_iterations=1,
                ema_batch_size=16,
                kmeans_iterations=1,
            ),
        )
        self.assertEqual(compact.semantics.shape, (1, asset.semantics.shape[1]))
        self.assertEqual(compact.metadata["semantic_encoding"], "constant")
        torch.testing.assert_close(compact.decode().semantics, asset.semantics)

    def test_proxy_orbit_cameras_are_rigid_and_look_at_asset(self) -> None:
        config = OrbitVisibilityConfig(
            azimuth_views=4,
            elevations_degrees=(0.0, 20.0),
            image_width=128,
            image_height=64,
        )
        minimum = torch.tensor([-2.0, -1.5, -1.0])
        maximum = torch.tensor([2.0, 0.0, 1.0])
        viewmats, intrinsics = orbit_view_matrices((minimum, maximum), config)
        self.assertEqual(viewmats.shape, (8, 4, 4))
        self.assertEqual(intrinsics.shape, (8, 3, 3))
        rotations = viewmats[:, :3, :3]
        identity = torch.eye(3)[None].expand(8, -1, -1)
        torch.testing.assert_close(rotations @ rotations.transpose(-1, -2), identity, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(torch.linalg.det(rotations), torch.ones(8), atol=2e-6, rtol=2e-6)
        center = 0.5 * (minimum + maximum)
        camera_center = torch.linalg.inv(viewmats)[:, :3, 3]
        forward = torch.linalg.inv(viewmats)[:, :3, 2]
        expected = torch.nn.functional.normalize(center[None] - camera_center, dim=-1)
        torch.testing.assert_close(forward, expected, atol=2e-6, rtol=2e-6)


class RasterizerStructureTests(unittest.TestCase):
    def test_incremental_intersection_merge_is_stable_and_combines_offsets(self) -> None:
        static_ids = torch.tensor([10, 20, 20, 50], dtype=torch.int64)
        static_flatten = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        static_offsets = torch.tensor([[[0, 2, 3]]], dtype=torch.int32)
        dynamic_ids = torch.tensor([5, 20, 30, 60], dtype=torch.int64)
        dynamic_flatten = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        dynamic_offsets = torch.tensor([[[0, 1, 3]]], dtype=torch.int32)

        flatten, offsets = merge_sorted_intersections(
            static_ids,
            static_flatten,
            static_offsets,
            dynamic_ids,
            dynamic_flatten,
            dynamic_offsets,
            dynamic_index_offset=4,
        )

        # Static entries precede a dynamic entry at the equal key 20.
        torch.testing.assert_close(
            flatten,
            torch.tensor([4, 0, 1, 2, 5, 6, 3, 7], dtype=torch.int32),
        )
        torch.testing.assert_close(offsets, torch.tensor([[[0, 3, 6]]], dtype=torch.int32))

    def test_incremental_intersection_merge_handles_empty_side(self) -> None:
        offsets = torch.zeros((1, 1, 2), dtype=torch.int32)
        dynamic_flatten = torch.tensor([2, 1], dtype=torch.int32)
        flatten, actual_offsets = merge_sorted_intersections(
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.int32),
            offsets,
            torch.tensor([7, 9], dtype=torch.int64),
            dynamic_flatten,
            torch.tensor([[[0, 1]]], dtype=torch.int32),
            dynamic_index_offset=8,
        )
        torch.testing.assert_close(flatten, dynamic_flatten + 8)
        torch.testing.assert_close(actual_offsets, torch.tensor([[[0, 1]]], dtype=torch.int32))


class RegistrationTests(unittest.TestCase):
    def test_hugsim_x_forward_negative_y_up_ground_rotation(self) -> None:
        config = RegistrationConfig(
            vertical_axis=1,
            vertical_sign=-1,
            horizontal_axes=(0, 2),
            forward_axis=0,
            up_axis=1,
            up_sign=-1,
        )
        normal = torch.nn.functional.normalize(torch.tensor([0.1, -1.0, 0.2]), dim=0)
        yaw = torch.tensor(0.4)
        rotation = rotation_from_yaw_and_normal(yaw, normal, config)
        flat_forward = torch.tensor([torch.cos(yaw), 0.0, torch.sin(yaw)])
        expected_forward = torch.nn.functional.normalize(
            flat_forward - (flat_forward @ normal) * normal,
            dim=0,
        )
        torch.testing.assert_close(rotation[:, 0], expected_forward)
        # Local physical up is -Y, so R @ -e_y equals world physical up.
        torch.testing.assert_close(-rotation[:, 1], normal)
        torch.testing.assert_close(torch.linalg.det(rotation), torch.tensor(1.0))

    def test_grounding_uses_physical_not_three_sigma_bounds(self) -> None:
        asset = GaussianSet(
            means=torch.tensor([[-2.0, -1.5, -1.0], [2.0, 0.0, 1.0]]),
            scales=torch.full((2, 3), 100.0),
            quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
            opacities=torch.ones(2),
            sh=torch.zeros((2, 1, 3)),
        )
        config = RegistrationConfig(
            vertical_axis=1,
            vertical_sign=-1,
            horizontal_axes=(0, 2),
            forward_axis=0,
            up_axis=1,
            up_sign=-1,
        )
        anchors = bottom_anchors(asset, config)
        torch.testing.assert_close(anchors[:, 1], torch.zeros(4))
        torch.testing.assert_close(anchors[:, 0].amin(), torch.tensor(-2.0))
        torch.testing.assert_close(anchors[:, 0].amax(), torch.tensor(2.0))

    def test_map_polyline_is_rasterized_at_decimeter_resolution(self) -> None:
        sampled = resample_polyline(torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.25, 0.0]]))
        torch.testing.assert_close(
            sampled,
            torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.25, 0.0]]),
        )

    def test_procrustes_recovers_se2(self) -> None:
        source = torch.stack((torch.linspace(0, 8, 40), torch.sin(torch.linspace(0, 2, 40))), dim=-1)
        angle = torch.tensor(0.18)
        rotation = torch.tensor([
            [torch.cos(angle), -torch.sin(angle)],
            [torch.sin(angle), torch.cos(angle)],
        ])
        target = source @ rotation.T + torch.tensor([1.2, -0.7])
        transform = orthogonal_procrustes_se2(source, target)
        torch.testing.assert_close(apply_se2(source, transform), target, atol=2e-5, rtol=2e-5)
        self.assertGreater(float(torch.linalg.det(transform[:2, :2])), 0.999)

    def test_dtw_lane_selection_and_correction(self) -> None:
        trajectory = torch.stack((torch.linspace(0, 10, 50), 0.2 * torch.sin(torch.linspace(0, 3, 50))), dim=-1)
        correct_lane = trajectory + torch.tensor([0.1, 0.08])
        wrong_lane = trajectory + torch.tensor([0.0, 5.0])
        result = register_trajectory_to_lanes(
            trajectory,
            [wrong_lane, correct_lane],
            RegistrationConfig(map_resolution=None),
        )
        self.assertEqual(result.lane_index, 1)
        self.assertLess(float((apply_se2(trajectory, result.transform) - correct_lane).abs().max()), 0.03)

    def test_monotone_continuous_lane_projection(self) -> None:
        lane_x = torch.linspace(0.0, 12.0, 121)
        lane = torch.stack((lane_x, 0.15 * torch.sin(lane_x / 2.0)), dim=-1)
        trajectory = lane[5:111:5].clone()
        trajectory[:, 1] += 0.45
        # Inject a longitudinal reversal that independent closest-point
        # projection would preserve. H-GEO-01 must return ordered arc lengths.
        trajectory[9, 0] = trajectory[8, 0] - 0.2
        result = project_trajectory_to_polyline(trajectory, lane)
        self.assertTrue(bool((result.arc_lengths[1:] >= result.arc_lengths[:-1]).all()))
        nearest = torch.cdist(result.points, lane).min(dim=-1).values
        self.assertLess(float(nearest.max()), 0.051)

        reversed_result = project_trajectory_to_polyline(trajectory, lane.flip(0))
        torch.testing.assert_close(reversed_result.points, result.points)

    def test_opacity_ground_plane(self) -> None:
        horizontal = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        samples = []
        for x, y in horizontal:
            z = 0.1 * x - 0.05 * y + 0.3
            samples.append(torch.tensor([x, y, z]))
        background = torch.stack(samples)
        heights, valid = opacity_accumulated_heights(
            horizontal,
            background,
            torch.ones(4),
            RegistrationConfig(column_radius=0.1),
        )
        self.assertTrue(bool(valid.all()))
        plane = fit_ground_plane(horizontal, heights)
        expected = 0.1 * horizontal[:, 0] - 0.05 * horizontal[:, 1] + 0.3
        torch.testing.assert_close(plane.height(horizontal), expected, atol=2e-5, rtol=2e-5)

    def test_grounding_uses_opacity_only_within_column(self) -> None:
        heights, valid = opacity_accumulated_heights(
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 1.0], [0.09, 0.0, 3.0]]),
            torch.tensor([0.25, 0.75]),
            RegistrationConfig(column_radius=0.1),
        )
        self.assertTrue(bool(valid.item()))
        torch.testing.assert_close(heights, torch.tensor([2.5]))

        downward_plane = GroundPlane(torch.tensor([0.0, 0.0, -1.0]), torch.tensor(2.0))
        torch.testing.assert_close(downward_plane.height(torch.tensor([[4.0, -3.0]])), torch.tensor([2.0]))

    def test_indexed_ground_columns_match_dense_equation(self) -> None:
        generator = torch.Generator().manual_seed(33)
        background = torch.randn((607, 3), generator=generator)
        opacity = torch.rand(607, generator=generator)
        queries = torch.tensor([[0.0, 0.0], [0.4, -0.7], [15.0, 15.0]])
        config = RegistrationConfig(
            column_radius=0.45,
            opacity_threshold=0.2,
            max_vertical_distance=1.1,
        )
        reference = torch.tensor([0.1, -0.2, 0.0])
        expected_heights, expected_valid = opacity_accumulated_heights(
            queries,
            background,
            opacity,
            config,
            reference_height=reference,
            chunk_size=89,
        )
        actual_heights, actual_valid = opacity_accumulated_heights(
            queries,
            background,
            opacity,
            config,
            reference_height=reference,
            spatial_index=BackgroundSpatialIndex(background),
        )
        torch.testing.assert_close(actual_valid, expected_valid)
        torch.testing.assert_close(
            actual_heights,
            expected_heights,
            rtol=2e-6,
            atol=2e-6,
            equal_nan=True,
        )


class RelightingTests(unittest.TestCase):
    def test_prepared_compact_cache_key_tracks_effect_switches(self) -> None:
        runtime = DecoupleRuntime(
            AssetLibrary(),
            opacity_grounding=False,
            semantic_grounding=False,
            relighting=False,
            adaptive_relighting=False,
            contact_shadows=False,
        )
        bridge = HUGSIMRuntimeBridge(runtime, static_background_cache=False)
        viewpoint = SimpleNamespace(timestamp=3.0)
        background = object()
        initial = bridge._compact_cache_key(viewpoint, background, {})
        for field in (
            "opacity_grounding",
            "semantic_grounding",
            "relighting",
            "adaptive_relighting",
            "contact_shadows",
        ):
            setattr(runtime, field, True)
            changed = bridge._compact_cache_key(viewpoint, background, {})
            self.assertNotEqual(changed, initial, field)
            setattr(runtime, field, False)

    def test_compact_codebook_relighting_matches_decode_then_relight(self) -> None:
        asset = synthetic_gaussians(24, degree=1)
        compact, _ = compress_asset(
            asset,
            CompressionConfig(
                prune_threshold=0.0,
                shape_codebook_size=4,
                color_codebook_size=6,
                ema_iterations=1,
                ema_batch_size=12,
                kmeans_iterations=1,
            ),
        )
        descriptor = torch.zeros(27)
        descriptor[:3] = torch.tensor((0.3, -0.1, 0.2))
        config = RelightingConfig(adaptive_strength=0.8)
        expected = adaptive_relight_asset(compact.decode(), descriptor, config)
        actual = relight_compact_codebook(compact, descriptor, config=config).decode()
        torch.testing.assert_close(actual.sh, expected.sh)

    def test_negative_stored_dc_coefficients_still_cast_a_shadow(self) -> None:
        descriptor = torch.zeros(27)
        descriptor[:3] = torch.tensor([-0.12, -0.15, -0.20])
        expected = (descriptor[:3] * 0.28209479177387814 + 0.5).max()
        torch.testing.assert_close(dominant_light_intensity(descriptor), expected)
        self.assertGreater(float(expected), 0.0)

    def test_adaptive_fallback_transfers_dc_exposure_and_tint(self) -> None:
        coefficient = 0.28209479177387814
        asset = synthetic_gaussians(4, degree=1)
        canonical_rgb = torch.full((4, 3), 0.4)
        asset.sh[:, 0] = (canonical_rgb - 0.5) / coefficient
        asset.sh[:, 1:] = 0.2
        ambient_rgb = torch.tensor([1.0, 0.5, 0.25])
        descriptor = torch.zeros(27)
        descriptor[:3] = (ambient_rgb - 0.5) / coefficient
        config = RelightingConfig(
            adaptive_strength=1.0,
            adaptive_reference_intensity=0.5,
            adaptive_min_gain=0.1,
            adaptive_max_gain=3.0,
        )
        relit = adaptive_relight_asset(asset, descriptor, config)
        expected_gain = torch.tensor([2.0, 1.0, 0.5])
        torch.testing.assert_close(relit.sh[:, 0] * coefficient + 0.5, canonical_rgb * expected_gain)
        torch.testing.assert_close(relit.sh[:, 1:], asset.sh[:, 1:] * expected_gain)
        torch.testing.assert_close(dominant_light_intensity(descriptor), torch.tensor(1.0))
        self.assertEqual(relit.metadata["relighting_kind"], "adaptive_fallback")

    def test_ols_affine_operator(self) -> None:
        generator = torch.Generator().manual_seed(14)
        descriptors = torch.randn((14, 27), generator=generator)
        canonical = torch.randn((7, 4), generator=generator)
        weight_scale = torch.randn((4, 27), generator=generator) * 0.02
        bias_scale = torch.randn(4, generator=generator) * 0.05 + 1.0
        weight_bias = torch.randn((4, 27), generator=generator) * 0.02
        bias_bias = torch.randn(4, generator=generator) * 0.05
        scales = descriptors @ weight_scale.T + bias_scale
        biases = descriptors @ weight_bias.T + bias_bias
        targets = canonical[None] * scales[:, None] + biases[:, None]
        fitted = RelightingCalibration.fit_ols(descriptors, canonical, targets)
        self.assertEqual(fitted.weight_scale.dtype, torch.float32)
        for probe in (1, 9):
            actual = fitted.apply(canonical.reshape(7, 2, 2), descriptors[probe])
            torch.testing.assert_close(actual.reshape(7, 4), targets[probe], atol=1e-4, rtol=1e-4)

    def test_cached_normal_equations_match_direct_fit_and_probe_batch(self) -> None:
        generator = torch.Generator().manual_seed(141)
        descriptors = torch.randn((32, 27), generator=generator) * 0.15
        canonical = torch.randn((9, 4), generator=generator) * 0.2
        targets = canonical[None] * (1.0 + descriptors[:, None, :1]) + 0.03
        direct = RelightingCalibration.fit_ols(
            descriptors,
            canonical,
            targets,
            ridge=0.01,
            ridge_prior="identity",
        )
        cached = RelightingNormalEquations.from_samples(
            descriptors, canonical, targets
        ).solve(ridge=0.01, ridge_prior="identity")
        torch.testing.assert_close(cached.weight_scale, direct.weight_scale)
        torch.testing.assert_close(cached.bias_scale, direct.bias_scale)
        expected = torch.stack(
            [cached.apply(canonical.reshape(9, 2, 2), value) for value in descriptors[:3]]
        )
        actual = apply_probe_batch(cached, canonical.reshape(9, 2, 2), descriptors[:3])
        torch.testing.assert_close(actual, expected)

    def test_operator_oracle_improves_heldout_proxy_metrics(self) -> None:
        generator = torch.Generator().manual_seed(2026)
        canonical = torch.randn((24, 16, 3), generator=generator) * 0.2
        descriptors = torch.randn((40, 27), generator=generator) * 0.12
        targets = global_affine_targets(canonical, descriptors)
        calibration = RelightingCalibration.fit_ols(
            descriptors[:32],
            canonical.reshape(24, -1),
            targets[:32].reshape(32, 24, -1),
            ridge=1e-6,
            ridge_prior="identity",
        )
        predicted = apply_probe_batch(calibration, canonical, descriptors[32:])
        baseline = canonical[None].expand_as(targets[32:])
        gate = improvement_gate(
            error_metrics(baseline, targets[32:]),
            error_metrics(predicted, targets[32:]),
        )
        self.assertTrue(gate["passed"])

    def test_probe_and_superellipse_shadow(self) -> None:
        means = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        sh = torch.zeros((2, 9, 3))
        sh[0] = 1.0
        sh[1] = 3.0
        descriptor = sample_local_probe(torch.tensor([1.0, 0.0, 0.0]), means, sh)
        torch.testing.assert_close(descriptor, torch.full((27,), 2.0))
        coordinates = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
        shadow = contact_shadow_mask(coordinates, (2.0, 2.0), exponent=4.0, decay=2.0)
        self.assertEqual(float(shadow[0]), 1.0)
        self.assertGreater(float(shadow[1]), float(shadow[2]))

    def test_indexed_probe_matches_dense_equation_and_nearest_fallback(self) -> None:
        generator = torch.Generator().manual_seed(91)
        means = torch.randn((401, 3), generator=generator)
        sh = torch.randn((401, 9, 3), generator=generator)
        visibility = torch.rand(401, generator=generator)
        visibility[:8] = 0.0
        positions = torch.stack((means[2], means[220], torch.tensor([20.0, 20.0, 20.0])))
        config = RelightingConfig(probe_sigma=0.35, probe_radius=0.55)
        expected = sample_local_probe(
            positions,
            means,
            sh,
            visibility=visibility,
            config=config,
            chunk_size=73,
        )
        actual = sample_local_probe(
            positions,
            means,
            sh,
            visibility=visibility,
            config=config,
            spatial_index=BackgroundSpatialIndex(means),
        )
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


class RuntimeAndClosedLoopTests(unittest.TestCase):
    def test_runtime_spatial_index_accepts_inference_tensors(self) -> None:
        runtime = DecoupleRuntime(
            AssetLibrary(),
            opacity_grounding=True,
            semantic_grounding=False,
            relighting=False,
            contact_shadows=False,
            frustum_culling=False,
        )
        with torch.inference_mode():
            background = synthetic_gaussians(19, degree=1)
            asset = synthetic_gaussians(6, degree=1)
            runtime.library.add("car", asset)
            instance = AgentInstance("car-0", "car", torch.eye(4))
            first = runtime.compose(background, [instance])
            second = runtime.compose(background, [instance])
        self.assertEqual(len(first.merged), len(background) + len(asset))
        self.assertEqual(len(second.merged), len(first.merged))

    def test_scenario_lighting_is_deterministic_and_identity_is_exact(self) -> None:
        image = np.full((32, 48, 3), 128, dtype=np.uint8)
        self.assertIs(
            apply_sensor_lighting(image, "CAM_FRONT", SensorLightingConfig()),
            image,
        )
        config = SensorLightingConfig(exposure=0.4, gamma=1.2, glare_strength=0.5)
        front = apply_sensor_lighting(image, "CAM_FRONT", config)
        front_again = apply_sensor_lighting(image, "CAM_FRONT", config)
        back = apply_sensor_lighting(image, "CAM_BACK", config)
        np.testing.assert_array_equal(front, front_again)
        self.assertGreater(int(front.max()), int(back.max()))
        self.assertLess(float(back.mean()), float(image.mean()))

    def test_polyline_idm_mobil_executes_smooth_safe_lane_change(self) -> None:
        network = PolylineLaneNetwork.from_centerline(
            [[0.0, 0.0], [100.0, 0.0]], [-3.5, 0.0]
        )
        engine = PolylineIDMMOBILEngine(network)
        follower = InteractiveTrafficState("follower", 0, 10.0, 10.0)
        leader = InteractiveTrafficState("leader", 0, 20.0, 0.0)
        updated = engine.step([follower, leader], 0.5)
        self.assertEqual(updated[0].target_lane, 1)
        pose = engine.pose(updated[0])
        self.assertGreater(pose.lane_change_fraction, 0.0)
        self.assertLess(pose.lane_change_fraction, 1.0)
        self.assertGreater(pose.position[1], -3.5)
        self.assertLess(pose.position[1], 0.0)

        braking = engine.step(updated, 0.5, hard_brake_ids=["follower"])
        self.assertLessEqual(braking[0].acceleration, -6.0)

    def test_polyline_traffic_sync_prevents_centerline_overlap(self) -> None:
        network = PolylineLaneNetwork.from_centerline(
            [[0.0, 0.0], [100.0, 0.0]], [0.0]
        )
        engine = PolylineIDMMOBILEngine(network)
        states = [
            InteractiveTrafficState("leader", 0, 20.0, 0.0, length=4.0),
            InteractiveTrafficState("follower", 0, 19.0, 8.0, length=4.0),
        ]
        leader, follower = engine.step(states, 0.1, allow_lane_changes=False)
        self.assertGreaterEqual(leader.progress - follower.progress, 4.1 - 1e-9)

    def test_polyline_terminal_vehicle_stops_and_emits_retirement_event(self) -> None:
        network = PolylineLaneNetwork.from_centerline(
            [[0.0, 0.0], [10.0, 0.0]], [0.0]
        )
        engine = PolylineIDMMOBILEngine(network)
        state = InteractiveTrafficState("terminal", 0, 9.0, 5.0)

        (updated,) = engine.step([state], 0.5, allow_lane_changes=False)

        self.assertEqual(updated.progress, 10.0)
        self.assertEqual(updated.speed, 0.0)
        self.assertEqual(engine.last_terminal_vehicle_ids, {"terminal"})

    def test_closed_loop_ttc_keeps_literal_and_closing_definitions_separate(self) -> None:
        approaching = {
            "ego_box": [0.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0],
            "ego_velocity_xy": [2.0, 0.0],
            "obj_boxes": [[10.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0]],
            "obj_velocities": [[0.0, 0.0]],
        }
        metrics = frame_ttc(approaching)
        self.assertAlmostEqual(metrics["paper_literal_center"], 5.0)
        self.assertAlmostEqual(metrics["closing_center"], 5.0)
        self.assertAlmostEqual(metrics["closing_clearance"], 3.0)

        receding = dict(approaching)
        receding["ego_velocity_xy"] = [-2.0, 0.0]
        metrics = frame_ttc(receding)
        self.assertAlmostEqual(metrics["paper_literal_center"], 5.0)
        self.assertTrue(math.isinf(metrics["closing_center"]))

    def test_closed_loop_ds_is_not_invented_when_paper_factors_are_missing(self) -> None:
        record = {
            "frames": [
                {
                    "ego_box": [0.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0],
                    "ego_velocity_xy": [1.0, 0.0],
                    "obj_boxes": [],
                    "obj_velocities": [],
                    "route_distance_traveled": 9.5,
                    "route_distance_planned": 10.0,
                    "rc_distance": 0.95,
                    "rc": 0.9,
                    "transition": {"collision": False},
                }
            ],
            "episode": {"steps": 1, "termination_reason": "time_limit"},
        }
        unresolved = evaluate_episode(record)
        self.assertIsNone(unresolved["driving_score"])
        self.assertAlmostEqual(unresolved["route_completion"], 0.95)
        self.assertFalse(
            unresolved["success_variants"]["strict_completion_h_metric_01"]
        )
        self.assertTrue(unresolved["success_variants"]["safety_only"])
        self.assertFalse(unresolved["success_variants"]["terminal_route_complete"])
        resolved = evaluate_episode(record, penalty_factors={"collision": 0.5})
        self.assertAlmostEqual(resolved["driving_score"], 0.95)
        aggregate = aggregate_episodes([resolved, resolved])
        self.assertAlmostEqual(aggregate["route_completion"]["mean"], 0.95)
        self.assertAlmostEqual(
            aggregate["success_rate_variants"]["safety_only"]["mean"], 1.0
        )

    def test_ilqr_reference_heading_uses_post_swap_axes(self) -> None:
        straight = trajectory_to_ilqr_reference(
            np.asarray([[0.0, 1.0], [0.0, 2.0]], dtype=np.float64)
        )
        np.testing.assert_allclose(straight[1:, :2], [[1.0, 0.0], [2.0, 0.0]])
        np.testing.assert_allclose(straight[1:, 2], 0.0, atol=1e-12)

        diagonal = trajectory_to_ilqr_reference(
            np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
        )
        np.testing.assert_allclose(diagonal[1:, 2], math.pi / 4.0, atol=1e-12)

    def test_exact_calibration_takes_priority_over_adaptive_fallback(self) -> None:
        background = synthetic_gaussians(11, degree=2)
        background.visibility = torch.ones(len(background))
        asset = synthetic_gaussians(6, degree=2)
        calibration = RelightingCalibration.identity(asset.sh.shape[1] * asset.sh.shape[2])
        library = AssetLibrary()
        library.add("car", asset, calibration)
        runtime = DecoupleRuntime(
            library,
            rotate_sh=False,
            opacity_grounding=False,
            relighting=True,
            adaptive_relighting=True,
            contact_shadows=False,
            frustum_culling=False,
        )
        scene = runtime.compose(
            background,
            [AgentInstance("car-0", "car", torch.eye(4))],
        )
        torch.testing.assert_close(scene.assets["car-0"].sh, asset.sh)
        self.assertNotIn("relighting_kind", scene.assets["car-0"].metadata)

    def test_runtime_merges_one_canonical_instance(self) -> None:
        background = synthetic_gaussians(11, degree=1)
        background.visibility = torch.ones(len(background))
        asset = synthetic_gaussians(6, degree=1)
        library = AssetLibrary()
        library.add("car", asset)
        runtime = DecoupleRuntime(
            library,
            opacity_grounding=False,
            relighting=False,
            contact_shadows=False,
            frustum_culling=False,
        )
        transform = torch.eye(4)
        transform[:3, 3] = torch.tensor([4.0, 0.0, 2.0])
        scene = runtime.compose(background, [AgentInstance("car-0", "car", transform)])
        self.assertEqual(len(scene.merged), len(background) + len(asset))
        self.assertIsNone(scene.merged.visibility)
        torch.testing.assert_close(scene.assets["car-0"].means, asset.means + transform[:3, 3])

    def test_runtime_can_estimate_shadow_plane_without_moving_logged_pose(self) -> None:
        background = synthetic_gaussians(12, degree=1)
        asset = synthetic_gaussians(6, degree=1)
        library = AssetLibrary()
        library.add("car", asset)
        runtime = DecoupleRuntime(
            library,
            opacity_grounding=True,
            adjust_grounding_pose=False,
            relighting=False,
            contact_shadows=False,
            frustum_culling=False,
        )
        logged = torch.eye(4)
        logged[:3, 3] = torch.tensor([3.0, -2.0, 0.4])
        adjusted = logged.clone()
        adjusted[2, 3] += 0.75
        plane = GroundPlane(torch.tensor([0.0, 0.0, 1.0]), torch.tensor(0.0))
        grounding = SimpleNamespace(transform=adjusted, plane=plane)
        with patch("decouplegs.runtime.ground_asset_from_pose", return_value=grounding):
            scene = runtime.compose(
                background,
                [AgentInstance("car-0", "car", logged)],
            )
        torch.testing.assert_close(scene.transforms["car-0"], logged)
        torch.testing.assert_close(
            scene.assets["car-0"].means, asset.means + logged[:3, 3]
        )
        self.assertIs(scene.ground_planes["car-0"], plane)

    def test_idm_mobil_and_metrics(self) -> None:
        ego = LaneVehicleState("ego", "a", 0.0, 10.0)
        idm = IDMParameters(desired_speed=20.0)
        free = idm_acceleration(ego, None, idm)
        expected_free = 2.0 * (1.0 - (10.0 / 20.0) ** 4 - ((5.0 + 10.0 * 2.0) / 10000.0) ** 2)
        self.assertAlmostEqual(free, expected_free)
        self.assertGreater(free, 0.0)
        close_leader = LaneVehicleState("leader", "a", 5.0, 0.0)
        self.assertLess(idm_acceleration(ego, close_leader), 0.0)
        target_follower = LaneVehicleState("follower", "b", -0.1, 20.0)
        decision = mobil_incentive(
            ego,
            current_leader=close_leader,
            current_follower=None,
            target_leader=None,
            target_follower=target_follower,
        )
        self.assertFalse(decision.safe)
        image = torch.zeros((2, 2, 3))
        self.assertGreater(float(masked_psnr(image, image, torch.ones((2, 2), dtype=torch.bool))), 100.0)
        ttc = minimum_ttc(
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([[[10.0, 0.0]]]),
            torch.tensor([[2.0, 0.0]]),
            torch.tensor([[[0.0, 0.0]]]),
        )
        self.assertAlmostEqual(float(ttc), 5.0, places=5)
        self.assertEqual(route_completion(12.0, 10.0), 1.0)

    def test_trajectory_ade_is_stable_in_large_map_coordinates(self) -> None:
        trajectory = torch.tensor(
            [[1230.0, 605.0], [1230.5, 606.0], [1231.0, 607.0]],
            dtype=torch.float32,
        )
        lane = trajectory + torch.tensor([0.03, -0.04])
        self.assertAlmostEqual(float(trajectory_ade(trajectory, lane)), 0.05, places=4)

    def test_psnr_paper_formula_and_library_formula_are_explicit(self) -> None:
        image = torch.zeros((2, 3, 3))
        target = torch.ones_like(image)
        mask = torch.ones((2, 3), dtype=torch.bool)
        conventional = channel_mse_psnr(image, target)
        masked_conventional = masked_channel_mse_psnr(image, target, mask)
        paper_equation = masked_psnr(image, target, mask)
        self.assertAlmostEqual(float(conventional), 0.0, places=6)
        self.assertAlmostEqual(float(masked_conventional), 0.0, places=6)
        self.assertAlmostEqual(
            float(conventional - paper_equation),
            10.0 * torch.log10(torch.tensor(3.0)).item(),
            places=6,
        )

    def test_batched_contact_shadows_match_sequential_application(self) -> None:
        background = synthetic_gaussians(37, degree=2)
        assets = [synthetic_gaussians(9, degree=2), synthetic_gaussians(11, degree=2)]
        transforms = [torch.eye(4), torch.eye(4)]
        transforms[0][:3, 3] = torch.tensor([0.5, -0.3, 0.0])
        transforms[1][:3, 3] = torch.tensor([-0.8, 0.4, 0.0])
        planes = [
            GroundPlane(torch.tensor([0.0, 0.0, 1.0]), torch.tensor(0.0)),
            GroundPlane(torch.tensor([0.0, 0.0, 1.0]), torch.tensor(-0.05)),
        ]
        descriptors = [torch.zeros(27), torch.linspace(-0.1, 0.1, 27)]
        sequential = background
        for asset, transform, plane, descriptor in zip(
            assets, transforms, planes, descriptors, strict=True
        ):
            sequential = apply_contact_shadow(
                sequential, asset, transform, plane, descriptor
            )
        batched = apply_contact_shadows_batched(
            background,
            assets,
            transforms,
            planes,
            descriptors,
            chunk_size=7,
        )
        torch.testing.assert_close(batched.sh, sequential.sh, rtol=2e-6, atol=2e-6)

    def test_indexed_truncated_shadows_match_dense_truncated_equation(self) -> None:
        background = synthetic_gaussians(503, degree=2)
        assets = [synthetic_gaussians(13, degree=2), synthetic_gaussians(17, degree=2)]
        transforms = [torch.eye(4), torch.eye(4)]
        transforms[0][:3, 3] = torch.tensor([0.25, -0.4, 0.0])
        transforms[1][:3, 3] = torch.tensor([-0.7, 0.65, 0.0])
        planes = [
            GroundPlane(torch.tensor([0.0, 0.0, 1.0]), torch.tensor(0.0)),
            GroundPlane(torch.tensor([0.0, 0.0, 1.0]), torch.tensor(0.0)),
        ]
        descriptors = [torch.zeros(27), torch.linspace(-0.1, 0.1, 27)]
        config = RelightingConfig(shadow_mask_epsilon=1e-4)
        dense = apply_contact_shadows_batched(
            background,
            assets,
            transforms,
            planes,
            descriptors,
            relighting_config=config,
            chunk_size=71,
        )
        indexed = apply_contact_shadows_batched(
            background,
            assets,
            transforms,
            planes,
            descriptors,
            relighting_config=config,
            spatial_index=BackgroundSpatialIndex(background.means),
        )
        torch.testing.assert_close(indexed.sh, dense.sh, rtol=2e-6, atol=2e-6)
        self.assertTrue(indexed.metadata["contact_shadow_indexed"])

    def test_semantic_sparse_shadow_matches_masked_sequential_equation(self) -> None:
        background = synthetic_gaussians(97, degree=2)
        assets = [synthetic_gaussians(9, degree=2), synthetic_gaussians(12, degree=2)]
        transforms = [torch.eye(4), torch.eye(4)]
        transforms[1][:3, 3] = torch.tensor([0.5, -0.2, 0.1])
        planes = [
            GroundPlane(torch.tensor([0.0, 0.0, 1.0]), torch.tensor(0.0)),
            GroundPlane(torch.tensor([0.0, 0.0, 1.0]), torch.tensor(-0.1)),
        ]
        descriptors = [torch.zeros(27), torch.linspace(-0.05, 0.12, 27)]
        ground_mask = torch.arange(len(background)) % 3 == 0
        expected = background
        for asset, transform, plane, descriptor in zip(
            assets, transforms, planes, descriptors, strict=True
        ):
            expected = apply_contact_shadow(
                expected,
                asset,
                transform,
                plane,
                descriptor,
                ground_mask=ground_mask,
            )
        actual = apply_contact_shadows_batched(
            background,
            assets,
            transforms,
            planes,
            descriptors,
            ground_mask=ground_mask,
            chunk_size=11,
        )
        torch.testing.assert_close(actual.sh, expected.sh, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(actual.sh[~ground_mask], background.sh[~ground_mask])

    def test_nuscenes_asap_manifest_selects_keyframe_then_sweep(self) -> None:
        sensors = [
            {"token": f"sensor-{camera}", "channel": camera, "modality": "camera"}
            for camera in NUSCENES_CAMERAS
        ]
        calibrations = [
            {
                "token": f"calib-{camera}",
                "sensor_token": f"sensor-{camera}",
            }
            for camera in NUSCENES_CAMERAS
        ]
        sample_data = []
        for camera in NUSCENES_CAMERAS:
            sample_data.extend(
                [
                    {
                        "token": f"key-{camera}",
                        "sample_token": "sample-0",
                        "calibrated_sensor_token": f"calib-{camera}",
                        "filename": f"samples/{camera}/key.jpg",
                        "timestamp": 1_000_000,
                        "is_key_frame": True,
                        "prev": "",
                        "next": f"sweep-{camera}",
                    },
                    {
                        "token": f"sweep-{camera}",
                        "sample_token": "sample-0",
                        "calibrated_sensor_token": f"calib-{camera}",
                        "filename": f"sweeps/{camera}/sweep.jpg",
                        "timestamp": 1_083_333,
                        "is_key_frame": False,
                        "prev": f"key-{camera}",
                        "next": f"next-key-{camera}",
                    },
                    {
                        "token": f"next-key-{camera}",
                        "sample_token": "sample-1",
                        "calibrated_sensor_token": f"calib-{camera}",
                        "filename": f"samples/{camera}/next.jpg",
                        "timestamp": 1_500_000,
                        "is_key_frame": True,
                        "prev": f"sweep-{camera}",
                        "next": "",
                    },
                ]
            )
        mappings = build_hugsim_12hz_manifest(
            scene_name="scene-test",
            scenes=[
                {
                    "token": "scene-token",
                    "name": "scene-test",
                    "first_sample_token": "sample-0",
                }
            ],
            samples=[
                {
                    "token": "sample-0",
                    "timestamp": 1_000_000,
                    "prev": "",
                    "next": "sample-1",
                },
                {
                    "token": "sample-1",
                    "timestamp": 1_500_000,
                    "prev": "sample-0",
                    "next": "",
                },
            ],
            sample_data=sample_data,
            calibrated_sensors=calibrations,
            sensors=sensors,
            frame_count=2,
        )
        self.assertEqual(len(mappings), 12)
        self.assertTrue(all(mapping.is_key_frame for mapping in mappings[:6]))
        self.assertTrue(all(not mapping.is_key_frame for mapping in mappings[6:]))
        self.assertEqual(mappings[6].source_path, "sweeps/CAM_FRONT/sweep.jpg")
        self.assertEqual(mappings[3].output_height, 410)

    def test_nuscenes_protocol_uses_right_forward_lidar_axes(self) -> None:
        poses = []
        for index in range(8):
            pose = np.eye(4, dtype=np.float64)
            pose[:2, 3] = [0.25 * index, 1.5 * index]
            poses.append(pose)
        trajectory, mask = future_lidar_trajectory(poses, 1, steps=6)
        expected = np.asarray([[0.25 * i, 1.5 * i] for i in range(1, 7)])
        np.testing.assert_allclose(trajectory, expected)
        self.assertTrue(mask.all())
        self.assertEqual(navigation_command(trajectory, mask), 2)
        trajectory[-1, 0] = -2.1
        self.assertEqual(navigation_command(trajectory, mask), 1)

    def test_nuscenes_protocol_vad_history_extrapolates_scene_start(self) -> None:
        poses = []
        for index in range(4):
            pose = np.eye(4, dtype=np.float64)
            pose[:3, 3] = [0.2 * index, 1.5 * index, 0.0]
            poses.append(pose)
        np.testing.assert_allclose(
            past_lidar_offsets(poses, 0),
            [[0.2, 1.5], [0.2, 1.5]],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            past_lidar_offsets(poses, 3),
            [[0.2, 1.5], [0.2, 1.5]],
            atol=1e-12,
        )

    def test_nuscenes_protocol_quaternion_pose(self) -> None:
        half = math.sqrt(0.5)
        rotation = quaternion_matrix([half, 0.0, 0.0, half])
        np.testing.assert_allclose(
            rotation,
            np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            atol=1e-12,
        )
        pose = pose_matrix([1.0, 2.0, 3.0], [half, 0.0, 0.0, half])
        np.testing.assert_allclose(pose[:3, :3], rotation)
        np.testing.assert_allclose(pose[:3, 3], [1.0, 2.0, 3.0])

    def test_nuscenes_protocol_matches_uniad_row_vector_contract(self) -> None:
        half = math.sqrt(0.5)
        camera = {
            "sensor2lidar_rotation": np.eye(3),
            "sensor2lidar_translation": np.asarray([1.0, 2.0, 3.0]),
            "cam_intrinsic": np.diag([2.0, 3.0, 1.0]),
            "sample_data_token": "camera-token",
            "timestamp": 999_990,
        }
        info = {
            "token": "sample-token",
            "scene_token": "scene-token",
            "lidar2ego_rotation": [1.0, 0.0, 0.0, 0.0],
            "lidar2ego_translation": [1.0, 0.0, 0.0],
            "ego2global_rotation": [half, 0.0, 0.0, half],
            "ego2global_translation": [10.0, 20.0, 0.0],
            "can_bus": np.arange(18, dtype=np.float64),
            "cams": {name: dict(camera) for name in NUSCENES_CAMERAS},
        }
        state = official_uniad_planner_state(info)
        expected_rotation = np.asarray(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        np.testing.assert_allclose(state["l2g_r_mat"], expected_rotation, atol=1e-12)
        np.testing.assert_allclose(state["l2g_t"], [10.0, 21.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(state["can_bus"][:3], [10.0, 20.0, 0.0])
        self.assertAlmostEqual(state["can_bus"][-2], math.pi / 2)
        self.assertAlmostEqual(state["can_bus"][-1], 90.0)
        expected_lidar_to_camera = np.eye(4)
        expected_lidar_to_camera[:3, 3] = [-1.0, -2.0, -3.0]
        expected_intrinsic = np.eye(4)
        expected_intrinsic[:3, :3] = np.diag([2.0, 3.0, 1.0])
        np.testing.assert_allclose(
            state["camera_state"]["CAM_FRONT"]["lidar2img"],
            expected_intrinsic @ expected_lidar_to_camera,
        )

    def test_bottom_anchors_separate_world_and_asset_axes(self) -> None:
        asset = synthetic_gaussians(8)
        config = RegistrationConfig(
            vertical_axis=1,
            horizontal_axes=(0, 2),
            vertical_sign=-1,
            forward_axis=1,
            up_axis=2,
            up_sign=1,
        )
        anchors = bottom_anchors(asset, config)
        minimum, maximum = asset.physical_bounds
        torch.testing.assert_close(anchors[:, 2], minimum[2].expand(4))
        self.assertEqual(set(anchors[:, 0].tolist()), {float(minimum[0]), float(maximum[0])})
        self.assertEqual(set(anchors[:, 1].tolist()), {float(minimum[1]), float(maximum[1])})


if __name__ == "__main__":
    unittest.main()
