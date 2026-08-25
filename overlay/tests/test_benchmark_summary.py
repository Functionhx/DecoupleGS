from __future__ import annotations

import pytest

from decouplegs.benchmark_summary import aggregate_scene_results


def scene(name: str, raw_ms: float, compact_ms: float, drift: float) -> dict:
    fidelity = {
        "pair_count": 10,
        "vehicle_mask_images": 2,
        "psnr_all_channel_mse_db": 20.0,
        "ssim_all": 0.8,
        "lpips_all_alex": 0.2,
        "psnr_vehicle_channel_mse_db": 21.0,
        "psnr_vehicle_pixel_l2_db": 16.0,
        "peak_intensity_error": 0.3,
        "peak_angular_error_deg": 8.0,
    }
    behavior = {
        "reference": {
            "mADE_to_logged_gt_m": 1.0,
            "minTTC_paper_literal_center_s": 3.0,
        },
        "variants": {
            "raw_3dgs": {
                "frames": 5,
                "mADE_to_reference_plan_m": drift,
                "mFDE_to_reference_plan_m": 2.0 * drift,
                "mADE_to_logged_gt_m": 1.1,
                "minTTC_paper_literal_center_s": 2.9,
            },
            "compact_base": {
                "frames": 5,
                "mADE_to_reference_plan_m": drift + 0.01,
                "mFDE_to_reference_plan_m": 2.0 * drift + 0.01,
                "mADE_to_logged_gt_m": 1.1,
                "minTTC_paper_literal_center_s": 2.9,
            },
        },
    }
    return {
        "scene": name,
        "rendering": {
            "raw_3dgs": {
                "pair_count": 10,
                "mean_ms_per_camera": raw_ms,
                "camera_fps": 1000.0 / raw_ms,
                "peak_allocated_mib": 100.0,
                "gpu": "test-gpu",
            },
            "compact_base": {
                "pair_count": 10,
                "mean_ms_per_camera": compact_ms,
                "camera_fps": 1000.0 / compact_ms,
                "peak_allocated_mib": 90.0,
                "gpu": "test-gpu",
            },
        },
        "fidelity": {"raw_3dgs": dict(fidelity), "compact_base": dict(fidelity)},
        "behavior": {"uniad": behavior, "vad": behavior},
    }


def test_aggregate_uses_timed_view_and_frame_weights() -> None:
    result = aggregate_scene_results(
        [scene("a", raw_ms=10.0, compact_ms=5.0, drift=0.2), scene("b", 20.0, 10.0, 0.4)]
    )

    rendering = result["aggregate"]["rendering"]
    assert rendering["raw_3dgs"]["camera_fps"] == pytest.approx(1000.0 / 15.0)
    assert rendering["compact_base"]["camera_fps"] == pytest.approx(1000.0 / 7.5)
    assert rendering["compact_vs_raw_speedup"] == pytest.approx(2.0)
    assert (
        result["aggregate"]["behavior"]["uniad"]["variants"]["raw_3dgs"]
        ["mADE_to_reference_plan_m"]
        == pytest.approx(0.3)
    )


def test_aggregate_rejects_empty_suite() -> None:
    with pytest.raises(ValueError, match="at least one scene"):
        aggregate_scene_results([])


def test_aggregate_ignores_scenes_without_dynamic_actor_ttc() -> None:
    with_ttc = scene("actors", raw_ms=10.0, compact_ms=5.0, drift=0.2)
    without_ttc = scene("no-actors", raw_ms=10.0, compact_ms=5.0, drift=0.2)
    for planner in ("uniad", "vad"):
        without_ttc["behavior"][planner]["reference"][
            "minTTC_paper_literal_center_s"
        ] = None
        for variant in without_ttc["behavior"][planner]["variants"].values():
            variant["minTTC_paper_literal_center_s"] = None

    result = aggregate_scene_results([with_ttc, without_ttc])

    behavior = result["aggregate"]["behavior"]["uniad"]
    assert behavior["reference"]["minTTC_paper_literal_center_s"] == pytest.approx(3.0)
    assert behavior["variants"]["raw_3dgs"][
        "minTTC_paper_literal_center_s"
    ] == pytest.approx(2.9)
