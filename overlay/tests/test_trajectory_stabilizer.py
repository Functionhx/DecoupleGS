import numpy as np

from decouplegs.trajectory_stabilizer import (
    RouteTrajectoryStabilizer,
    TrajectoryStabilizerConfig,
)


def info(*, objects=None, velocities=None, behavior_states=None, object_ids=None):
    return {
        "ego_box": [0.0, 0.0, 0.0, 1.6, 3.0, 1.5, 0.0],
        "obj_boxes": [] if objects is None else objects,
        "obj_velocities": [] if velocities is None else velocities,
        "obj_behavior_states": [] if behavior_states is None else behavior_states,
        "obj_ids": [] if object_ids is None else object_ids,
    }


def straight_route():
    return np.stack((np.linspace(0.0, 60.0, 121), np.zeros(121)), axis=-1)


def test_disabled_stabilizer_preserves_planner_trajectory():
    plan = np.asarray([[0.2, 1.0], [0.4, 2.0], [0.6, 3.0]])
    stabilizer = RouteTrajectoryStabilizer(
        straight_route(),
        0.5,
        TrajectoryStabilizerConfig(enabled=False),
    )

    output, diagnostics = stabilizer.stabilize(plan, info())

    np.testing.assert_array_equal(output, plan)
    assert diagnostics == {"enabled": False, "applied": False}


def test_stabilizer_applies_speed_floor_and_bounds_lateral_residual():
    plan = np.stack((np.full(6, 4.0), np.arange(1.0, 7.0)), axis=-1)
    stabilizer = RouteTrajectoryStabilizer(
        straight_route(),
        0.5,
        TrajectoryStabilizerConfig(
            enabled=True,
            # Force the correction branch for this explicit bound test.
            stable_input_lateral_threshold_m=0.5,
        ),
    )

    output, diagnostics = stabilizer.stabilize(plan, info())

    np.testing.assert_allclose(output[:, 1], np.arange(3.0, 18.1, 3.0), atol=1e-8)
    np.testing.assert_allclose(output[:, 0], 1.0, atol=1e-8)
    assert diagnostics["learned_path_speed_mps"] < 6.0
    assert diagnostics["target_speed_mps"] == 6.0
    assert diagnostics["output_max_route_lateral_error_m"] == 1.0


def test_lead_vehicle_caps_assisted_target_speed():
    plan = np.stack((np.zeros(6), np.arange(1.0, 7.0)), axis=-1)
    lead_box = [10.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0]
    stabilizer = RouteTrajectoryStabilizer(
        straight_route(),
        0.5,
        TrajectoryStabilizerConfig(enabled=True),
    )

    output, diagnostics = stabilizer.stabilize(
        plan,
        info(objects=[lead_box], velocities=[[0.0, 0.0]]),
    )

    assert diagnostics["lead_bumper_gap_m"] == 6.5
    assert diagnostics["lead_speed_cap_mps"] == 0.75
    assert diagnostics["target_speed_mps"] == 0.75
    np.testing.assert_allclose(output[-1], [0.0, 2.25], atol=1e-8)


def test_adjacent_vehicle_targeting_ego_lane_is_treated_as_lead():
    plan = np.stack((np.zeros(6), np.arange(1.0, 7.0)), axis=-1)
    cut_in_box = [10.0, 3.5, 0.0, 2.0, 4.0, 1.5, 0.0]
    stabilizer = RouteTrajectoryStabilizer(
        straight_route(),
        0.5,
        TrajectoryStabilizerConfig(enabled=True),
    )

    _, diagnostics = stabilizer.stabilize(
        plan,
        info(
            objects=[cut_in_box],
            velocities=[[5.0, 0.0]],
            behavior_states=[{"targets_ego_lane": True, "speed_mps": 0.0}],
            object_ids=["cut-in"],
        ),
    )

    assert diagnostics["lead_selection_reason"] == "adjacent_lane_cut_in"
    assert diagnostics["lead_vehicle_id"] == "cut-in"
    assert diagnostics["lead_bumper_gap_m"] == 6.5
    assert diagnostics["target_speed_mps"] == 0.75
