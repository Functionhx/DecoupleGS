from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class IDMParameters:
    desired_speed: float = 30.0
    minimum_gap: float = 5.0
    time_headway: float = 2.0
    max_acceleration: float = 2.0
    comfortable_deceleration: float = 4.0
    exponent: float = 4.0
    minimum_numerical_gap: float = 0.1


@dataclass(frozen=True)
class MOBILParameters:
    politeness: float = 0.3
    safe_deceleration: float = 4.0
    acceleration_threshold: float = 0.2
    right_lane_bias: float = 0.0


@dataclass(frozen=True)
class LaneVehicleState:
    vehicle_id: Hashable
    lane_id: Hashable
    position: float
    speed: float
    length: float = 4.5
    acceleration: float = 0.0


@dataclass(frozen=True)
class MOBILDecision:
    safe: bool
    incentive: float
    ego_gain: float
    new_follower_gain: float
    old_follower_gain: float


def idm_acceleration(
    vehicle: LaneVehicleState,
    leader: LaneVehicleState | None,
    parameters: IDMParameters | None = None,
) -> float:
    """Longitudinal IDM response from Supplementary Eq. (25)/Algorithm 3."""

    parameters = IDMParameters() if parameters is None else parameters
    speed = max(vehicle.speed, 0.0)
    free_road = (speed / max(parameters.desired_speed, 1e-6)) ** parameters.exponent
    if leader is None:
        gap = 10000.0
        delta_speed = 0.0
    else:
        gap = max(leader.position - vehicle.position - vehicle.length, parameters.minimum_numerical_gap)
        delta_speed = speed - max(leader.speed, 0.0)
    desired_gap = (
        parameters.minimum_gap
        + speed * parameters.time_headway
        + speed
        * delta_speed
        / (2.0 * math.sqrt(parameters.max_acceleration * parameters.comfortable_deceleration))
    )
    interaction = (desired_gap / gap) ** 2
    return parameters.max_acceleration * (1.0 - free_road - interaction)


def mobil_incentive(
    ego: LaneVehicleState,
    *,
    current_leader: LaneVehicleState | None,
    current_follower: LaneVehicleState | None,
    target_leader: LaneVehicleState | None,
    target_follower: LaneVehicleState | None,
    idm: IDMParameters | None = None,
    mobil: MOBILParameters | None = None,
    lane_bias: float = 0.0,
) -> MOBILDecision:
    """Full safety and politeness criterion from Supplementary Eq. (26)."""

    idm = IDMParameters() if idm is None else idm
    mobil = MOBILParameters() if mobil is None else mobil
    ego_before = idm_acceleration(ego, current_leader, idm)
    ego_after = idm_acceleration(ego, target_leader, idm)

    if target_follower is None:
        target_before = target_after = 0.0
        safe = True
    else:
        target_before = idm_acceleration(target_follower, target_leader, idm)
        target_after = idm_acceleration(target_follower, ego, idm)
        safe = target_after >= -mobil.safe_deceleration

    if current_follower is None:
        old_before = old_after = 0.0
    else:
        old_before = idm_acceleration(current_follower, ego, idm)
        old_after = idm_acceleration(current_follower, current_leader, idm)

    ego_gain = ego_after - ego_before
    new_follower_gain = target_after - target_before
    old_follower_gain = old_after - old_before
    incentive = ego_gain + mobil.politeness * (new_follower_gain + old_follower_gain) + lane_bias
    return MOBILDecision(
        safe=safe,
        incentive=incentive,
        ego_gain=ego_gain,
        new_follower_gain=new_follower_gain,
        old_follower_gain=old_follower_gain,
    )


class IDMMOBILEngine:
    """Synchronous lane-coordinate background-agent engine for Algorithm 5."""

    def __init__(
        self,
        *,
        idm: IDMParameters | None = None,
        mobil: MOBILParameters | None = None,
    ) -> None:
        self.idm = IDMParameters() if idm is None else idm
        self.mobil = MOBILParameters() if mobil is None else mobil

    @staticmethod
    def _neighbors(
        states: Sequence[LaneVehicleState],
        subject: LaneVehicleState,
        lane_id: Hashable,
    ) -> tuple[LaneVehicleState | None, LaneVehicleState | None]:
        lane = [state for state in states if state.lane_id == lane_id and state.vehicle_id != subject.vehicle_id]
        leaders = [state for state in lane if state.position >= subject.position]
        followers = [state for state in lane if state.position < subject.position]
        leader = min(leaders, key=lambda state: state.position, default=None)
        follower = max(followers, key=lambda state: state.position, default=None)
        return leader, follower

    def choose_lane(
        self,
        subject: LaneVehicleState,
        states: Sequence[LaneVehicleState],
        feasible_lanes: Sequence[Hashable],
    ) -> Hashable:
        current_leader, current_follower = self._neighbors(states, subject, subject.lane_id)
        best_lane = subject.lane_id
        best_incentive = self.mobil.acceleration_threshold
        for target_lane in feasible_lanes:
            if target_lane == subject.lane_id:
                continue
            target_leader, target_follower = self._neighbors(states, subject, target_lane)
            decision = mobil_incentive(
                subject,
                current_leader=current_leader,
                current_follower=current_follower,
                target_leader=target_leader,
                target_follower=target_follower,
                idm=self.idm,
                mobil=self.mobil,
            )
            if decision.safe and decision.incentive > best_incentive:
                best_lane = target_lane
                best_incentive = decision.incentive
        return best_lane

    def step(
        self,
        states: Sequence[LaneVehicleState],
        feasible_lanes: Mapping[Hashable, Sequence[Hashable]],
        dt: float,
    ) -> list[LaneVehicleState]:
        if dt <= 0:
            raise ValueError("dt must be positive")
        lane_choices = {
            state.vehicle_id: self.choose_lane(
                state,
                states,
                feasible_lanes.get(state.lane_id, (state.lane_id,)),
            )
            for state in states
        }
        output = []
        for state in states:
            lane = lane_choices[state.vehicle_id]
            leader, _ = self._neighbors(states, state, lane)
            acceleration = idm_acceleration(state, leader, self.idm)
            speed = max(0.0, state.speed + acceleration * dt)
            position = state.position + state.speed * dt + 0.5 * acceleration * dt * dt
            output.append(
                replace(
                    state,
                    lane_id=lane,
                    position=position,
                    speed=speed,
                    acceleration=acceleration,
                )
            )
        return output


class PolylineLaneNetwork:
    """Metric lane-coordinate adapter for interactive background traffic."""

    def __init__(self, lanes: Mapping[Hashable, Sequence[Sequence[float]]]) -> None:
        if not lanes:
            raise ValueError("at least one lane polyline is required")
        self.points: dict[Hashable, np.ndarray] = {}
        self.segment_lengths: dict[Hashable, np.ndarray] = {}
        self.cumulative: dict[Hashable, np.ndarray] = {}
        for lane_id, raw_points in lanes.items():
            points = np.asarray(raw_points, dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
                raise ValueError("each lane must have shape [N>=2, 2]")
            keep = np.concatenate(
                [[True], np.linalg.norm(np.diff(points, axis=0), axis=-1) > 1e-6]
            )
            points = points[keep]
            if len(points) < 2:
                raise ValueError("lane polyline must contain two distinct points")
            lengths = np.linalg.norm(np.diff(points, axis=0), axis=-1)
            self.points[lane_id] = points
            self.segment_lengths[lane_id] = lengths
            self.cumulative[lane_id] = np.concatenate([[0.0], np.cumsum(lengths)])

    @classmethod
    def from_centerline(
        cls,
        centerline: Sequence[Sequence[float]],
        offsets: Sequence[float],
    ) -> "PolylineLaneNetwork":
        points = np.asarray(centerline, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("centerline must have shape [N>=2, 2]")
        tangent = np.empty_like(points)
        tangent[0] = points[1] - points[0]
        tangent[-1] = points[-1] - points[-2]
        if len(points) > 2:
            tangent[1:-1] = points[2:] - points[:-2]
        tangent /= np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-9)
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)
        return cls(
            {
                lane_index: points + float(offset) * normal
                for lane_index, offset in enumerate(offsets)
            }
        )

    def length(self, lane_id: Hashable) -> float:
        return float(self.cumulative[lane_id][-1])

    def sample(self, lane_id: Hashable, progress: float) -> tuple[np.ndarray, np.ndarray]:
        cumulative = self.cumulative[lane_id]
        points = self.points[lane_id]
        progress = float(np.clip(progress, 0.0, cumulative[-1]))
        index = min(int(np.searchsorted(cumulative, progress, side="right") - 1), len(points) - 2)
        length = self.segment_lengths[lane_id][index]
        ratio = (progress - cumulative[index]) / max(float(length), 1e-9)
        tangent = (points[index + 1] - points[index]) / max(float(length), 1e-9)
        return points[index] + ratio * (points[index + 1] - points[index]), tangent

    def project(self, lane_id: Hashable, point: Sequence[float]) -> tuple[float, float]:
        point_array = np.asarray(point, dtype=np.float64)
        points = self.points[lane_id]
        delta = np.diff(points, axis=0)
        denom = np.maximum(self.segment_lengths[lane_id] ** 2, 1e-12)
        ratio = np.clip(
            np.sum((point_array - points[:-1]) * delta, axis=-1) / denom,
            0.0,
            1.0,
        )
        projected = points[:-1] + ratio[:, None] * delta
        distance = np.linalg.norm(projected - point_array, axis=-1)
        index = int(np.argmin(distance))
        progress = self.cumulative[lane_id][index] + ratio[index] * self.segment_lengths[lane_id][index]
        return float(progress), float(distance[index])

    def nearest_lane(self, point: Sequence[float]) -> tuple[Hashable, float, float]:
        candidates = [
            (lane_id, *self.project(lane_id, point)) for lane_id in self.points
        ]
        return min(candidates, key=lambda candidate: candidate[2])


@dataclass(frozen=True)
class InteractiveTrafficState:
    vehicle_id: Hashable
    lane_id: Hashable
    progress: float
    speed: float
    length: float = 4.5
    acceleration: float = 0.0
    source_lane: Hashable | None = None
    target_lane: Hashable | None = None
    lane_change_elapsed: float = 0.0
    lane_change_duration: float = 3.0
    lane_change_cooldown: float = 0.0


@dataclass(frozen=True)
class InteractiveTrafficPose:
    vehicle_id: Hashable
    position: tuple[float, float]
    yaw: float
    speed: float
    acceleration: float
    lane_id: Hashable
    lane_change_fraction: float


class PolylineIDMMOBILEngine:
    """Synchronous IDM+MOBIL engine with smooth polyline lane changes.

    This is H-BEHAVIOR-01: the paper publishes the longitudinal/lateral
    equations but not the state synchronization, lane-change interpolation,
    decision cadence, or overlap resolution needed by an executable system.
    """

    def __init__(
        self,
        network: PolylineLaneNetwork,
        *,
        feasible_lanes: Mapping[Hashable, Sequence[Hashable]] | None = None,
        idm: IDMParameters | None = None,
        mobil: MOBILParameters | None = None,
        lane_change_duration: float = 3.0,
        lane_change_cooldown: float = 2.0,
        minimum_acceleration: float = -6.0,
        maximum_acceleration: float = 3.0,
        hard_brake_deceleration: float = 6.0,
        overlap_buffer: float = 0.1,
    ) -> None:
        if lane_change_duration <= 0:
            raise ValueError("lane_change_duration must be positive")
        self.network = network
        self.idm = IDMParameters() if idm is None else idm
        self.mobil = MOBILParameters() if mobil is None else mobil
        self.longitudinal = IDMMOBILEngine(idm=self.idm, mobil=self.mobil)
        lane_ids = list(network.points)
        if feasible_lanes is None:
            feasible_lanes = {
                lane_id: tuple(
                    candidate
                    for candidate in lane_ids
                    if candidate == lane_id
                    or (
                        isinstance(candidate, int)
                        and isinstance(lane_id, int)
                        and abs(candidate - lane_id) == 1
                    )
                )
                for lane_id in lane_ids
            }
        self.feasible_lanes = {key: tuple(value) for key, value in feasible_lanes.items()}
        self.lane_change_duration = lane_change_duration
        self.lane_change_cooldown = lane_change_cooldown
        self.minimum_acceleration = minimum_acceleration
        self.maximum_acceleration = maximum_acceleration
        self.hard_brake_deceleration = hard_brake_deceleration
        self.overlap_buffer = overlap_buffer
        self.last_terminal_vehicle_ids: set[Hashable] = set()

    @staticmethod
    def _effective_lane(state: InteractiveTrafficState) -> Hashable:
        return state.target_lane if state.target_lane is not None else state.lane_id

    @staticmethod
    def _as_lane_state(state: InteractiveTrafficState) -> LaneVehicleState:
        return LaneVehicleState(
            vehicle_id=state.vehicle_id,
            lane_id=PolylineIDMMOBILEngine._effective_lane(state),
            position=state.progress,
            speed=state.speed,
            length=state.length,
            acceleration=state.acceleration,
        )

    def step(
        self,
        states: Sequence[InteractiveTrafficState],
        dt: float,
        *,
        ego: InteractiveTrafficState | None = None,
        hard_brake_ids: Sequence[Hashable] = (),
        allow_lane_changes: bool = True,
    ) -> list[InteractiveTrafficState]:
        if dt <= 0:
            raise ValueError("dt must be positive")
        lane_states = [self._as_lane_state(state) for state in states]
        if ego is not None:
            lane_states.append(self._as_lane_state(ego))
        hard_brake = set(hard_brake_ids)
        terminal_vehicle_ids: set[Hashable] = set()
        output = []
        for state in states:
            active_change = state.target_lane is not None
            current_lane = self._effective_lane(state)
            target_lane = current_lane
            if allow_lane_changes and not active_change and state.lane_change_cooldown <= 0.0:
                subject = self._as_lane_state(state)
                target_lane = self.longitudinal.choose_lane(
                    subject,
                    lane_states,
                    self.feasible_lanes.get(subject.lane_id, (subject.lane_id,)),
                )
                active_change = target_lane != state.lane_id

            subject = replace(self._as_lane_state(state), lane_id=target_lane)
            leader, _ = self.longitudinal._neighbors(lane_states, subject, target_lane)
            acceleration = idm_acceleration(subject, leader, self.idm)
            if state.vehicle_id in hard_brake:
                acceleration = min(acceleration, -self.hard_brake_deceleration)
            acceleration = float(
                np.clip(acceleration, self.minimum_acceleration, self.maximum_acceleration)
            )
            speed = max(0.0, state.speed + acceleration * dt)
            progress_unclipped = (
                state.progress + state.speed * dt + 0.5 * acceleration * dt * dt
            )
            lane_length = self.network.length(target_lane)
            reached_terminal = progress_unclipped >= lane_length - 1e-9
            progress = float(np.clip(progress_unclipped, 0.0, lane_length))
            if reached_terminal:
                # H-BEHAVIOR-03: a finite reconstructed polyline is not a
                # closed loop.  Never retain non-zero velocity at its clamped
                # endpoint; the owning simulator retires the vehicle instead
                # of letting overlap resolution teleport it backwards.
                speed = 0.0
                acceleration = min(acceleration, 0.0)
                terminal_vehicle_ids.add(state.vehicle_id)

            source_lane = state.source_lane
            change_target = state.target_lane
            elapsed = state.lane_change_elapsed
            cooldown = max(0.0, state.lane_change_cooldown - dt)
            lane_id = state.lane_id
            if active_change:
                if change_target is None:
                    source_lane = state.lane_id
                    change_target = target_lane
                    elapsed = 0.0
                elapsed += dt
                if elapsed >= state.lane_change_duration:
                    lane_id = change_target
                    source_lane = None
                    change_target = None
                    elapsed = 0.0
                    cooldown = self.lane_change_cooldown

            output.append(
                replace(
                    state,
                    lane_id=lane_id,
                    progress=progress,
                    speed=speed,
                    acceleration=acceleration,
                    source_lane=source_lane,
                    target_lane=change_target,
                    lane_change_elapsed=elapsed,
                    lane_change_duration=self.lane_change_duration,
                    lane_change_cooldown=cooldown,
                )
            )

        # H-BEHAVIOR-02: synchronous post-step gap projection. The paper says
        # global synchronization prevents overlap but gives no solver. Clamp a
        # follower behind the next centre by the two half-lengths plus 10 cm.
        corrected = {state.vehicle_id: state for state in output}
        for lane_id in self.network.points:
            lane = sorted(
                [state for state in output if self._effective_lane(state) == lane_id],
                key=lambda item: item.progress,
                reverse=True,
            )
            for leader, follower in zip(lane, lane[1:]):
                maximum = (
                    corrected[leader.vehicle_id].progress
                    - 0.5 * (leader.length + follower.length)
                    - self.overlap_buffer
                )
                current = corrected[follower.vehicle_id]
                if current.progress > maximum:
                    corrected[follower.vehicle_id] = replace(
                        current,
                        progress=max(0.0, maximum),
                        speed=min(current.speed, corrected[leader.vehicle_id].speed),
                    )
        self.last_terminal_vehicle_ids = terminal_vehicle_ids
        return [corrected[state.vehicle_id] for state in states]

    def pose(self, state: InteractiveTrafficState) -> InteractiveTrafficPose:
        if state.source_lane is None or state.target_lane is None:
            position, tangent = self.network.sample(state.lane_id, state.progress)
            fraction = 0.0
        else:
            start, start_tangent = self.network.sample(state.source_lane, state.progress)
            end, end_tangent = self.network.sample(state.target_lane, state.progress)
            linear = np.clip(
                state.lane_change_elapsed / max(state.lane_change_duration, 1e-9),
                0.0,
                1.0,
            )
            fraction = float(linear * linear * (3.0 - 2.0 * linear))
            position = (1.0 - fraction) * start + fraction * end
            tangent = (1.0 - fraction) * start_tangent + fraction * end_tangent
            tangent /= max(float(np.linalg.norm(tangent)), 1e-9)
        # HUGSIM planar state advances as da=-v*sin(yaw), db=v*cos(yaw).
        yaw = math.atan2(-float(tangent[0]), float(tangent[1]))
        return InteractiveTrafficPose(
            vehicle_id=state.vehicle_id,
            position=(float(position[0]), float(position[1])),
            yaw=yaw,
            speed=state.speed,
            acceleration=state.acceleration,
            lane_id=state.lane_id,
            lane_change_fraction=fraction,
        )
