from __future__ import annotations

import heapq
import math
import os
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional, Sequence

import airsim
import numpy as np

from model_wrapper.base_model import BaseModelWrapper
from utils.logger import logger


def _debug(message: str) -> None:
    logger.debug(message)
    print(f"[AStarOracle] {message}", flush=True)


def _wrap_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def _yaw_degrees_from_quaternion(quaternion: Sequence[float]) -> float:
    x, y, z, w = [float(value) for value in quaternion]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def _sanitize_path_segment(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value).strip("_") or "unknown"


def _target_points(item: dict[str, Any]) -> np.ndarray:
    coords_value = item.get("object_position")
    if coords_value is None:
        coords_value = item.get("pose")
    coords = np.asarray(coords_value, dtype=np.float32)
    if coords.ndim == 1:
        coords = coords[None, :]
    if coords.ndim != 2 or coords.shape[1] < 3:
        raise ValueError(f"UAV-ON target coordinates must be Nx3, got shape {coords.shape}")
    return coords[:, :3]


def _read_binvox_occupancy(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = handle.readline().strip()
        if header != b"#binvox 1":
            raise ValueError(f"Unsupported binvox header in {path}: {header!r}")

        raw_dims: Optional[tuple[int, int, int]] = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"binvox file ended before data block: {path}")
            stripped = line.strip()
            if stripped == b"data":
                break
            parts = stripped.split()
            if parts and parts[0] == b"dim":
                raw_dims = (int(parts[1]), int(parts[2]), int(parts[3]))

        if raw_dims is None:
            raise ValueError(f"binvox file does not contain dimensions: {path}")
        encoded = np.frombuffer(handle.read(), dtype=np.uint8)

    if encoded.size % 2:
        encoded = encoded[:-1]

    values = encoded[0::2].astype(bool)
    counts = encoded[1::2].astype(np.int64)
    flat = np.repeat(values, counts)

    # AirSim writes binvox dimensions as X, Z, Y and indexes cells as
    # i + nx * (k + nz * j). Convert to a direct X, Y, Z occupancy array.
    nx, nz, ny = raw_dims
    expected = nx * ny * nz
    if flat.size < expected:
        flat = np.pad(flat, (0, expected - flat.size), constant_values=False)
    elif flat.size > expected:
        flat = flat[:expected]

    return flat.reshape((ny, nz, nx)).transpose(2, 0, 1).copy()


def _world_to_grid_index(
    position: np.ndarray,
    *,
    center: np.ndarray,
    resolution: float,
    shape: Sequence[int],
) -> tuple[int, int, int]:
    offset = np.asarray(position, dtype=np.float32) - np.asarray(center, dtype=np.float32)
    # AirSim poses are NED; the binvox z axis follows Unreal's up axis.
    offset[2] *= -1.0
    index = np.rint(offset / resolution + np.asarray(shape, dtype=np.float32) / 2.0).astype(np.int64)
    clipped = np.clip(index, 0, np.asarray(shape, dtype=np.int64) - 1)
    return int(clipped[0]), int(clipped[1]), int(clipped[2])


def _grid_index_to_world_position(
    index: tuple[int, int, int],
    *,
    center: np.ndarray,
    resolution: float,
    shape: Sequence[int],
) -> list[float]:
    offset = (np.asarray(index, dtype=np.float32) - np.asarray(shape, dtype=np.float32) / 2.0) * resolution
    offset[2] *= -1.0
    position = np.asarray(center, dtype=np.float32) + offset
    return [float(value) for value in position]


def _free_indices_near(
    occupied: np.ndarray,
    preferred: tuple[int, int, int],
    *,
    max_radius_cells: int,
    max_candidates: int,
) -> list[tuple[int, int, int]]:
    shape = occupied.shape
    px, py, pz = preferred
    candidates: list[tuple[int, tuple[int, int, int]]] = []

    for radius in range(max(0, max_radius_cells) + 1):
        x0, x1 = max(0, px - radius), min(shape[0], px + radius + 1)
        y0, y1 = max(0, py - radius), min(shape[1], py + radius + 1)
        z0, z1 = max(0, pz - radius), min(shape[2], pz + radius + 1)
        for x in range(x0, x1):
            for y in range(y0, y1):
                for z in range(z0, z1):
                    if max(abs(x - px), abs(y - py), abs(z - pz)) != radius:
                        continue
                    if occupied[x, y, z]:
                        continue
                    distance = (x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2
                    candidates.append((distance, (x, y, z)))

        if len(candidates) >= max_candidates:
            break

    candidates.sort(key=lambda item: item[0])
    return [candidate for _distance, candidate in candidates[:max_candidates]]


def _astar_grid_path_to_any(
    occupied: np.ndarray,
    start: tuple[int, int, int],
    goals: Sequence[tuple[int, int, int]],
) -> tuple[Optional[list[tuple[int, int, int]]], Optional[tuple[int, int, int]]]:
    goal_set = {goal for goal in goals if not occupied[goal]}
    if not goal_set or occupied[start]:
        return None, None
    if start in goal_set:
        return [start], start

    goal_list = list(goal_set)
    shape = occupied.shape
    offsets = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

    def heuristic(index: tuple[int, int, int]) -> float:
        best = min(
            (index[0] - goal[0]) ** 2
            + (index[1] - goal[1]) ** 2
            + (index[2] - goal[2]) ** 2
            for goal in goal_list
        )
        return math.sqrt(best)

    open_set: list[tuple[float, int, tuple[int, int, int]]] = []
    heapq.heappush(open_set, (heuristic(start), 0, start))
    came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    g_score: dict[tuple[int, int, int], float] = {start: 0.0}
    closed: set[tuple[int, int, int]] = set()
    counter = 1

    while open_set:
        _priority, _counter, current = heapq.heappop(open_set)
        if current in closed:
            continue
        if current in goal_set:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, path[-1]

        closed.add(current)
        current_score = g_score[current]
        for dx, dy, dz in offsets:
            neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
            if (
                neighbor[0] < 0
                or neighbor[0] >= shape[0]
                or neighbor[1] < 0
                or neighbor[1] >= shape[1]
                or neighbor[2] < 0
                or neighbor[2] >= shape[2]
                or occupied[neighbor]
                or neighbor in closed
            ):
                continue

            tentative = current_score + 1.0
            if tentative >= g_score.get(neighbor, float("inf")):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative
            heapq.heappush(open_set, (tentative + heuristic(neighbor), counter, neighbor))
            counter += 1

    return None, None


def _compress_path_to_actions(
    path: Sequence[tuple[int, int, int]],
    *,
    resolution: float,
    start_quaternion: Sequence[float],
    max_move_voxels: int,
) -> tuple[list[str], list[float]]:
    if len(path) <= 1:
        return ["stop"], [0.0]

    actions: list[str] = []
    step_sizes: list[float] = []
    yaw = _yaw_degrees_from_quaternion(start_quaternion)
    axis_yaws = {
        (0, 1): 0.0,
        (0, -1): 180.0,
        (1, 1): 90.0,
        (1, -1): -90.0,
    }
    max_chunk_steps = max(1, int(max_move_voxels))

    run_axis: Optional[int] = None
    run_sign = 0
    run_steps = 0

    def flush_run() -> None:
        nonlocal yaw, run_axis, run_sign, run_steps
        if run_axis is None or run_steps <= 0:
            return

        if run_axis == 2:
            remaining = run_steps
            while remaining > 0:
                chunk = min(max_chunk_steps, remaining)
                actions.append("ascend" if run_sign > 0 else "descend")
                step_sizes.append(float(chunk) * float(resolution))
                remaining -= chunk
        else:
            target_yaw = axis_yaws[(run_axis, run_sign)]
            delta_yaw = _wrap_degrees(target_yaw - yaw)
            if abs(delta_yaw) > 1e-3:
                actions.append("rotr" if delta_yaw > 0 else "rotl")
                step_sizes.append(abs(delta_yaw))
                yaw = _wrap_degrees(yaw + delta_yaw)

            remaining = run_steps
            while remaining > 0:
                chunk = min(max_chunk_steps, remaining)
                actions.append("forward")
                step_sizes.append(float(chunk) * float(resolution))
                remaining -= chunk

        run_axis = None
        run_sign = 0
        run_steps = 0

    for previous, current in zip(path, path[1:]):
        delta = np.asarray(current, dtype=np.int64) - np.asarray(previous, dtype=np.int64)
        nonzero = np.flatnonzero(delta)
        if len(nonzero) != 1:
            flush_run()
            continue
        axis = int(nonzero[0])
        sign = int(delta[axis])
        if run_axis == axis and run_sign == sign:
            run_steps += 1
        else:
            flush_run()
            run_axis = axis
            run_sign = sign
            run_steps = 1

    flush_run()
    actions.append("stop")
    step_sizes.append(0.0)
    return actions, step_sizes


class AStarOracle(BaseModelWrapper):
    """3D A* expert policy for UAV-ON training trajectory generation."""

    def __init__(self, *, batch_size: int, args: Any) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.args = args
        self.queues: list[deque[tuple[str, float]]] = [deque() for _ in range(batch_size)]
        self.plan_summaries: list[str] = ["" for _ in range(batch_size)]

        configured_voxel_dir = getattr(args, "astar_voxel_dir", None)
        voxel_dir = configured_voxel_dir or os.path.join(args.eval_save_path, "astar_voxels")
        self.voxel_dir = Path(voxel_dir).expanduser().resolve()
        if bool(getattr(args, "astar_keep_voxels", False)):
            self.voxel_dir.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(self.voxel_dir, 0o777)
            except OSError:
                pass

    def _client_for_batch_index(self, env: Any, batch_index: int) -> Any:
        cursor = 0
        for machine_index, machine in enumerate(env.machines_info):
            scene_count = len(machine["open_scenes"])
            if batch_index < cursor + scene_count:
                return env.simulator_tool.airsim_clients[machine_index][batch_index - cursor]
            cursor += scene_count
        raise IndexError(f"batch index {batch_index} is outside active UAV-ON clients")

    def _voxel_output_path(self, item: dict[str, Any], batch_index: int) -> tuple[Path, bool]:
        map_name = _sanitize_path_segment(str(item.get("map_name", "unknown")))
        episode_id = _sanitize_path_segment(str(item.get("task_id", item.get("episode_id", batch_index))))
        if bool(getattr(self.args, "astar_keep_voxels", False)):
            return self.voxel_dir / f"{map_name}_{episode_id}_{batch_index}.binvox", False

        handle = tempfile.NamedTemporaryFile(prefix="uav_on_astar_", suffix=".binvox", delete=False)
        handle.close()
        return Path(handle.name), True

    def _list_vehicle_name(self, client: Any) -> str:
        try:
            vehicles = client.listVehicles()
            if vehicles:
                return str(vehicles[0])
        except Exception:
            pass
        return "Drone_1"

    def _sim_get_vehicle_pose(self, client: Any, vehicle_name: str) -> airsim.Pose:
        try:
            return client.simGetVehiclePose(vehicle_name=vehicle_name)
        except TypeError:
            return client.simGetVehiclePose()

    def _sim_set_vehicle_pose(self, client: Any, pose: airsim.Pose, vehicle_name: str) -> None:
        try:
            client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=vehicle_name)
        except TypeError:
            client.simSetVehiclePose(pose=pose, ignore_collision=True)

    def _drain_collision(self, client: Any, vehicle_name: str) -> None:
        for _ in range(2):
            try:
                client.simGetCollisionInfo(vehicle_name=vehicle_name)
            except TypeError:
                try:
                    client.simGetCollisionInfo()
                except Exception:
                    pass
            except Exception:
                pass

    def _create_occupancy_grid(
        self,
        *,
        client: Any,
        vehicle_name: str,
        center: np.ndarray,
        extent: np.ndarray,
        resolution: float,
        voxel_path: Path,
    ) -> np.ndarray:
        current_pose = self._sim_get_vehicle_pose(client, vehicle_name)
        parking_pose = airsim.Pose(
            airsim.Vector3r(
                float(center[0]),
                float(center[1]),
                float(center[2] - extent[2] - 1000.0),
            ),
            current_pose.orientation,
        )

        try:
            client.simPause(False)
            # The occupancy grid should describe the scene, not the drone body.
            self._sim_set_vehicle_pose(client, parking_pose, vehicle_name)
            time.sleep(0.2)
            client.simPause(True)
            ok = client.simCreateVoxelGrid(
                airsim.Vector3r(float(center[0]), float(center[1]), float(center[2])),
                int(round(float(extent[0]))),
                int(round(float(extent[1]))),
                int(round(float(extent[2]))),
                float(resolution),
                str(voxel_path),
            )
        finally:
            client.simPause(False)
            self._sim_set_vehicle_pose(client, current_pose, vehicle_name)
            time.sleep(0.2)
            self._drain_collision(client, vehicle_name)
            client.simPause(True)

        if not ok or not voxel_path.exists():
            raise RuntimeError(f"simCreateVoxelGrid failed: {voxel_path}")
        return _read_binvox_occupancy(voxel_path)

    def _plan_one(self, *, env: Any, batch_index: int, item: dict[str, Any]) -> tuple[list[str], list[float], str]:
        if bool(getattr(self.args, "is_fixed", False)):
            raise RuntimeError("AStarOracle requires --is_fixed false so 1-unit action step sizes are honored.")

        resolution = float(getattr(self.args, "astar_voxel_resolution", 1.0))
        if resolution <= 0:
            raise ValueError("--astar_voxel_resolution must be positive")
        if abs(resolution - 1.0) > 1e-6:
            logger.warning("AStarOracle is configured with resolution %.3f; the paper setup uses 1-unit grids.", resolution)

        start = np.asarray(item["start_pose"]["start_position"], dtype=np.float32)
        targets = _target_points(item)
        points = np.concatenate([start[None, :], targets], axis=0)

        margin = np.asarray(
            [
                float(getattr(self.args, "astar_voxel_margin_xy", 25.0)),
                float(getattr(self.args, "astar_voxel_margin_xy", 25.0)),
                float(getattr(self.args, "astar_voxel_margin_z", 20.0)),
            ],
            dtype=np.float32,
        )
        min_extent = np.asarray(
            [
                float(getattr(self.args, "astar_min_extent_xy", 100.0)),
                float(getattr(self.args, "astar_min_extent_xy", 100.0)),
                float(getattr(self.args, "astar_min_extent_z", 60.0)),
            ],
            dtype=np.float32,
        )
        bbox_min = points.min(axis=0) - margin
        bbox_max = points.max(axis=0) + margin
        center = (bbox_min + bbox_max) / 2.0
        extent = np.maximum(bbox_max - bbox_min, min_extent)
        cell_counts = np.maximum(3, np.ceil(extent / resolution).astype(np.int64))
        extent = cell_counts.astype(np.float32) * resolution
        max_voxels = int(getattr(self.args, "astar_max_voxels", 4000000))
        voxel_count = int(np.prod(cell_counts))
        if voxel_count > max_voxels:
            raise RuntimeError(
                f"A* voxel crop is too large for episode {item.get('task_id')}: "
                f"cells={tuple(int(v) for v in cell_counts)} count={voxel_count} max={max_voxels}"
            )

        episode_id = str(item.get("task_id", item.get("episode_id", batch_index)))
        map_name = str(item.get("map_name", "unknown"))
        _debug(
            f"plan start batch={batch_index} map={map_name} episode={episode_id} "
            f"center={center.tolist()} extent={extent.tolist()} resolution={resolution} cells={cell_counts.tolist()}"
        )

        voxel_path, should_delete = self._voxel_output_path(item, batch_index)
        try:
            client = self._client_for_batch_index(env, batch_index)
            vehicle_name = self._list_vehicle_name(client)
            occupied = self._create_occupancy_grid(
                client=client,
                vehicle_name=vehicle_name,
                center=center,
                extent=extent,
                resolution=resolution,
                voxel_path=voxel_path,
            )
            shape = occupied.shape
            start_index = _world_to_grid_index(start, center=center, resolution=resolution, shape=shape)
            target_indices = [
                _world_to_grid_index(target, center=center, resolution=resolution, shape=shape)
                for target in targets
            ]

            if occupied[start_index]:
                raise RuntimeError(
                    f"A* start voxel is occupied: map={map_name} episode={episode_id} start_index={start_index}"
                )

            max_radius_cells = max(
                1,
                int(math.ceil(float(getattr(self.args, "astar_target_search_radius", 20.0)) / resolution)),
            )
            max_candidates = max(1, int(getattr(self.args, "astar_max_goal_candidates", 128)))
            candidates: list[tuple[int, int, int]] = []
            seen_candidates: set[tuple[int, int, int]] = set()
            for target_index in target_indices:
                for candidate in _free_indices_near(
                    occupied,
                    target_index,
                    max_radius_cells=max_radius_cells,
                    max_candidates=max_candidates,
                ):
                    if candidate not in seen_candidates:
                        seen_candidates.add(candidate)
                        candidates.append(candidate)
                    if len(candidates) >= max_candidates:
                        break
                if len(candidates) >= max_candidates:
                    break

            if not candidates:
                raise RuntimeError(
                    f"A* could not find a free target-adjacent voxel: map={map_name} episode={episode_id}"
                )

            path, goal_index = _astar_grid_path_to_any(occupied, start_index, candidates)
            if path is None or goal_index is None:
                raise RuntimeError(
                    f"A* could not find a traversable path: map={map_name} episode={episode_id} "
                    f"start={start_index} candidates={len(candidates)}"
                )

            actions, step_sizes = _compress_path_to_actions(
                path,
                resolution=resolution,
                start_quaternion=item["start_pose"]["start_quaternionr"],
                max_move_voxels=int(getattr(self.args, "astar_max_move_voxels", 1)),
            )
            preview = [
                _grid_index_to_world_position(index, center=center, resolution=resolution, shape=shape)
                for index in path[:6]
            ]
            summary = (
                f"3D A* oracle, map={map_name}, episode={episode_id}, path_len={len(path)}, "
                f"actions={len(actions)}, start_index={start_index}, goal_index={goal_index}"
            )
            _debug(f"plan done {summary} preview={preview} actions_preview={actions[:10]}")
            return actions, step_sizes, summary
        finally:
            if should_delete:
                try:
                    os.unlink(voxel_path)
                except OSError:
                    pass

    def prepare_batch(self, *, env: Any, batch: Sequence[dict[str, Any]]) -> None:
        self.queues = []
        self.plan_summaries = []
        _debug(f"prepare batch size={len(batch)}")
        for batch_index, item in enumerate(batch):
            try:
                actions, step_sizes, summary = self._plan_one(env=env, batch_index=batch_index, item=item)
            except Exception as exc:
                episode_id = str(item.get("task_id", item.get("episode_id", batch_index)))
                logger.exception("AStarOracle planning failed for episode %s", episode_id)
                print(f"[AStarOracle] plan failed episode={episode_id}: {exc}", flush=True)
                actions, step_sizes = ["stop"], [0.0]
                summary = f"3D A* oracle failed, episode={episode_id}, error={exc}"

            self.queues.append(deque(zip(actions, step_sizes)))
            self.plan_summaries.append(summary)
        _debug(f"prepared queue_lengths={[len(queue) for queue in self.queues]}")

    def prepare_inputs(self, episodes: Sequence[Sequence[dict[str, Any]]], fixed: bool = False):
        inputs = list(range(len(episodes)))
        prompts = self.plan_summaries[: len(episodes)]
        if len(prompts) < len(episodes):
            prompts.extend(["3D A* oracle plan unavailable"] * (len(episodes) - len(prompts)))
        return inputs, prompts

    def run(self, inputs, fixed: bool = False, prompt_info_list=None):
        if fixed:
            raise RuntimeError("AStarOracle requires --is_fixed false.")

        actions: list[str] = []
        step_sizes: list[float] = []
        dones: list[bool] = []
        for batch_index in inputs:
            if batch_index >= len(self.queues) or not self.queues[batch_index]:
                action, step_size = "stop", 0.0
            else:
                action, step_size = self.queues[batch_index].popleft()
            actions.append(action)
            step_sizes.append(float(step_size))
            dones.append(action == "stop")
        return actions, step_sizes, dones
