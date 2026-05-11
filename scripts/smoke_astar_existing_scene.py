#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import airsim
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

_ORIGINAL_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]
from model_wrapper.AStarOracle import AStarOracle
sys.argv = _ORIGINAL_ARGV


def _pose_from_dataset(item):
    position = item["start_pose"]["start_position"]
    quaternion = item["start_pose"]["start_quaternionr"]
    return airsim.Pose(
        airsim.Vector3r(float(position[0]), float(position[1]), float(position[2])),
        airsim.Quaternionr(
            x_val=float(quaternion[0]),
            y_val=float(quaternion[1]),
            z_val=float(quaternion[2]),
            w_val=float(quaternion[3]),
        ),
    )


def _pose_from_state(client, vehicle_name):
    state = client.getMultirotorState(vehicle_name=vehicle_name).kinematics_estimated
    return airsim.Pose(
        airsim.Vector3r(
            float(state.position.x_val),
            float(state.position.y_val),
            float(state.position.z_val),
        ),
        airsim.Quaternionr(
            x_val=float(state.orientation.x_val),
            y_val=float(state.orientation.y_val),
            z_val=float(state.orientation.z_val),
            w_val=float(state.orientation.w_val),
        ),
    )


def _frame(client, frame_index, action, step_size, vehicle_name, target_pos):
    state = client.getMultirotorState(vehicle_name=vehicle_name)
    pose = state.kinematics_estimated
    collision = client.simGetCollisionInfo(vehicle_name=vehicle_name)
    position = pose.position
    orientation = pose.orientation
    curr = np.array([position.x_val, position.y_val, position.z_val])
    dist = np.linalg.norm(curr - np.array(target_pos))
    return {
        "frame": frame_index,
        "is_collision": bool(collision.has_collided),
        "action": action,
        "steps_size": float(step_size),
        "move_distance": 0.0,
        "distance_to_end": float(dist),
        "sensors": {
            "state": {
                "position": [position.x_val, position.y_val, position.z_val],
                "quaternionr": [
                    orientation.x_val,
                    orientation.y_val,
                    orientation.z_val,
                    orientation.w_val,
                ],
            }
        },
    }


def _next_pose(current_pose, action, step_size):
    current_position = np.array(
        [
            current_pose.position.x_val,
            current_pose.position.y_val,
            current_pose.position.z_val,
        ],
        dtype=np.float32,
    )
    current_orientation = current_pose.orientation
    pitch, roll, yaw = airsim.to_eularian_angles(current_orientation)

    if action == "forward":
        unit = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
        new_position = current_position + unit * float(step_size)
        new_orientation = current_orientation
        fly_type = "move"
    elif action == "ascend":
        new_position = current_position + np.array([0.0, 0.0, -float(step_size)], dtype=np.float32)
        new_orientation = current_orientation
        fly_type = "move"
    elif action == "descend":
        new_position = current_position + np.array([0.0, 0.0, float(step_size)], dtype=np.float32)
        new_orientation = current_orientation
        fly_type = "move"
    elif action == "rotl":
        new_orientation = airsim.to_quaternion(pitch, roll, yaw - math.radians(float(step_size)))
        new_position = current_position
        fly_type = "rotate"
    elif action == "rotr":
        new_orientation = airsim.to_quaternion(pitch, roll, yaw + math.radians(float(step_size)))
        new_position = current_position
        fly_type = "rotate"
    else:
        new_position = current_position
        new_orientation = current_orientation
        fly_type = "none"

    return airsim.Pose(
        airsim.Vector3r(float(new_position[0]), float(new_position[1]), float(new_position[2])),
        new_orientation,
    ), fly_type


def _move(client, action, step_size, vehicle_name, execution_mode):
    if action == "stop":
        return
    current_pose = _pose_from_state(client, vehicle_name)
    next_pose, fly_type = _next_pose(current_pose, action, step_size)
    if execution_mode == "kinematic":
        if fly_type != "none":
            client.simPause(True)
            client.simSetVehiclePose(next_pose, ignore_collision=True, vehicle_name=vehicle_name)
        return

    client.simPause(False)
    if fly_type == "move":
        client.moveToPositionAsync(
            next_pose.position.x_val,
            next_pose.position.y_val,
            next_pose.position.z_val,
            velocity=1,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            vehicle_name=vehicle_name,
        ).join()
    elif fly_type == "rotate":
        _pitch, _roll, yaw = airsim.to_eularian_angles(next_pose.orientation)
        client.rotateToYawAsync(math.degrees(yaw), vehicle_name=vehicle_name).join()
    client.simPause(True)


def main():
    parser = argparse.ArgumentParser(description="Run A* oracle against an already-open AirSim scene.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30100)
    parser.add_argument("--max-actions", type=int, default=8)
    parser.add_argument("--voxel-resolution", type=float, default=2.0)
    parser.add_argument("--astar-turn-cooldown-after", type=int, default=0)
    parser.add_argument("--astar-turn-cooldown-steps", type=int, default=0)
    parser.add_argument(
        "--execution-mode",
        choices=("kinematic", "physics"),
        default="kinematic",
        help="Use kinematic pose updates for faithful A* path visualization, or physics for SimpleFlight diagnostics.",
    )
    args = parser.parse_args()

    with open(args.dataset, "r") as handle:
        item = json.load(handle)[0]
    item = dict(item)
    item["task_id"] = item.get("episode_id", 0)
    # Move drone up by 2 meters to avoid "occupied start voxel" error
    item["start_pose"]["start_position"][2] -= 2.0

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(output_dir, 0o755)
    voxel_dir = output_dir / "voxels"
    voxel_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(voxel_dir, 0o777)
    trajectory_path = output_dir / "log" / "trajectory.jsonl"
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)

    client = airsim.MultirotorClient(ip=args.ip, port=args.port, timeout_value=300)
    print("connecting to AirSim", flush=True)
    client.confirmConnection()
    print("confirmed AirSim connection", flush=True)
    vehicles = client.listVehicles()
    print(f"vehicles={vehicles}", flush=True)
    vehicle_name = vehicles[0] if vehicles else "Drone_1"

    print("setting initial pose", flush=True)
    client.enableApiControl(True, vehicle_name=vehicle_name)
    client.armDisarm(True, vehicle_name=vehicle_name)
    client.simPause(True)
    client.simSetVehiclePose(_pose_from_dataset(item), ignore_collision=True, vehicle_name=vehicle_name)
    if args.execution_mode == "physics":
        client.simPause(False)
        time.sleep(0.5)
        client.simPause(True)
    print("initial pose ready", flush=True)

    target_pos = item["pose"][0] if isinstance(item["pose"][0], list) else item["pose"]

    fake_env = SimpleNamespace(
        machines_info=[{"open_scenes": [item["map_name"]]}],
        simulator_tool=SimpleNamespace(airsim_clients=[[client]]),
    )
    oracle_args = SimpleNamespace(
        is_fixed=False,
        eval_save_path=str(output_dir),
        astar_voxel_dir=str(voxel_dir),
        astar_keep_voxels=True,
        astar_voxel_resolution=args.voxel_resolution,
        astar_voxel_margin_xy=10.0,
        astar_voxel_margin_z=8.0,
        astar_min_extent_xy=30.0,
        astar_min_extent_z=24.0,
        astar_max_voxels=500000,
        astar_target_search_radius=10.0,
        astar_max_goal_candidates=32,
        astar_max_move_voxels=1,
        astar_turn_cooldown_after=args.astar_turn_cooldown_after,
        astar_turn_cooldown_steps=args.astar_turn_cooldown_steps,
    )
    oracle = AStarOracle(batch_size=1, args=oracle_args)
    print("planning with AStarOracle", flush=True)
    oracle.prepare_batch(env=fake_env, batch=[item])
    print(oracle.plan_summaries[0], flush=True)
    if "failed" in oracle.plan_summaries[0]:
        raise RuntimeError(oracle.plan_summaries[0])
    client.simPause(True)
    client.simSetVehiclePose(_pose_from_dataset(item), ignore_collision=True, vehicle_name=vehicle_name)

    frames = [_frame(client, 0, "start", 0.0, vehicle_name, target_pos)]
    for step in range(args.max_actions):
        actions, step_sizes, dones = oracle.run([0], fixed=False)
        action, step_size = actions[0], step_sizes[0]
        print(f"step={step} action={action} step_size={step_size}", flush=True)
        if dones[0]:
            frames.append(_frame(client, len(frames), action, step_size, vehicle_name, target_pos))
            break
        _move(client, action, step_size, vehicle_name, args.execution_mode)
        frames.append(_frame(client, len(frames), action, step_size, vehicle_name, target_pos))

    with trajectory_path.open("w") as handle:
        for frame in frames:
            handle.write(json.dumps(frame) + "\n")
    print(f"wrote {trajectory_path}", flush=True)


if __name__ == "__main__":
    main()
