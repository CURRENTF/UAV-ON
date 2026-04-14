#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

import airsim
import cv2
import numpy as np


def _parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_trajectory(path, stride, max_frames):
    frames = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                frames.append(json.loads(line))
    frames = frames[:: max(1, stride)]
    if max_frames > 0:
        frames = frames[:max_frames]
    if not frames:
        raise ValueError(f"No frames found in {path}")
    return frames


def _pose_from_frame(frame):
    state = frame["sensors"]["state"]
    position = state["position"]
    quaternion = state["quaternionr"]
    return airsim.Pose(
        airsim.Vector3r(float(position[0]), float(position[1]), float(position[2])),
        airsim.Quaternionr(
            x_val=float(quaternion[0]),
            y_val=float(quaternion[1]),
            z_val=float(quaternion[2]),
            w_val=float(quaternion[3]),
        ),
    )


def _decode_scene(response):
    data = response.image_data_uint8
    if data is None or len(data) == 0:
        raise RuntimeError("Empty image response")
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        image = array.reshape(response.height, response.width, 3)
    return image


def _resize_to_square(image, size):
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def _put_label(image, label):
    cv2.rectangle(image, (0, 0), (image.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        image,
        label,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _make_grid(images, labels, frame_info):
    size = min(min(image.shape[:2]) for image in images)
    tiles = []
    for image, label in zip(images, labels):
        tile = _resize_to_square(image, size)
        _put_label(tile, label)
        tiles.append(tile)
    while len(tiles) < 4:
        tiles.append(np.zeros_like(tiles[0]))
    top = np.concatenate(tiles[:2], axis=1)
    bottom = np.concatenate(tiles[2:4], axis=1)
    grid = np.concatenate([top, bottom], axis=0)
    cv2.rectangle(grid, (0, grid.shape[0] - 30), (grid.shape[1], grid.shape[0]), (0, 0, 0), -1)
    cv2.putText(
        grid,
        frame_info,
        (8, grid.shape[0] - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return grid


def _open_writer(path, fps, frame_shape):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer


def main():
    parser = argparse.ArgumentParser(description="Render a UAV-ON trajectory.jsonl as four-view videos.")
    parser.add_argument("--trajectory", required=True, help="Path to log/trajectory.jsonl")
    parser.add_argument("--output", required=True, help="Output 2x2 mp4 path")
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30100)
    parser.add_argument("--vehicle-name", default="")
    parser.add_argument("--cameras", default="0,1,2,3")
    parser.add_argument("--camera-labels", default="front,left,right,down")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--settle-seconds", type=float, default=0.2)
    parser.add_argument("--individual", action="store_true", help="Also write one mp4 per camera")
    args = parser.parse_args()

    cameras = _parse_csv(args.cameras)
    labels = _parse_csv(args.camera_labels)
    if len(labels) != len(cameras):
        raise ValueError("--camera-labels must have the same length as --cameras")
    if len(cameras) > 4:
        raise ValueError("The 2x2 grid renderer supports at most four cameras")

    frames = _load_trajectory(args.trajectory, args.stride, args.max_frames)
    client = airsim.MultirotorClient(ip=args.ip, port=args.port, timeout_value=args.timeout)
    client.confirmConnection()
    vehicles = client.listVehicles()
    vehicle_name = args.vehicle_name or (vehicles[0] if vehicles else "")
    client.enableApiControl(True, vehicle_name=vehicle_name)
    client.armDisarm(True, vehicle_name=vehicle_name)

    output = Path(args.output)
    individual_writers = {}
    grid_writer = None

    try:
        for idx, frame in enumerate(frames):
            pose = _pose_from_frame(frame)
            client.simPause(False)
            client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=vehicle_name)
            time.sleep(args.settle_seconds)
            requests = [
                airsim.ImageRequest(camera, airsim.ImageType.Scene, pixels_as_float=False, compress=True)
                for camera in cameras
            ]
            responses = client.simGetImages(requests=requests)
            client.simPause(True)
            if len(responses) != len(cameras):
                raise RuntimeError(f"Expected {len(cameras)} images, got {len(responses)}")

            images = [_decode_scene(response) for response in responses]
            info = (
                f"frame={frame.get('frame', idx)} "
                f"action={frame.get('action')} "
                f"step={frame.get('steps_size')} "
                f"collision={frame.get('is_collision')}"
            )
            grid = _make_grid(images, labels, info)

            if grid_writer is None:
                grid_writer = _open_writer(output, args.fps, grid.shape)
                if args.individual:
                    for camera, label, image in zip(cameras, labels, images):
                        stem = f"{output.stem}_camera{camera}_{label}.mp4"
                        individual_writers[camera] = _open_writer(output.with_name(stem), args.fps, image.shape)

            grid_writer.write(grid)
            for camera, image in zip(cameras, images):
                if camera in individual_writers:
                    individual_writers[camera].write(image)

            if idx % 10 == 0 or idx == len(frames) - 1:
                print(f"rendered {idx + 1}/{len(frames)} frames", flush=True)
    finally:
        if grid_writer is not None:
            grid_writer.release()
        for writer in individual_writers.values():
            writer.release()
        try:
            client.simPause(False)
        except Exception:
            pass

    print(f"wrote {output}")
    if individual_writers:
        print(f"wrote individual camera videos next to {output}")


if __name__ == "__main__":
    main()
