#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _load_positions(path):
    frames = []
    with open(path, "r") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                frames.append(
                    {
                        "position": np.asarray(item["sensors"]["state"]["position"], dtype=np.float32),
                        "action": item.get("action"),
                        "step": item.get("steps_size"),
                    }
                )
    if not frames:
        raise ValueError(f"No trajectory frames in {path}")
    return frames


def _project(points, bounds, size, pad):
    min_xy, max_xy = bounds
    span = np.maximum(max_xy - min_xy, 1e-6)
    xy = (points[:, :2] - min_xy) / span
    xy[:, 1] = 1.0 - xy[:, 1]
    return (xy * (size - 2 * pad) + pad).astype(np.int32)


def main():
    parser = argparse.ArgumentParser(description="Render a simple top-down UAV trajectory mp4.")
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--size", type=int, default=900)
    parser.add_argument("--pad", type=int, default=80)
    args = parser.parse_args()

    frames = _load_positions(args.trajectory)
    with open(args.dataset, "r") as handle:
        episode = json.load(handle)[0]
    target = np.asarray(episode["pose"][0], dtype=np.float32)
    start = np.asarray(episode["start_pose"]["start_position"], dtype=np.float32)
    positions = np.asarray([frame["position"] for frame in frames], dtype=np.float32)
    all_points = np.concatenate([positions, start[None, :], target[None, :]], axis=0)
    min_xy = all_points[:, :2].min(axis=0) - 5.0
    max_xy = all_points[:, :2].max(axis=0) + 5.0
    bounds = (min_xy, max_xy)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.size, args.size),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output}")

    try:
        projected_positions = _project(positions, bounds, args.size, args.pad)
        projected_start = _project(start[None, :], bounds, args.size, args.pad)[0]
        projected_target = _project(target[None, :], bounds, args.size, args.pad)[0]

        for idx, frame in enumerate(frames):
            canvas = np.full((args.size, args.size, 3), 245, dtype=np.uint8)
            cv2.rectangle(canvas, (args.pad, args.pad), (args.size - args.pad, args.size - args.pad), (200, 200, 200), 1)
            cv2.circle(canvas, tuple(projected_start), 9, (40, 150, 60), -1)
            cv2.circle(canvas, tuple(projected_target), 10, (40, 40, 220), -1)
            cv2.putText(canvas, "start", tuple(projected_start + np.array([12, -8])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 120, 40), 2)
            cv2.putText(canvas, "target", tuple(projected_target + np.array([12, -8])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 180), 2)
            if idx > 0:
                cv2.polylines(canvas, [projected_positions[: idx + 1]], False, (220, 120, 20), 3, cv2.LINE_AA)
            cv2.circle(canvas, tuple(projected_positions[idx]), 8, (20, 80, 230), -1)
            text = f"frame={idx} action={frame['action']} step={frame['step']}"
            cv2.putText(canvas, text, (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()

    print(f"wrote {output}")


if __name__ == "__main__":
    main()
