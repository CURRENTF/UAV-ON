#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

import cv2


SYSTEM_PROMPT = "You are a UAV navigation policy. Return only one action name."
VALID_ACTIONS = {"forward", "left", "right", "rotl", "rotr", "ascend", "descend", "stop"}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_trajectory(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
    if not rows:
        raise ValueError(f"No trajectory rows found in {path}")
    return rows


def _episode_number_from_path(path: Path) -> str | None:
    match = re.search(r"episode(\d+)", path.as_posix(), flags=re.IGNORECASE)
    if match:
        return match.group(1).lstrip("0") or "0"
    return None


def _scene_hint_from_path(path: Path) -> str:
    lowered = path.as_posix().lower()
    if "slum" in lowered:
        return "slum"
    if "citypark" in lowered or "city_park" in lowered:
        return "citypark"
    return ""


def _load_task_index(metadata_paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for path in metadata_paths:
        data = _load_json(path)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list of UAV-ON tasks in {path}")
        for item in data:
            episode_id = str(item.get("episode_id", "")).lstrip("0") or "0"
            copied = dict(item)
            copied["_metadata_path"] = str(path)
            index.setdefault(episode_id, []).append(copied)
    return index


def _select_task(trajectory_path: Path, task_index: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    episode_id = _episode_number_from_path(trajectory_path)
    if episode_id is None:
        raise ValueError(f"Could not infer episode id from {trajectory_path}")
    candidates = task_index.get(episode_id)
    if not candidates:
        raise ValueError(f"No metadata matched episode id {episode_id} for {trajectory_path}")
    scene_hint = _scene_hint_from_path(trajectory_path)
    if scene_hint:
        scene_candidates = [
            item for item in candidates if scene_hint in str(item.get("map_name", "")).replace("_", "").lower()
        ]
        if len(scene_candidates) == 1:
            return scene_candidates[0]
        if len(scene_candidates) > 1:
            candidates = scene_candidates
    if len(candidates) != 1:
        raise ValueError(f"Ambiguous metadata for {trajectory_path}: {candidates}")
    return candidates[0]


def _select_fourview_video(episode_dir: Path) -> Path:
    videos = sorted(episode_dir.glob("*4view*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No four-view mp4 found under {episode_dir}")

    def score(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        penalty = sum(10 for token in ("settle", "paused", "ideal") if token in name)
        if name == "4view.mp4":
            penalty -= 5
        return penalty, path.name

    return sorted(videos, key=score)[0]


def _task_instruction(task: dict[str, Any]) -> str:
    target_name = str(task.get("true_name") or task.get("object_name") or "target object").strip()
    size = str(task.get("size") or "").strip()
    description = str(task.get("description") or "").strip()
    parts = [f"Find and stop near the target object: {target_name}."]
    if size:
        parts.append(f"Target size: {size}.")
    if description:
        parts.append(f"Target description: {description}")
    return " ".join(parts)


def _user_prompt(instruction: str) -> str:
    return (
        "Current four-view observation is provided as one 2x2 image grid "
        "(front, left, right, down).\n"
        f"Task instruction:\n{instruction.strip()}\n"
        "Choose the last A* action for the current state. "
        "Allowed actions: forward, left, right, rotl, rotr, ascend, descend, stop.\n"
        "Return only the action name."
    )


def _read_frame(cap: cv2.VideoCapture, frame_index: int, video_path: Path, strip_overlay: bool) -> Any:
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"Could not read frame {frame_index} from {video_path}")
    if strip_overlay and frame.shape[0] > 30:
        frame = frame[: frame.shape[0] - 30, :, :]
    return frame


def export_dataset(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.astar_logs_dir)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    metadata_paths = sorted(root.glob(args.metadata_glob))
    if not metadata_paths:
        raise FileNotFoundError(f"No metadata matched {root / args.metadata_glob}")
    task_index = _load_task_index(metadata_paths)

    trajectory_paths = sorted(root.glob(args.trajectory_glob))
    if not trajectory_paths:
        raise FileNotFoundError(f"No trajectories matched {root / args.trajectory_glob}")

    rows = []
    skipped = []
    used_by_episode = []
    for trajectory_path in trajectory_paths:
        episode_dir = trajectory_path.parent.parent
        try:
            task = _select_task(trajectory_path, task_index)
            video_path = _select_fourview_video(episode_dir)
        except Exception as exc:
            if args.strict_metadata:
                raise
            skipped.append({"trajectory_path": str(trajectory_path), "reason": str(exc)})
            continue

        trajectory = _load_trajectory(trajectory_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open {video_path}")
        video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_frame_count and video_frame_count < len(trajectory):
            cap.release()
            raise ValueError(f"Video {video_path} has {video_frame_count} frames but trajectory has {len(trajectory)} rows")

        instruction = _task_instruction(task)
        used_rows = 0
        try:
            for frame_index, frame_row in enumerate(trajectory):
                frame = _read_frame(cap, frame_index, video_path, strip_overlay=not args.keep_render_overlay)
                if frame_index % args.frame_stride != 0:
                    continue
                if args.target_alignment == "next":
                    action_index = frame_index + 1
                    if action_index >= len(trajectory):
                        continue
                    action_row = trajectory[action_index]
                else:
                    action_index = frame_index
                    action_row = frame_row
                action = str(action_row.get("action", "")).strip()
                if action not in VALID_ACTIONS:
                    continue

                rel_image_path = Path("images") / episode_dir.name / f"{frame_index:06d}.jpg"
                image_path = output_dir / rel_image_path
                image_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(image_path), frame):
                    raise RuntimeError(f"Failed to write {image_path}")

                row = {
                    "id": f"{episode_dir.name}_{frame_index:06d}",
                    "image": str(rel_image_path),
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": str(rel_image_path)},
                                {"type": "text", "text": _user_prompt(instruction)},
                            ],
                        },
                        {"role": "assistant", "content": action},
                    ],
                    "instruction": instruction,
                    "action": action,
                    "episode_dir": str(episode_dir),
                    "episode_id": str(task.get("episode_id", "")),
                    "map_name": str(task.get("map_name", "")),
                    "trajectory_path": str(trajectory_path),
                    "fourview_video_path": str(video_path),
                    "frame_index": frame_index,
                    "target_action_frame_index": action_index,
                }
                rows.append(row)
                used_rows += 1
                if args.max_samples is not None and len(rows) >= args.max_samples:
                    break
        finally:
            cap.release()
        used_by_episode.append(
            {
                "episode_dir": str(episode_dir),
                "trajectory_path": str(trajectory_path),
                "fourview_video_path": str(video_path),
                "trajectory_rows": len(trajectory),
                "used_rows": used_rows,
            }
        )
        if args.max_samples is not None and len(rows) >= args.max_samples:
            break

    if not rows:
        raise ValueError("No Qwen3-VL UAV-ON action rows were exported")

    jsonl_path = output_dir / "train.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "output_dir": str(output_dir),
        "jsonl_path": str(jsonl_path),
        "num_rows": len(rows),
        "num_trajectories": len(used_by_episode),
        "actions": {action: sum(1 for row in rows if row["action"] == action) for action in sorted(VALID_ACTIONS)},
        "mean_prompt_chars": mean(len(row["messages"][1]["content"][1]["text"]) for row in rows),
        "target_alignment": args.target_alignment,
        "frame_stride": args.frame_stride,
        "strip_render_overlay": not args.keep_render_overlay,
        "skipped": skipped,
        "trajectories": used_by_episode,
    }
    (output_dir / "export_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Export UAV-ON A* four-view action data for Qwen3-VL SFT.")
    parser.add_argument("--astar_logs_dir", default="../UAV-ON/astar_logs")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metadata_glob", default="*.json")
    parser.add_argument("--trajectory_glob", default="*/log/trajectory.jsonl")
    parser.add_argument("--target_alignment", choices=("next", "current"), default="next")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--strict_metadata", action="store_true")
    parser.add_argument("--keep_render_overlay", action="store_true")
    args = parser.parse_args()
    if args.frame_stride <= 0:
        raise ValueError("--frame_stride must be positive")
    stats = export_dataset(args)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
