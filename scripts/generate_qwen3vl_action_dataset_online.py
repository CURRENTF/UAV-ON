#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from common.runtime_config import (  # noqa: E402
    UavOnRuntimeConfig,
    runtime_config_from_env,
    str_to_bool,
)


def _str2bool(value):
    return str_to_bool(value)


DEFAULT_RUNTIME_CONFIG = runtime_config_from_env()
CUSTOM_ARGV = sys.argv[:]
custom_parser = argparse.ArgumentParser(add_help=False)
custom_parser.add_argument("--output_dir", required=True)
custom_parser.add_argument("--max_samples", type=int, default=20000)
custom_parser.add_argument("--max_episodes", type=int, default=0)
custom_parser.add_argument("--flush_every", type=int, default=100)
custom_parser.add_argument("--image_quality", type=int, default=90)
custom_parser.add_argument("--rgb_only", type=_str2bool, default=DEFAULT_RUNTIME_CONFIG.rgb_only)
custom_parser.add_argument("--rgb_compress", type=_str2bool, default=DEFAULT_RUNTIME_CONFIG.rgb_compress)
custom_parser.add_argument("--jpeg_optimize", type=_str2bool, default=DEFAULT_RUNTIME_CONFIG.jpeg_optimize)
custom_parser.add_argument("--inline_vector_env", type=_str2bool, default=DEFAULT_RUNTIME_CONFIG.inline_vector_env)
custom_parser.add_argument("--image_settle_seconds", type=float, default=DEFAULT_RUNTIME_CONFIG.image_settle_seconds)
custom_parser.add_argument("--set_pose_settle_seconds", type=float, default=DEFAULT_RUNTIME_CONFIG.set_pose_settle_seconds)
custom_parser.add_argument("--overwrite", action="store_true")
custom_parser.add_argument("--status_every", type=int, default=100)
custom_args, remaining_argv = custom_parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_argv

CUSTOM_RUNTIME_CONFIG = UavOnRuntimeConfig(
    inline_vector_env=custom_args.inline_vector_env,
    vector_env_start_method=DEFAULT_RUNTIME_CONFIG.vector_env_start_method,
    kinematic_actions=DEFAULT_RUNTIME_CONFIG.kinematic_actions,
    scene_boot_seconds=DEFAULT_RUNTIME_CONFIG.scene_boot_seconds,
    action_timeout_seconds=DEFAULT_RUNTIME_CONFIG.action_timeout_seconds,
    verbose_pose=DEFAULT_RUNTIME_CONFIG.verbose_pose,
    set_pose_settle_seconds=custom_args.set_pose_settle_seconds,
    rgb_only=custom_args.rgb_only,
    rgb_compress=custom_args.rgb_compress,
    image_settle_seconds=custom_args.image_settle_seconds,
    jpeg_optimize=custom_args.jpeg_optimize,
    astar_plan_cache=DEFAULT_RUNTIME_CONFIG.astar_plan_cache,
)
custom_env = CUSTOM_RUNTIME_CONFIG.to_env()
os.environ["UAV_ON_RGB_ONLY"] = custom_env["UAV_ON_RGB_ONLY"]
os.environ["UAV_ON_RGB_COMPRESS"] = custom_env["UAV_ON_RGB_COMPRESS"]
os.environ["UAV_ON_JPEG_OPTIMIZE"] = custom_env["UAV_ON_JPEG_OPTIMIZE"]
os.environ["UAV_ON_INLINE_VECTOR_ENV"] = custom_env["UAV_ON_INLINE_VECTOR_ENV"]
os.environ["UAV_ON_IMAGE_SETTLE_SECONDS"] = custom_env["UAV_ON_IMAGE_SETTLE_SECONDS"]
os.environ["UAV_ON_SET_POSE_SETTLE_SECONDS"] = custom_env["UAV_ON_SET_POSE_SETTLE_SECONDS"]

from common.param import args  # noqa: E402
from common.uavon_action_schema import (  # noqa: E402
    QWEN3VL_ACTION_SYSTEM_PROMPT,
    UAVON_ACTION_PROMPT_VERSION,
    UAVON_VALID_ACTIONS,
    build_qwen3vl_action_user_prompt,
)
from env_uav import AirVLNENV  # noqa: E402
from model_wrapper.AStarOracle import AStarOracle  # noqa: E402


SYSTEM_PROMPT = QWEN3VL_ACTION_SYSTEM_PROMPT


def _task_instruction(task: dict[str, Any]) -> str:
    target_name = str(task.get("true_name") or task.get("object_name") or "target object").strip()
    size = str(task.get("size") or task.get("object_size") or "").strip()
    description = str(task.get("description") or "").strip()
    parts = [f"Find and stop near the target object: {target_name}."]
    if size:
        parts.append(f"Target size: {size}.")
    if description:
        parts.append(f"Target description: {description}")
    return " ".join(parts)


def _user_prompt(instruction: str) -> str:
    return build_qwen3vl_action_user_prompt(instruction)


def _decode_rgb(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        buffer = np.frombuffer(image, dtype=np.uint8)
        decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Failed to decode compressed AirSim image bytes")
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        return Image.fromarray(decoded).convert("RGB")
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[-1] == 3:
        return Image.fromarray(array.astype(np.uint8)).convert("RGB")
    raise ValueError(f"Unsupported RGB image type/shape: {type(image)!r} {getattr(array, 'shape', None)}")


def _make_fourview_grid(images: list[Any]) -> Image.Image:
    if len(images) != 4:
        raise ValueError(f"Expected four RGB images, got {len(images)}")
    pil_images = [_decode_rgb(image) for image in images]
    tile_size = min(min(image.size) for image in pil_images)
    labels = ["front", "left", "right", "down"]
    tiles = []
    for image, label in zip(pil_images, labels):
        tile = image.resize((tile_size, tile_size), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, tile_size, 30), fill=(0, 0, 0))
        draw.text((8, 8), label, fill=(255, 255, 255))
        tiles.append(tile)

    grid = Image.new("RGB", (tile_size * 2, tile_size * 2), (0, 0, 0))
    grid.paste(tiles[0], (0, 0))
    grid.paste(tiles[1], (tile_size, 0))
    grid.paste(tiles[2], (0, tile_size))
    grid.paste(tiles[3], (tile_size, tile_size))
    return grid


def _save_image(observation: dict[str, Any], output_path: Path, quality: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid = _make_fourview_grid(observation["rgb"])
    grid.save(output_path, format="JPEG", quality=quality, optimize=CUSTOM_RUNTIME_CONFIG.jpeg_optimize)


def _record_sample(
    *,
    sample_id: str,
    rel_image_path: Path,
    task: dict[str, Any],
    observation: dict[str, Any],
    action: str,
    step_size: float,
    plan_summary: str,
) -> dict[str, Any]:
    instruction = _task_instruction(task)
    prompt = _user_prompt(instruction)
    return {
        "id": sample_id,
        "image": rel_image_path.as_posix(),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rel_image_path.as_posix()},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": action},
        ],
        "instruction": instruction,
        "action": action,
        "step_size": float(step_size),
        "map_name": task.get("map_name", ""),
        "episode_id": str(task.get("episode_id", task.get("task_id", ""))),
        "frame_index": int(observation.get("step", -1)),
        "distance_to_target": observation.get("distance_to_target", observation.get("distance_to_end")),
        "plan_summary": plan_summary,
    }


def _write_stats(stats_path: Path, stats: dict[str, Any]) -> None:
    tmp_path = stats_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(stats_path)


def _add_timing(stats: dict[str, Any], name: str, elapsed: float) -> None:
    timings = stats.setdefault("timings", {})
    item = timings.setdefault(name, {"seconds": 0.0, "count": 0})
    item["seconds"] = round(float(item.get("seconds", 0.0)) + float(elapsed), 6)
    item["count"] = int(item.get("count", 0)) + 1


def _new_stats(output_dir: Path, jsonl_path: Path) -> dict[str, Any]:
    return {
        "output_dir": str(output_dir),
        "jsonl_path": str(jsonl_path),
        "dataset_path": args.dataset_path,
        "max_samples": custom_args.max_samples,
        "max_episodes": custom_args.max_episodes,
        "batch_size": args.batchSize,
        "max_actions": args.maxActions,
        "simulator_tool_port": args.simulator_tool_port,
        "gpu_id": args.gpu_id,
        "performance": {
            "rgb_only": CUSTOM_RUNTIME_CONFIG.rgb_only,
            "rgb_compress": CUSTOM_RUNTIME_CONFIG.rgb_compress,
            "jpeg_optimize": CUSTOM_RUNTIME_CONFIG.jpeg_optimize,
            "inline_vector_env": CUSTOM_RUNTIME_CONFIG.inline_vector_env,
            "image_settle_seconds": CUSTOM_RUNTIME_CONFIG.image_settle_seconds,
            "set_pose_settle_seconds": CUSTOM_RUNTIME_CONFIG.set_pose_settle_seconds,
        },
        "runtime_config": CUSTOM_RUNTIME_CONFIG.to_dict(),
        "astar": {
            "voxel_resolution": args.astar_voxel_resolution,
            "voxel_margin_xy": args.astar_voxel_margin_xy,
            "voxel_margin_z": args.astar_voxel_margin_z,
            "min_extent_xy": args.astar_min_extent_xy,
            "min_extent_z": args.astar_min_extent_z,
            "max_voxels": args.astar_max_voxels,
            "target_search_radius": args.astar_target_search_radius,
            "max_goal_candidates": args.astar_max_goal_candidates,
            "max_move_voxels": args.astar_max_move_voxels,
        },
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": None,
        "elapsed_seconds": None,
        "num_rows": 0,
        "num_batches": 0,
        "num_episodes_started": 0,
        "num_episodes_with_samples": 0,
        "num_plan_failed": 0,
        "num_collision_or_env_done": 0,
        "actions": {},
        "scenes": {},
        "failures": [],
        "timings": {},
        "command_argv": CUSTOM_ARGV,
        "action_prompt_version": UAVON_ACTION_PROMPT_VERSION,
    }


def _rebuild_progress_from_jsonl(jsonl_path: Path) -> tuple[int, Counter[str], dict[str, Counter[str]], set[str], int]:
    action_counts: Counter[str] = Counter()
    scene_counts: dict[str, Counter[str]] = defaultdict(Counter)
    episodes_with_samples: set[str] = set()
    bad_lines = 0
    num_rows = 0

    if not jsonl_path.exists():
        return num_rows, action_counts, scene_counts, episodes_with_samples, bad_lines

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue

            num_rows += 1
            action = data.get("action")
            if action:
                action_counts[str(action)] += 1
            scene = str(data.get("map_name") or "unknown")
            episode_id = str(data.get("episode_id", data.get("task_id", "")))
            episodes_with_samples.add(f"{scene}/{episode_id}")
            scene_counts[scene]["rows"] += 1
            if action:
                scene_counts[scene][f"action_{action}"] += 1

    return num_rows, action_counts, scene_counts, episodes_with_samples, bad_lines


def main() -> None:
    if args.is_fixed:
        raise RuntimeError("A* data generation requires --is_fixed false.")
    if custom_args.max_samples <= 0:
        raise ValueError("--max_samples must be positive")

    output_dir = Path(custom_args.output_dir).resolve()
    jsonl_path = output_dir / "train.jsonl"
    image_root = output_dir / "images"
    stats_path = output_dir / "generation_stats.json"
    action_counts: Counter[str] = Counter()
    scene_counts: dict[str, Counter[str]] = defaultdict(Counter)
    episodes_with_samples: set[str] = set()
    if stats_path.exists() and not custom_args.overwrite:
        stats = json.loads(stats_path.read_text())
        (
            rebuilt_rows,
            action_counts,
            scene_counts,
            episodes_with_samples,
            bad_lines,
        ) = _rebuild_progress_from_jsonl(jsonl_path)
        if bad_lines:
            raise RuntimeError(f"Cannot resume with {bad_lines} invalid JSONL lines in {jsonl_path}")
        if rebuilt_rows != stats.get("num_rows"):
            print(
                f"[generate] correcting stale stats rows {stats.get('num_rows')} -> {rebuilt_rows} "
                f"from {jsonl_path}",
                flush=True,
            )
        stats["num_rows"] = rebuilt_rows
        stats["num_episodes_with_samples"] = len(episodes_with_samples)
        stats["actions"] = dict(sorted(action_counts.items()))
        stats["scenes"] = {key: dict(value) for key, value in sorted(scene_counts.items())}
        stats["resume_batch_size"] = args.batchSize
        stats["resume_max_actions"] = args.maxActions
        stats["resume_command_argv"] = CUSTOM_ARGV
        _write_stats(stats_path, stats)
        print(f"Resuming from {stats['num_rows']} rows, {len(episodes_with_samples)} episodes already done")
    else:
        if output_dir.exists() and custom_args.overwrite:
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stats = _new_stats(output_dir, jsonl_path)
    
    image_root.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    stage_start = time.time()
    env = AirVLNENV(batch_size=args.batchSize, dataset_path=args.dataset_path, save_path=args.eval_save_path)
    _add_timing(stats, "init_env", time.time() - stage_start)
    stage_start = time.time()
    oracle = AStarOracle(batch_size=args.batchSize, args=args)
    _add_timing(stats, "init_oracle", time.time() - stage_start)

    try:
        with jsonl_path.open("a" if not custom_args.overwrite else "w", encoding="utf-8") as handle:
            while stats["num_rows"] < custom_args.max_samples:
                if custom_args.max_episodes and stats["num_episodes_started"] >= custom_args.max_episodes:
                    break
                stage_start = time.time()
                env_batch = env.next_minibatch(skip_scenes=[])
                _add_timing(stats, "next_minibatch", time.time() - stage_start)
                if env_batch is None:
                    break
                
                env_batch = [
                    task
                    for task in env_batch
                    if (
                        f"{str(task.get('map_name', 'unknown'))}/"
                        f"{str(task.get('episode_id', task.get('task_id', '')))}"
                    )
                    not in episodes_with_samples
                ]
                if not env_batch:
                    continue

                stats["num_batches"] += 1
                stats["num_episodes_started"] += len(env_batch)
                stage_start = time.time()
                outputs = env.reset()
                _add_timing(stats, "env_reset", time.time() - stage_start)
                observations, env_dones, collisions, _oracle_success = [list(x) for x in zip(*outputs)]
                stage_start = time.time()
                oracle.prepare_batch(env=env, batch=env_batch)
                _add_timing(stats, "oracle_prepare_batch", time.time() - stage_start)
                stats["astar_cache"] = {
                    "hits": int(getattr(oracle, "cache_hits", 0)),
                    "misses": int(getattr(oracle, "cache_misses", 0)),
                    "enabled": runtime_config_from_env().astar_plan_cache,
                }
                plan_summaries = list(oracle.plan_summaries)
                active = []
                for batch_index, summary in enumerate(plan_summaries):
                    failed = "failed" in summary.lower()
                    active.append(not failed)
                    if failed:
                        stats["num_plan_failed"] += 1
                        stats["failures"].append(
                            {
                                "episode_id": str(env_batch[batch_index].get("episode_id", "")),
                                "map_name": env_batch[batch_index].get("map_name", ""),
                                "summary": summary,
                            }
                        )

                for _step in range(args.maxActions):
                    if stats["num_rows"] >= custom_args.max_samples or not any(active):
                        break
                    inputs = list(range(len(env_batch)))
                    actions, step_sizes, dones = oracle.run(inputs, fixed=False)

                    for batch_index, task in enumerate(env_batch):
                        if stats["num_rows"] >= custom_args.max_samples:
                            break
                        if not active[batch_index]:
                            continue
                        action = actions[batch_index]
                        if action not in UAVON_VALID_ACTIONS:
                            stats["failures"].append(
                                {
                                    "episode_id": str(task.get("episode_id", "")),
                                    "map_name": task.get("map_name", ""),
                                    "summary": f"invalid action: {action}",
                                }
                            )
                            active[batch_index] = False
                            continue

                        scene = str(task.get("map_name", "unknown"))
                        episode_id = str(task.get("episode_id", task.get("task_id", batch_index)))
                        sample_id = f"{scene}_{episode_id}_{int(observations[batch_index][-1].get('step', _step)):06d}_{stats['num_rows']:08d}"
                        rel_image_path = Path("images") / scene / episode_id / f"{sample_id}.jpg"
                        stage_start = time.time()
                        _save_image(observations[batch_index][-1], output_dir / rel_image_path, custom_args.image_quality)
                        _add_timing(stats, "save_image", time.time() - stage_start)
                        record = _record_sample(
                            sample_id=sample_id,
                            rel_image_path=rel_image_path,
                            task=task,
                            observation=observations[batch_index][-1],
                            action=action,
                            step_size=step_sizes[batch_index],
                            plan_summary=plan_summaries[batch_index],
                        )
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

                        stats["num_rows"] += 1
                        action_counts[action] += 1
                        scene_counts[scene]["rows"] += 1
                        scene_counts[scene][f"action_{action}"] += 1
                        episodes_with_samples.add(f"{scene}/{episode_id}")

                        if stats["num_rows"] % custom_args.flush_every == 0:
                            handle.flush()
                            os.fsync(handle.fileno())
                            stats["actions"] = dict(sorted(action_counts.items()))
                            stats["scenes"] = {key: dict(value) for key, value in sorted(scene_counts.items())}
                            stats["num_episodes_with_samples"] = len(episodes_with_samples)
                            stats["elapsed_seconds"] = round(time.time() - start_time, 3)
                            _write_stats(stats_path, stats)
                        if stats["num_rows"] % custom_args.status_every == 0:
                            elapsed = max(time.time() - start_time, 1e-6)
                            rate = stats["num_rows"] / elapsed
                            print(
                                f"[generate] rows={stats['num_rows']} episodes={len(episodes_with_samples)} "
                                f"rate={rate:.3f}/s actions={dict(action_counts)}",
                                flush=True,
                            )

                        if dones[batch_index]:
                            active[batch_index] = False

                    if stats["num_rows"] >= custom_args.max_samples or not any(active):
                        break

                    stage_start = time.time()
                    env.makeActions(actions, step_sizes, is_fixed=False)
                    _add_timing(stats, "env_make_actions", time.time() - stage_start)
                    stage_start = time.time()
                    outputs = env.get_obs()
                    _add_timing(stats, "env_get_obs", time.time() - stage_start)
                    observations, env_dones, collisions, _oracle_success = [list(x) for x in zip(*outputs)]
                    for batch_index, (env_done, collision) in enumerate(zip(env_dones, collisions)):
                        if env_done or collision:
                            if active[batch_index]:
                                stats["num_collision_or_env_done"] += 1
                            active[batch_index] = False

            handle.flush()
            os.fsync(handle.fileno())
    finally:
        try:
            if hasattr(env, "simulator_tool"):
                env.simulator_tool.closeScenes()
        except Exception as exc:
            print(f"[generate] closeScenes failed: {exc}", flush=True)
        try:
            env.delete_VectorEnvUtil()
        except Exception:
            pass

    stats["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    stats["action_prompt_version"] = UAVON_ACTION_PROMPT_VERSION
    stats["elapsed_seconds"] = round(time.time() - start_time, 3)
    stats["num_episodes_with_samples"] = len(episodes_with_samples)
    stats["actions"] = dict(sorted(action_counts.items()))
    stats["scenes"] = {key: dict(value) for key, value in sorted(scene_counts.items())}
    _write_stats(stats_path, stats)
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
