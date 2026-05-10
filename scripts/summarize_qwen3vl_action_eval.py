#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path}:{line_no}: {exc}") from exc
    return rows


def _task_status(task_dir: Path) -> str:
    parent = task_dir.parent.name
    if parent.startswith("success_"):
        return "success"
    if parent.startswith("oracle_"):
        return "oracle"
    return "fail"


def _termination_type(rows: list[dict[str, Any]], max_actions: int) -> str:
    if not rows:
        return "missing"
    last = rows[-1]
    if bool(last.get("is_collision", False)):
        return "collision"
    if int(last.get("frame", -1)) >= max_actions or len(rows) >= max_actions + 1:
        return "step_limit"
    return "stop"


def summarize(eval_root: Path, max_actions: int) -> dict[str, Any]:
    if not eval_root.exists():
        raise FileNotFoundError(f"Evaluation root does not exist: {eval_root}")

    task_dirs = sorted(path for path in eval_root.rglob("task_*") if path.is_dir())
    if not task_dirs:
        raise ValueError(f"No task_* directories found under {eval_root}")

    task_records: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    termination_counts: Counter[str] = Counter()
    trajectory_action_counts: Counter[str] = Counter()
    final_distances: list[float] = []
    spl_terms: list[float] = []
    metric_failed = 0

    for task_dir in task_dirs:
        status = _task_status(task_dir)
        status_counts[status] += 1
        trajectory_path = task_dir / "log" / "trajectory.jsonl"
        object_path = task_dir / "object_description.json"
        if not trajectory_path.exists() or not object_path.exists():
            metric_failed += 1
            task_records.append(
                {
                    "task_dir": str(task_dir),
                    "status": status,
                    "metric_status": "metric_failed",
                    "missing_trajectory": not trajectory_path.exists(),
                    "missing_object_description": not object_path.exists(),
                }
            )
            continue

        rows = _read_jsonl(trajectory_path)
        termination = _termination_type(rows, max_actions=max_actions)
        termination_counts[termination] += 1
        for row in rows:
            action = row.get("action")
            if action is not None:
                trajectory_action_counts[str(action)] += 1

        final_distance = None
        final_move_distance = None
        if rows:
            final = rows[-1]
            final_distance = final.get("distance_to_end")
            final_move_distance = final.get("move_distance")
            if isinstance(final_distance, (int, float)):
                final_distances.append(float(final_distance))

        geodesic_distance = None
        object_desc = _read_json(object_path)
        info = object_desc.get("info", {}) if isinstance(object_desc, dict) else {}
        if isinstance(info, dict):
            geodesic_distance = info.get("geodesic_distance")

        spl = 0.0
        if (
            status == "success"
            and isinstance(final_move_distance, (int, float))
            and isinstance(geodesic_distance, (int, float))
            and final_move_distance > 0
        ):
            spl = float(geodesic_distance) / max(float(final_move_distance), float(geodesic_distance))
        spl_terms.append(spl)

        task_records.append(
            {
                "task_dir": str(task_dir),
                "status": status,
                "metric_status": "success",
                "termination": termination,
                "final_distance_to_end": final_distance,
                "final_move_distance": final_move_distance,
                "geodesic_distance": geodesic_distance,
                "num_logged_frames": len(rows),
            }
        )

    total_tasks = len(task_dirs)
    raw_path = eval_root / "qwen3vl_action_raw_outputs.jsonl"
    raw_status_counts: Counter[str] = Counter()
    raw_action_counts: Counter[str] = Counter()
    if raw_path.exists():
        for row in _read_jsonl(raw_path):
            raw_status_counts[str(row.get("status", "missing"))] += 1
            raw_action_counts[str(row.get("action", "missing"))] += 1

    summary = {
        "eval_root": str(eval_root),
        "max_actions": max_actions,
        "total_tasks": total_tasks,
        "metric_failed_tasks": metric_failed,
        "success_tasks": status_counts["success"],
        "oracle_tasks": status_counts["oracle"],
        "fail_tasks": status_counts["fail"],
        "success_rate": status_counts["success"] / total_tasks,
        "oracle_success_rate": (status_counts["success"] + status_counts["oracle"]) / total_tasks,
        "distance_to_success": mean(final_distances) if final_distances else None,
        "spl": mean(spl_terms) if spl_terms else None,
        "termination_counts": dict(sorted(termination_counts.items())),
        "trajectory_action_counts": dict(sorted(trajectory_action_counts.items())),
        "raw_output_path": str(raw_path) if raw_path.exists() else None,
        "raw_output_status_counts": dict(sorted(raw_status_counts.items())),
        "raw_output_action_counts": dict(sorted(raw_action_counts.items())),
        "tasks": task_records,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize UAV-ON Qwen3VL action evaluation outputs.")
    parser.add_argument("--eval_root", required=True, help="Directory passed as UAV_ON_EVAL_SAVE_PATH.")
    parser.add_argument("--max_actions", type=int, default=150)
    parser.add_argument("--output", default="", help="Optional JSON output path. Defaults to eval_root/metrics_summary.json.")
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    summary = summarize(eval_root=eval_root, max_actions=args.max_actions)
    output = Path(args.output) if args.output else eval_root / "metrics_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    printable = {key: value for key, value in summary.items() if key != "tasks"}
    print(json.dumps(printable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
