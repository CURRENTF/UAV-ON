#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split UAV-ON task metadata into one JSON shard per scene.")
    parser.add_argument(
        "--input",
        default="/root/autodl-fs/datasets/uavon_train_generation_balanced.json",
        help="Input JSON list containing UAV-ON task dictionaries.",
    )
    parser.add_argument(
        "--output_dir",
        default="/root/autodl-fs/datasets/uavon_train_generation_scene_shards",
        help="Directory where scene JSON shards and manifest.json are written.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing shard files.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = output_dir / "manifest.json"

    if not input_path.exists():
        raise FileNotFoundError(f"Input metadata does not exist: {input_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Use --overwrite.")

    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"Expected input JSON list, got {type(data).__name__}: {input_path}")

    scenes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"Expected task dict at row {index}, got {type(item).__name__}")
        scene = str(item.get("map_name") or "").strip()
        if not scene:
            raise ValueError(f"Missing map_name at row {index}")
        scenes[scene].append(item)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for scene in sorted(scenes):
        safe = _safe_name(scene)
        shard_path = output_dir / f"{safe}.json"
        shard_path.write_text(json.dumps(scenes[scene], ensure_ascii=False, indent=2), encoding="utf-8")
        manifest.append({"scene": scene, "path": str(shard_path), "num_tasks": len(scenes[scene])})

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(manifest)} scene shards to {output_dir}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
