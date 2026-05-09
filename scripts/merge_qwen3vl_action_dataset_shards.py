#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Qwen3-VL UAV-ON action dataset shards.")
    parser.add_argument("--shards_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    shards_dir = Path(args.shards_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_root = output_dir / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    rows = 0
    actions = Counter()
    scenes = Counter()
    shards = []
    with (output_dir / "train.jsonl").open("w", encoding="utf-8") as out_handle:
        for shard_jsonl in sorted(shards_dir.glob("*/train.jsonl")):
            shard_dir = shard_jsonl.parent
            shard_rows = 0
            with shard_jsonl.open("r", encoding="utf-8") as in_handle:
                for line in in_handle:
                    if args.max_samples and rows >= args.max_samples:
                        break
                    item = json.loads(line)
                    src_image = shard_dir / item["image"]
                    scene = str(item.get("map_name", shard_dir.name))
                    rel_image = Path("images") / shard_dir.name / Path(item["image"]).name
                    dst_image = output_dir / rel_image
                    dst_image.parent.mkdir(parents=True, exist_ok=True)
                    if not dst_image.exists():
                        shutil.copy2(src_image, dst_image)
                    item["image"] = rel_image.as_posix()
                    item["id"] = f"{shard_dir.name}_{rows:08d}"
                    item["messages"][1]["content"][0]["image"] = item["image"]
                    out_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                    rows += 1
                    shard_rows += 1
                    actions[item.get("action", "")] += 1
                    scenes[scene] += 1
            shards.append({"shard": str(shard_dir), "rows": shard_rows})
            if args.max_samples and rows >= args.max_samples:
                break

    stats = {
        "output_dir": str(output_dir),
        "jsonl_path": str(output_dir / "train.jsonl"),
        "num_rows": rows,
        "actions": dict(sorted(actions.items())),
        "scenes": dict(sorted(scenes.items())),
        "shards": shards,
    }
    (output_dir / "merge_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
