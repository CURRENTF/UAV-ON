#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_FIELDS = (
    "action",
    "step_size",
    "frame_index",
    "map_name",
    "episode_id",
    "distance_to_target",
    "plan_summary",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_diff(left: Path, right: Path) -> dict[str, float | bool | str]:
    left_hash = _sha256(left)
    right_hash = _sha256(right)
    if left_hash == right_hash:
        return {
            "identical": True,
            "left_sha256": left_hash,
            "right_sha256": right_hash,
            "mean_abs": 0.0,
            "max_abs": 0.0,
            "psnr": float("inf"),
        }

    left_array = np.asarray(Image.open(left).convert("RGB"), dtype=np.float32)
    right_array = np.asarray(Image.open(right).convert("RGB"), dtype=np.float32)
    if left_array.shape != right_array.shape:
        return {
            "identical": False,
            "left_sha256": left_hash,
            "right_sha256": right_hash,
            "shape_mismatch": f"{left_array.shape} != {right_array.shape}",
            "mean_abs": float("inf"),
            "max_abs": float("inf"),
            "psnr": 0.0,
        }

    diff = np.abs(left_array - right_array)
    mse = float(np.mean((left_array - right_array) ** 2))
    psnr = float("inf") if mse == 0.0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
    return {
        "identical": False,
        "left_sha256": left_hash,
        "right_sha256": right_hash,
        "mean_abs": float(np.mean(diff)),
        "max_abs": float(np.max(diff)),
        "psnr": psnr,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--fields", nargs="*", default=list(DEFAULT_FIELDS))
    parser.add_argument("--check-images", action="store_true")
    parser.add_argument("--max-image-checks", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline_dir = args.baseline.resolve()
    candidate_dir = args.candidate.resolve()
    baseline_rows = _load_jsonl(baseline_dir / "train.jsonl")
    candidate_rows = _load_jsonl(candidate_dir / "train.jsonl")
    if args.limit:
        baseline_rows = baseline_rows[: args.limit]
        candidate_rows = candidate_rows[: args.limit]
    result: dict[str, Any] = {
        "baseline": str(baseline_dir),
        "candidate": str(candidate_dir),
        "fields": args.fields,
        "baseline_rows": len(baseline_rows),
        "candidate_rows": len(candidate_rows),
        "field_mismatches": [],
        "image_checks": [],
    }

    if len(baseline_rows) != len(candidate_rows):
        result["row_count_match"] = False
    else:
        result["row_count_match"] = True

    for index, (baseline_row, candidate_row) in enumerate(zip(baseline_rows, candidate_rows)):
        for field in args.fields:
            if baseline_row.get(field) != candidate_row.get(field):
                result["field_mismatches"].append(
                    {
                        "row": index,
                        "field": field,
                        "baseline": baseline_row.get(field),
                        "candidate": candidate_row.get(field),
                    }
                )

    if args.check_images:
        for index, (baseline_row, candidate_row) in enumerate(zip(baseline_rows, candidate_rows)):
            if index >= args.max_image_checks:
                break
            baseline_image = baseline_dir / str(baseline_row["image"])
            candidate_image = candidate_dir / str(candidate_row["image"])
            image_result = _image_diff(baseline_image, candidate_image)
            image_result["row"] = index
            image_result["baseline_image"] = str(baseline_image)
            image_result["candidate_image"] = str(candidate_image)
            result["image_checks"].append(image_result)

    result["fields_match"] = not result["field_mismatches"] and result["row_count_match"]
    result["images_identical"] = (
        all(item.get("identical") for item in result["image_checks"])
        if args.check_images
        else None
    )
    result["ok"] = bool(result["fields_match"])

    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
