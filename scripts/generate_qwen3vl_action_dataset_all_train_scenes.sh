#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

manifest="${UAV_ON_SCENE_SHARD_MANIFEST:-/root/autodl-fs/datasets/uavon_train_generation_scene_shards/manifest.json}"
base_output="${UAV_ON_GENERATE_BASE_OUTPUT:-/root/autodl-fs/datasets/uavon_qwen3vl_action_train_astar_scene_shards}"
target_per_scene="${UAV_ON_GENERATE_SAMPLES_PER_SCENE:-2000}"
batch_size="${UAV_ON_BATCH_SIZE:-1}"

mkdir -p "$base_output"

/root/miniconda3/envs/uavon/bin/python - <<PY | while IFS=$'\t' read -r scene dataset_path safe_name; do
import json, re
manifest = json.load(open("$manifest"))
for item in manifest:
    scene = item["scene"]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", scene)
    print(f"{scene}\t{item['path']}\t{safe}")
PY
  output_dir="$base_output/$safe_name"
  stats_path="$output_dir/generation_stats.json"
  if [[ -f "$stats_path" ]]; then
    rows="$(/root/miniconda3/envs/uavon/bin/python - <<PY
import json
print(json.load(open("$stats_path")).get("num_rows", 0))
PY
)"
    if [[ "$rows" -ge "$target_per_scene" ]]; then
      echo "[all-scenes] skip $scene rows=$rows"
      continue
    fi
  fi

  echo "[all-scenes] generating scene=$scene dataset=$dataset_path output=$output_dir target=$target_per_scene batchSize=$batch_size"
  UAV_ON_GENERATE_DATASET="$dataset_path" \
  UAV_ON_GENERATE_OUTPUT="$output_dir" \
  UAV_ON_GENERATE_MAX_SAMPLES="$target_per_scene" \
  UAV_ON_GENERATE_OVERWRITE= \
  UAV_ON_BATCH_SIZE="$batch_size" \
  bash scripts/generate_qwen3vl_action_dataset.sh
done
