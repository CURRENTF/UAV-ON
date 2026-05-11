#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

root_dir=.
python_bin="${PYTHON_BIN:-/root/miniconda3/envs/uavon/bin/python}"
dataset_path="${UAV_ON_GENERATE_DATASET:-/root/autodl-fs/datasets/uavon_train_generation_balanced.json}"
output_dir="${UAV_ON_GENERATE_OUTPUT:-/root/autodl-fs/datasets/uavon_qwen3vl_action_train_astar_20k}"
save_path="${UAV_ON_GENERATE_SAVE_PATH:-$output_dir/run_logs}"
cuda_visible_devices="${UAV_ON_CUDA_VISIBLE_DEVICES:-0}"
gpu_id="${UAV_ON_GPU_ID:-0}"
sim_port="${UAV_ON_SIM_PORT:-30000}"
extra_args=()
[[ -n "${UAV_ON_RGB_ONLY+x}" ]] && extra_args+=(--rgb_only "$UAV_ON_RGB_ONLY")
[[ -n "${UAV_ON_RGB_COMPRESS+x}" ]] && extra_args+=(--rgb_compress "$UAV_ON_RGB_COMPRESS")
[[ -n "${UAV_ON_JPEG_OPTIMIZE+x}" ]] && extra_args+=(--jpeg_optimize "$UAV_ON_JPEG_OPTIMIZE")
[[ -n "${UAV_ON_INLINE_VECTOR_ENV+x}" ]] && extra_args+=(--inline_vector_env "$UAV_ON_INLINE_VECTOR_ENV")
[[ -n "${UAV_ON_IMAGE_SETTLE_SECONDS+x}" ]] && extra_args+=(--image_settle_seconds "$UAV_ON_IMAGE_SETTLE_SECONDS")
[[ -n "${UAV_ON_SET_POSE_SETTLE_SECONDS+x}" ]] && extra_args+=(--set_pose_settle_seconds "$UAV_ON_SET_POSE_SETTLE_SECONDS")

CUDA_VISIBLE_DEVICES="$cuda_visible_devices" "$python_bin" -u "$root_dir/scripts/generate_qwen3vl_action_dataset_online.py" \
    --output_dir "$output_dir" \
    --max_samples "${UAV_ON_GENERATE_MAX_SAMPLES:-20000}" \
    --max_episodes "${UAV_ON_GENERATE_MAX_EPISODES:-0}" \
    --flush_every "${UAV_ON_GENERATE_FLUSH_EVERY:-100}" \
    --status_every "${UAV_ON_GENERATE_STATUS_EVERY:-100}" \
    "${extra_args[@]}" \
    ${UAV_ON_GENERATE_OVERWRITE:+--overwrite} \
    --name AStarQwen3VLData \
    --maxActions "${UAV_ON_ASTAR_MAX_ACTIONS:-150}" \
    --eval_save_path "$save_path" \
    --dataset_path "$dataset_path" \
    --is_fixed false \
    --gpu_id "$gpu_id" \
    --batchSize "${UAV_ON_BATCH_SIZE:-1}" \
    --simulator_tool_port "$sim_port" \
    --astar_voxel_resolution "${UAV_ON_ASTAR_VOXEL_RESOLUTION:-1.0}" \
    --astar_voxel_margin_xy "${UAV_ON_ASTAR_VOXEL_MARGIN_XY:-25.0}" \
    --astar_voxel_margin_z "${UAV_ON_ASTAR_VOXEL_MARGIN_Z:-20.0}" \
    --astar_min_extent_xy "${UAV_ON_ASTAR_MIN_EXTENT_XY:-100.0}" \
    --astar_min_extent_z "${UAV_ON_ASTAR_MIN_EXTENT_Z:-60.0}" \
    --astar_max_voxels "${UAV_ON_ASTAR_MAX_VOXELS:-4000000}" \
    --astar_target_search_radius "${UAV_ON_ASTAR_TARGET_SEARCH_RADIUS:-20.0}" \
    --astar_max_goal_candidates "${UAV_ON_ASTAR_MAX_GOAL_CANDIDATES:-128}" \
    --astar_max_move_voxels "${UAV_ON_ASTAR_MAX_MOVE_VOXELS:-1}" \
    --astar_keep_voxels "${UAV_ON_ASTAR_KEEP_VOXELS:-false}" \
    "$@"
