#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
root_dir=.
data_root="${UAV_ON_DATA_ROOT:-/data2/haojitai/datasets/uav-on}"
dataset_path="${UAV_ON_ASTAR_DATASET:-$data_root/DATASET/merged_trainset.json}"
save_path="${UAV_ON_ASTAR_SAVE_PATH:-$root_dir/astar_logs/scene}"
cuda_visible_devices="${UAV_ON_CUDA_VISIBLE_DEVICES:-0}"
gpu_id="${UAV_ON_GPU_ID:-0}"
sim_port="${UAV_ON_SIM_PORT:-30000}"
echo "$PWD"

CUDA_VISIBLE_DEVICES="$cuda_visible_devices" python -u $root_dir/src/eval_astar.py \
    --name AStarOracle \
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
    --astar_keep_voxels "${UAV_ON_ASTAR_KEEP_VOXELS:-false}"
