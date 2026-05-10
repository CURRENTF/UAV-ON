# UAV-ON A* Action Data Generation

Observed on 2026-05-09 in AutoDL under `/root/autodl-tmp/UAV-ON`.

The absolute paths in this document are machine-local paths for the current
AutoDL instance. They document where data is stored on this server only; do not
treat them as portable repo defaults.

## Purpose

Generate offline Qwen3-VL action-supervision samples from downloaded UAV-ON train scenes and trainset metadata:

```
current four-view observation + task instruction -> A* action label
```

The current production run targets 2,000 samples per train scene, for at least 20,000 samples total.

## Paths

- Repo: `/root/autodl-tmp/UAV-ON`
- Train environments: `/root/autodl-fs/datasets/UAV-ON-train-unpacked/TRAIN_ENVS`
- Scene shard manifest: `/root/autodl-fs/datasets/uavon_train_generation_scene_shards/manifest.json`
- Output shards: `/root/autodl-fs/datasets/uavon_qwen3vl_action_train_astar_scene_shards`
- Main generation log: `/root/autodl-fs/logs/uavon_qwen3vl_generate_20k_resume.log`
- Detached wrapper log: `/root/autodl-fs/logs/uavon_qwen3vl_generate_detached_wrapper.log`
- Detached server log: `/root/autodl-fs/logs/uavon_train_server_detached.log`

## Local AutoDL Dataset Inventory

Observed under `/root/autodl-fs/datasets` on this AutoDL machine:

- Official UAV-ON metadata repo: `/root/autodl-fs/datasets/UAV-ON-dataset`
- Official train metadata JSONs: `/root/autodl-fs/datasets/UAV-ON-dataset/UAV-ON-data/trainset`
- Official validation metadata JSONs: `/root/autodl-fs/datasets/UAV-ON-dataset/UAV-ON-data/valset`
- Downloaded train environment zips: `/root/autodl-fs/datasets/UAV-ON-envs-train`
- Downloaded test environment zips: `/root/autodl-fs/datasets/UAV-ON-envs-test`
- Unpacked train environments used by `scripts/start_server_train.sh`: `/root/autodl-fs/datasets/UAV-ON-train-unpacked/TRAIN_ENVS`
- Partially unpacked test environments currently present: `/root/autodl-fs/datasets/UAV-ON-test-unpacked`
- Balanced train-generation metadata: `/root/autodl-fs/datasets/uavon_train_generation_balanced.json`
- Per-scene train-generation shards: `/root/autodl-fs/datasets/uavon_train_generation_scene_shards`
- Generated Qwen3-VL A* action shards: `/root/autodl-fs/datasets/uavon_qwen3vl_action_train_astar_scene_shards`

Approximate sizes observed on this machine:

- `/root/autodl-fs/datasets/UAV-ON-dataset`: 11 MB
- `/root/autodl-fs/datasets/UAV-ON-envs-train`: 25 GB
- `/root/autodl-fs/datasets/UAV-ON-envs-test`: 42 GB
- `/root/autodl-fs/datasets/UAV-ON-train-unpacked`: 27 GB
- `/root/autodl-fs/datasets/UAV-ON-test-unpacked`: 5.4 GB
- `/root/autodl-fs/datasets/uavon_qwen3vl_action_train_astar_scene_shards`: 2.2 GB

The unpacked train environments currently include:

- `BrushifyUrban`
- `CabinLake`
- `CityPark`
- `DownTown`
- `Neighborhood`
- `Slum`
- `UrbanJapan`
- `Venice`
- `WesternTown`
- `WinterTown`

### PCD/YAML Resources

The trajectory code in `../temp/traj_gen` expects additional local resources
such as `configs/{env}.yaml` and `scene_data/pcd_map/{env}.ply`. Those resources
were not found in the UAV-ON repo or under the UAV-ON dataset directories on
this AutoDL machine as of 2026-05-09.

The only `.ply` files found under `/root/autodl-fs/datasets` were in
`/root/autodl-fs/datasets/scene_datasets/mp3d`, which is unrelated to UAV-ON.
So the currently available UAV-ON dataset resources are enough for the existing
online AirSim voxel A* pipeline, but not enough by themselves to run the
`../temp` point-cloud/YAML A* pipeline without generating or obtaining those
extra PCD/YAML assets.

Each scene output contains:

- `train.jsonl`
- `images/`
- `generation_stats.json`
- `run_logs/`

## Start

Start the AirSim train server on GPU 0:

```bash
cd /root/autodl-tmp/UAV-ON
setsid bash -lc 'cd /root/autodl-tmp/UAV-ON && UAV_ON_DATA_ROOT=/root/autodl-fs/datasets/UAV-ON-train-unpacked UAV_ON_SERVER_GPUS=0 UAV_ON_SIM_PORT=30000 bash scripts/start_server_train.sh' \
  >> /root/autodl-fs/logs/uavon_train_server_detached.log 2>&1 < /dev/null &
```

Start or resume all train scene shards:

```bash
cd /root/autodl-tmp/UAV-ON
setsid bash scripts/tmp/resume_generate_20k.sh \
  >> /root/autodl-fs/logs/uavon_qwen3vl_generate_detached_wrapper.log 2>&1 < /dev/null &
```

The active production settings in `scripts/tmp/resume_generate_20k.sh` are:

- `UAV_ON_BATCH_SIZE=1`
- `UAV_ON_VECTOR_ENV_START_METHOD=fork`
- `UAV_ON_SCENE_BOOT_SECONDS=60`
- `UAV_ON_SIM_PORT=30000`
- `UAV_ON_CUDA_VISIBLE_DEVICES=0`
- `UAV_ON_GPU_ID=0`

## Verify

Check generated rows and process state:

```bash
cd /root/autodl-tmp/UAV-ON
find /root/autodl-fs/datasets/uavon_qwen3vl_action_train_astar_scene_shards -mindepth 1 -maxdepth 1 -type d -print | sort | while read -r d; do
  rows=0
  imgs=0
  [[ -f "$d/train.jsonl" ]] && rows=$(wc -l < "$d/train.jsonl")
  [[ -d "$d/images" ]] && imgs=$(find "$d/images" -type f -name '*.jpg' | wc -l)
  statrows=NA
  [[ -f "$d/generation_stats.json" ]] && statrows=$(/root/miniconda3/envs/uavon/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("num_rows"))' "$d/generation_stats.json")
  printf '%s rows=%s images=%s stat_rows=%s\n' "$(basename "$d")" "$rows" "$imgs" "$statrows"
done

pgrep -af 'resume_generate_20k|generate_qwen3vl_action_dataset_online|AirVLNSimulatorServerTool|Linux-Shipping'
tail -f /root/autodl-fs/logs/uavon_qwen3vl_generate_20k_resume.log
```

`train.jsonl` can be ahead of `generation_stats.json` between flushes. The generator flushes stats every 100 rows and rebuilds stale stats from JSONL on resume.

## Stop

Stop the detached generator first, then the server:

```bash
pkill -TERM -f 'scripts/tmp/resume_generate_20k.sh|generate_qwen3vl_action_dataset_online.py'
pkill -TERM -f 'scripts/start_server_train.sh|AirVLNSimulatorServerTool.py'
pkill -TERM -f 'Linux-Shipping'
```

If a packaged Unreal process ignores `TERM`, use `kill -KILL <pid>` for that Unreal PID only.

## Observed Status

At 2026-05-09 20:22 CST, generated JSONL rows were:

- `BrushifyUrban_TrainSets`: 2,000
- `CabinLake_TrainSets`: 2,000
- `CityPark_TrainSets`: 2,000
- `DownTown_train`: 542 and still running

The generator had successfully resumed after interruption and automatically advanced from `CityPark_TrainSets` to `DownTown_train`.

## Batch Size Notes

Observed single-GPU tests:

- `batchSize=1` is the current stable production setting.
- `batchSize=2` can launch after per-port Unreal runtime isolation, but throughput was worse in CityPark and CPU/RPC contention increased.
- The GPU was not saturated, but increasing batch size caused Unreal/AirSim contention instead of improving throughput.

Additional speed benchmarks on 2026-05-10 on a separate AutoDL RTX 4080 SUPER
instance showed that the best tested setting there was `batchSize=4` with
RGB-only observation capture and in-process observation formatting:

| Run | Rows | Total seconds | Total rows/s | Non-reset rows/s | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `baseline_bs1_20_stock` | 20 | 136.2 | 0.147 | 0.470 | stock RGB+depth, JPEG optimize, vector worker |
| `opt_bs1_20_inline_rgbfast` | 20 | 121.5 | 0.165 | 0.702 | RGB-only, no JPEG optimize, inline vector, no pose/image settle |
| `opt_bs2_40_inline_rgbfast` | 40 | 136.4 | 0.293 | 1.007 | two CityPark instances on one GPU |
| `opt_bs4_80_inline_rgbfast` | 80 | 169.9 | 0.471 | 1.216 | four CityPark instances on one GPU |
| `opt_bs4_80_parallel_rgbfast` | 80 | 195.0 | 0.410 | 0.882 | experimental parallel A* was slower |

In these runs, scene startup dominated small benchmarks (`env_reset` was about
90-102 seconds). The steady-state improvement comes mostly from:

- `UAV_ON_RGB_ONLY=1`, because Qwen3-VL action data only saves RGB four-view images.
- `UAV_ON_INLINE_VECTOR_ENV=1`, because the generation path only needs to format `SimState` and does not need separate vector worker processes.
- `UAV_ON_JPEG_OPTIMIZE=0`, which reduces CPU time spent saving JPEGs.

The `image_settle_seconds=0` and `set_pose_settle_seconds=0` benchmark setting
is the fastest tested mode, but it is more aggressive. For production data,
keep the default 0.2-second settle values unless a visual sanity check confirms
that the rendered frame has updated correctly after each kinematic pose change.

Recommended conservative 4080 SUPER generation overrides:

```bash
export UAV_ON_BATCH_SIZE=4
export UAV_ON_KINEMATIC_ACTIONS=1
export UAV_ON_SCENE_BOOT_SECONDS=75

bash scripts/generate_qwen3vl_action_dataset.sh \
  --rgb_only true \
  --jpeg_optimize false \
  --inline_vector_env true
```

For a faster benchmark run after visual validation, add:

```bash
  --image_settle_seconds 0 \
  --set_pose_settle_seconds 0
```

The shell wrapper also accepts the older `UAV_ON_RGB_ONLY`,
`UAV_ON_JPEG_OPTIMIZE`, `UAV_ON_INLINE_VECTOR_ENV`,
`UAV_ON_IMAGE_SETTLE_SECONDS`, and `UAV_ON_SET_POSE_SETTLE_SECONDS`
environment variables and forwards them to the Python argparse interface.

Do not parallelize A* voxel planning across the batch on this machine. Testing
four concurrent `simCreateVoxelGrid` calls increased `oracle_prepare_batch`
from 32.7 seconds to 56.6 seconds for 4 episodes.

## Troubleshooting

Symptom: generator is alive but `train.jsonl` and images stop advancing.

- Check the latest image mtime and process CPU.
- The production code sets AirSim move/rotate `timeout_sec` through `UAV_ON_ACTION_TIMEOUT_SECONDS` with a default of 12 seconds.
- `src/env_uav.py` treats failed `move_to_next_pose` as collision so the current episode can be skipped instead of hanging the full run.

Symptom: closing Codex stops generation.

- The generator must be launched with `setsid` as shown above.
- Verify parent PID is `1` in `ps -eo pid,ppid,pgid,sid,tty,stat,etime,cmd`.

Symptom: server restart says address already in use.

- An old `AirVLNSimulatorServerTool.py` is still listening on `30000`; stop it before starting a new detached server.
