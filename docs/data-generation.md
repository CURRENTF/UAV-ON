# UAV-ON A* Action Data Generation

Observed on 2026-05-09 in AutoDL under `/root/autodl-tmp/UAV-ON`.

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

