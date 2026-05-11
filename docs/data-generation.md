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

A follow-up CityPark alignment test compared RGB images captured after 0.02
seconds against the same poses after 0.2 seconds for 20 A* steps and four
cameras per step. The 0.02-second images did not align with the 0.2-second
reference (`mean_abs_avg=64.1`, `mean_abs_p95=112.4`, `psnr_avg=10.37dB`), so
0.02 seconds is not recommended for production data on this machine.

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

For a faster benchmark run after visual validation, add explicit settle
overrides. Do not use the values below for production without re-running the
alignment check on the target machine:

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

## Local 4080 Simulator-Side Performance

Observed on 2026-05-11 in the local worktree
`/root/autodl-tmp/UAV-ON-sim-perf` on an RTX 4080.

For data-collection iteration, the useful metric is steady-state collection
throughput, excluding the first UE scene launch/reset:

```
steady_rows_per_second = num_rows / (elapsed_seconds - timings.env_reset.seconds)
```

The current best conservative local setting keeps the existing 0.2 second pose
and image settle delays, uses kinematic A* execution, and changes only the
collection-side transport/planning overhead:

```bash
cd /root/autodl-tmp/UAV-ON-sim-perf
export UAV_ON_BATCH_SIZE=4
export UAV_ON_KINEMATIC_ACTIONS=1
export UAV_ON_RGB_ONLY=1
export UAV_ON_RGB_COMPRESS=0
export UAV_ON_JPEG_OPTIMIZE=0
export UAV_ON_INLINE_VECTOR_ENV=1

bash scripts/generate_qwen3vl_action_dataset.sh
```

What each option does:

- `UAV_ON_RGB_ONLY=1`: skip depth capture for Qwen3-VL action data, which only
  saves RGB.
- `UAV_ON_RGB_COMPRESS=0`: request raw RGB buffers from AirSim instead of
  compressed image bytes, avoiding simulator-side image compression overhead.
- `UAV_ON_JPEG_OPTIMIZE=0`: avoid extra CPU work while saving JPEGs.
- `UAV_ON_INLINE_VECTOR_ENV=1`: format observations in-process instead of
  sending state through vector worker processes.
- `UAV_ON_KINEMATIC_ACTIONS=1`: use `simSetVehiclePose` for A* data collection
  so the saved samples follow the expert path deterministically.
- A* path cache is enabled by default with `UAV_ON_ASTAR_PLAN_CACHE=1`; it
  reuses the grid path for repeated start/target/voxel configurations and
  recomputes only the yaw-dependent action compression.

Local benchmark outputs were stored under:

`/root/autodl-fs/bench_logs/uavon_sim_perf_local`

| Run | Rows | Total seconds | Reset seconds | Steady rows/s | Speedup vs baseline | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline_bs1_64_kinematic` | 64 | 223.813 | 92.177 | 0.486 | 1.00x | stock RGB+depth, compressed RGB, vector worker, JPEG optimize, kinematic |
| `golden_bs4_64_compressed_nocache` | 64 | 161.830 | 102.272 | 1.075 | 2.21x | same batch as optimized run, compressed RGB, cache disabled |
| `opt_bs4_256_rawrgb_cache_run1` | 256 | 202.860 | 101.932 | 2.536 | 5.22x | raw RGB, RGB-only, inline vector, A* cache |
| `opt_bs4_256_rawrgb_cache_run2` | 256 | 181.743 | 101.703 | 3.198 | 6.58x | repeat run of the same optimized config |

The main remaining simulator-side cost after these changes is image capture and
JPEG save. In `opt_bs4_256_rawrgb_cache_run2`, the non-reset timing split was:

- `env_get_obs`: 36.459 seconds across 63 capture steps, 0.579 seconds/step.
- `save_image`: 21.800 seconds across 256 images, 0.085 seconds/image.
- `env_make_actions`: 13.181 seconds across 63 steps, 0.209 seconds/step.
- `oracle_prepare_batch`: 6.365 seconds, with 3 A* cache hits and 1 miss.

Correctness checks:

```bash
cd /root/autodl-tmp/UAV-ON-sim-perf

python scripts/compare_qwen3vl_generation_outputs.py \
  --baseline /root/autodl-fs/bench_logs/uavon_sim_perf_local/golden_bs4_64_compressed_nocache \
  --candidate /root/autodl-fs/bench_logs/uavon_sim_perf_local/opt_bs4_256_rawrgb_cache_run2 \
  --limit 64 \
  --check-images \
  --max-image-checks 16 \
  --output /root/autodl-fs/bench_logs/uavon_sim_perf_local/correctness_golden64_vs_rawrgb_cache_run2.json

python scripts/compare_qwen3vl_generation_outputs.py \
  --baseline /root/autodl-fs/bench_logs/uavon_sim_perf_local/opt_bs4_256_rawrgb_cache_run1 \
  --candidate /root/autodl-fs/bench_logs/uavon_sim_perf_local/opt_bs4_256_rawrgb_cache_run2 \
  --output /root/autodl-fs/bench_logs/uavon_sim_perf_local/correctness_rawrgb_cache_run1_vs_run2.json
```

The JSONL key fields matched exactly for both checks: `action`, `step_size`,
`frame_index`, `map_name`, `episode_id`, `distance_to_target`, and
`plan_summary`. The raw-RGB images were not byte-identical to AirSim compressed
RGB images; the comparison report records per-image SHA-256 hashes and pixel
differences. Treat raw RGB as label/trajectory-correct and much faster, but not
pixel-identical to the compressed AirSim image transport.

On this local 4080, `batchSize=8` with raw RGB saturated the GPU and stalled in
initial multi-instance capture. Keep the local production recommendation at
`batchSize=4` unless a later run proves a higher batch size is stable.

## Front RGB 3m Balanced 20k Run

Observed setup on 2026-05-11 in `/root/autodl-tmp/UAV-ON`.

Launch the fast front-only 3m approximate data collection with:

```bash
cd /root/autodl-tmp/UAV-ON
bash scripts/tmp/run_front_rgb_3m_20k_balanced_fast_approx.sh
```

The runner starts the train simulator server if port `30000` is not already
served, then launches `scripts/generate_qwen3vl_action_dataset_all_train_scenes.sh`
in the background. It writes:

- Output root:
  `/root/autodl-fs/datasets/uavon_qwen3vl_action_front_rgb_3m_20k_balanced_fast_approx`
- Generator log:
  `/root/autodl-fs/logs/front_rgb_3m_20k_balanced_fast_approx_generator.log`
- Server log:
  `/root/autodl-fs/logs/front_rgb_3m_20k_balanced_fast_approx_server.log`

The run targets 2,000 rows per train scene across 10 scenes. Each per-scene
shard records `collection.variant=front_rgb_3m_20k_balanced_fast_approx`,
`view_mode=front_rgb`, `rgb_only=true`, `front_view_only=true`,
`movement_unit_meters_approx=3.0`, and `movement_unit_non_strict=true` in
`generation_stats.json`. Each JSONL sample also carries the same
`collection_config` and uses the front-RGB prompt version.

Important run settings:

- `UAV_ON_BATCH_SIZE=4`
- `UAV_ON_KINEMATIC_ACTIONS=1`
- `UAV_ON_RGB_ONLY=1`
- `UAV_ON_FRONT_VIEW_ONLY=1`
- `UAV_ON_RGB_COMPRESS=0`
- `UAV_ON_JPEG_OPTIMIZE=0`
- `UAV_ON_INLINE_VECTOR_ENV=1`
- `UAV_ON_ASTAR_VOXEL_RESOLUTION=3.0`
- `UAV_ON_ASTAR_MAX_MOVE_VOXELS=1`
- `UAV_ON_ASTAR_PLAN_CACHE=1`

`UAV_ON_FRONT_VIEW_ONLY=1` changes the AirSim request itself to camera `0`
only; it is not just a post-save crop from four-view images.

### Image View Modes And Sizes

AirSim RGB cameras are configured at `512x512` in
`airsim_plugin/AirVLNSimulatorServerTool.py`.

The default `four_view` Qwen3-VL action sample decodes four RGB camera images
in this order:

```text
front, left, right, down
```

The generator resizes each camera image to the minimum square tile size and
stitches them into one 2x2 grid. With the current `512x512` RGB cameras, the
stitched image is therefore `1024x1024`.

The `front_rgb` mode records only camera `0` and stores one `512x512` image.
The `four_view_images` mode stores four separate `512x512` images per sample
instead of a stitched grid.

Observed with the local Qwen3-VL processor:

- `512x512` front-only image: 256 image pad tokens.
- `1024x1024` stitched four-view grid: 1024 image pad tokens.

So front-only data reduces the image-token count to about one quarter of the
stitched four-view grid. If a future run switches from stitched `four_view` to
`four_view_images`, verify the training/eval code consumes the same message
format; the prompt version and image count differ.

### Complete-Trajectory Collection

The default generator treats `--max_samples` as a hard row limit. If a run asks
for 2,000 rows, it stops immediately after row 2,000 and the final batch can
contain truncated episodes.

For trajectory training data, use complete-trajectory mode:

```bash
cd /root/autodl-tmp/UAV-ON
bash scripts/run_front_rgb_3m_20k_balanced_fast_approx_complete_traj.sh
```

This runner uses the same front-only, RGB-only, approximate 3m settings as the
fast run, but writes to a separate output root:

```text
/root/autodl-fs/datasets/uavon_qwen3vl_action_front_rgb_3m_20k_balanced_fast_approx_complete_traj
```

Complete-trajectory mode sets `UAV_ON_GENERATE_COMPLETE_TRAJECTORIES=1`. In
that mode the per-scene target is a lower bound: the generator stops launching
new batches after reaching the target, then lets the currently active batch
finish naturally at the A* `stop` action. The final row count can therefore be
slightly above the target, but every collected episode in a clean run should
have a terminal `stop` sample.

The mode is recorded in both `generation_stats.json` and per-sample
`collection_config` as:

- `complete_trajectories=true`
- `sample_target_is_minimum=true`

Each JSONL row also contains `is_terminal_action`, so complete trajectories can
be reconstructed by grouping on `(map_name, episode_id)`, sorting by
`frame_index`, and checking that the last row has `action=stop` or
`is_terminal_action=true`.

If the scene shard manifest was cleaned up, the all-scenes wrapper rebuilds it
from `/root/autodl-fs/datasets/uavon_train_generation_balanced.json` with
`scripts/build_uavon_scene_shards.py` before launching collection.

### Four-View Separate-Image Run

To collect the same approximate 3m, RGB-only, balanced 20k dataset but preserve
four independent camera images per sample instead of stitching them into one
grid, run:

```bash
cd /root/autodl-tmp/UAV-ON
bash scripts/run_fourview_images_3m_20k_balanced_fast_approx.sh
```

The output root is:

```text
/root/autodl-fs/datasets/uavon_qwen3vl_action_4view_images_3m_20k_balanced_fast_approx
```

Each sample stores:

- `image_paths`: four image paths.
- `image_view_names`: `["front", "left", "right", "down"]`.
- `images`: view-name/path pairs.
- `messages[1].content`: four image entries followed by the text prompt.

The collection metadata records `view_mode=four_view_images`,
`front_view_only=false`, `separate_view_images=true`, and
`image_count_per_sample=4`.

## A* Turn-Cooldown Variant

Observed on 2026-05-11 in `/root/autodl-tmp/UAV-ON`.

The default A* behavior is unchanged. Turn cooldown is disabled unless both
parameters are positive:

- `--astar_turn_cooldown_after`: after this many turn actions, start cooldown.
- `--astar_turn_cooldown_steps`: number of following non-turn actions during
  which additional turns are blocked.

The shell wrappers forward matching environment variables:

```bash
cd /root/autodl-tmp/UAV-ON
export UAV_ON_ASTAR_TURN_COOLDOWN_AFTER=1
export UAV_ON_ASTAR_TURN_COOLDOWN_STEPS=2

bash scripts/generate_qwen3vl_action_dataset.sh
```

For a single already-open AirSim scene, smoke it with:

```bash
cd /root/autodl-tmp/UAV-ON
python scripts/smoke_astar_existing_scene.py \
  --dataset astar_smoke_citypark_1.json \
  --output-dir astar_logs/citypark_turn_cooldown_after1_steps2 \
  --port 30100 \
  --max-actions 100 \
  --execution-mode kinematic \
  --astar-turn-cooldown-after 1 \
  --astar-turn-cooldown-steps 2
```

Observed result for that smoke on 2026-05-11: the planner produced
`turn_cooldown=on`, `path_len=32`, `actions=34`, and `turns=2`; the first 30
kinematic actions wrote
`astar_logs/citypark_turn_cooldown_after1_steps2/log/trajectory.jsonl` without
collisions.

When enabled, the planner searches in action space with state
`(x, y, z, heading, cooldown_remaining, turns_since_cooldown)`. This keeps the
cooldown constraint inside planning instead of smoothing a path afterward. The
A* path cache is disabled for this variant because the selected path depends on
start yaw and cooldown state, not only on the voxel grid and target.

`generation_stats.json` records the requested parameters under `astar` and the
current generator-process aggregate under `astar_turn_cooldown`:

- `action_count`, `turn_count`, `turn_density`
- `alternating_turn_count`
- `cooldown_triggers`
- `suppressed_turn_candidates`
- `expanded_states`

Do not resume an existing output directory with different turn-cooldown
settings. The generator now rejects that case so one JSONL shard does not mix
labels produced by different A* variants. Use a fresh output directory or
`--overwrite` when changing these parameters.

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
