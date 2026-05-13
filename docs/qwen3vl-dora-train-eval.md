# Qwen3-VL DoRA Train/Eval Pipeline

Observed on 2026-05-11 in AutoDL under `/root/autodl-tmp/UAV-ON`.

## Purpose

Run the Qwen3-VL UAV-ON action policy with a reproducible shell pipeline:

1. Smoke-test closed-loop Qwen3-VL + PEFT adapter evaluation.
2. Rebalance the 20k A* action JSONL toward non-`forward` actions.
3. Train a DoRA adapter for 1000 steps.
4. Evaluate the trained adapter on UAV-ON.

## Scripts

- Smoke eval:
  `/root/autodl-tmp/UAV-ON/scripts/qwen3vl_dora_smoke_eval.sh`
- Full pipeline:
  `/root/autodl-tmp/UAV-ON/scripts/run_qwen3vl_dora_nonforward_1k_pipeline.sh`
- Dataset rebalance helper:
  `/root/autodl-tmp/LightningVLN/scripts/make_qwen3vl_action_rebalanced_dataset.py`
- Qwen3-VL training wrapper:
  `/root/autodl-tmp/LightningVLN/scripts/train_qwen3vl_action_baseline.sh`

`scripts/tmp` is gitignored and should be used only for local one-off debugging.

## Training Settings

The full pipeline passes these settings to LightningVLN:

- `PEFT_METHOD=dora`
- `PER_DEVICE_TRAIN_BATCH_SIZE=4`
- `GRADIENT_ACCUMULATION_STEPS=1`
- `LEARNING_RATE=1e-5`
- `LR_SCHEDULER_TYPE=linear`
- `MAX_STEPS=1000`
- `SAVE_STEPS=200`
- `SAVE_TOTAL_LIMIT=5`
- `REPORT_TO=wandb` by default; set `REPORT_TO=none` explicitly to disable it

## Action Schema

Qwen3-VL action prompts and parsers now use prompt/schema version
`qwen3vl_action_v1`.

The nominal UAV-ON action set is:

- `forward`
- `left`
- `right`
- `rotl`
- `rotr`
- `ascend`
- `descend`
- `stop`

The current A* oracle can execute in the UAV-ON environment but emits labels by
rotating toward a direction and then moving forward. Its label set is therefore:

- `forward`
- `rotl`
- `rotr`
- `ascend`
- `descend`
- `stop`

`left` and `right` are valid executable actions and remain in the model-facing
allowed action list, but they are not present in the current A* SFT labels unless
the data generator or oracle is changed.

## Data Rebalancing

Source:

`/root/autodl-fs/datasets/uavon_qwen3vl_action_train_astar_20k/train.jsonl`

Output:

`/root/autodl-fs/datasets/uavon_qwen3vl_action_train_astar_20k_nonforward80_seed42/train.jsonl`

The default rebalance keeps all non-`forward` examples and samples enough
`forward` examples to target a 20% final `forward` fraction. The observed output
distribution was:

- `ascend`: 9
- `descend`: 1889
- `forward`: 2135
- `rotl`: 3198
- `rotr`: 3193
- `stop`: 253

## Run

Smoke eval only:

```bash
cd /root/autodl-tmp/UAV-ON
SMOKE_TASKS=1 UAV_ON_MAX_ACTIONS=1 \
UAV_ON_EVAL_SAVE_PATH=/root/autodl-fs/evals/qwen3vl_dora_smoke_eval_current \
bash scripts/qwen3vl_dora_smoke_eval.sh
```

Full train and eval pipeline:

```bash
cd /root/autodl-tmp/UAV-ON
RUN_SMOKE=0 REBALANCE_OVERWRITE=1 \
bash scripts/run_qwen3vl_dora_nonforward_1k_pipeline.sh
```

Detached launch:

```bash
cd /root/autodl-tmp/UAV-ON
setsid bash -lc 'cd /root/autodl-tmp/UAV-ON && RUN_SMOKE=0 REBALANCE_OVERWRITE=1 bash scripts/run_qwen3vl_dora_nonforward_1k_pipeline.sh' \
  >> /root/autodl-fs/logs/qwen3vl_nonforward80_dora_1k_pipeline_launcher.log 2>&1 < /dev/null &
```

## Outputs

- Rebalanced stats:
  `/root/autodl-fs/datasets/uavon_qwen3vl_action_train_astar_20k_nonforward80_seed42/rebalance_stats.json`
- Checkpoints:
  `/root/autodl-fs/checkpoints/uavon_qwen3vl_action_astar_20k_nonforward80_dora_bs4_lr1e-5_linear_1k`
- Final eval:
  `/root/autodl-fs/evals/uavon_qwen3vl_action_nonforward80_dora_bs4_lr1e-5_linear_1k_citypark_slum_20`
- Pipeline log:
  `/root/autodl-fs/logs/qwen3vl_nonforward80_dora_1k_pipeline_orchestrator.log`
- Train log:
  `/root/autodl-fs/logs/qwen3vl_nonforward80_dora_1k_train.log`
- Eval log:
  `/root/autodl-fs/logs/qwen3vl_nonforward80_dora_1k_eval.log`
- Summary log:
  `/root/autodl-fs/logs/qwen3vl_nonforward80_dora_1k_summary.log`
- Runtime config:
  `<eval_root>/runtime_config.launch.json` and `<eval_root>/runtime_config.json`
- Pipeline metadata:
  `<eval_root>/pipeline_metadata.json`

## Notes

The smoke eval uses the `kv` conda environment because it has the Qwen3-VL
runtime dependencies. On this machine, `yacs==0.1.8` and `numba==0.65.1` were
added to `kv` after the first smoke attempt exposed missing imports from the
UAV-ON environment code.

Runtime behavior variables are centralized in `src/common/runtime_config.py`.
See `docs/runtime-config.md` for the list of `UAV_ON_*` variables that affect
data generation and closed-loop evaluation.

## Front-Only Training Handoff

Observed on 2026-05-11: a front-camera-only 20k data run was generated under:

```text
/root/autodl-fs/datasets/uavon_qwen3vl_action_front_rgb_3m_20k_balanced_fast_approx
```

This dataset is scene-sharded: each scene subdirectory has its own
`train.jsonl`, and the root does not contain a training JSONL by default.
LightningVLN's current Qwen3-VL trainer expects one JSONL, so the scene shards
must be merged before training. The observed merged path was:

```text
/root/autodl-fs/datasets/uavon_qwen3vl_action_front_rgb_3m_20k_balanced_fast_approx/train_combined.jsonl
```

When merging, prefix each row's `image` path and any image entries inside
`messages` with the scene directory name. The per-scene shard JSONL paths are
relative to their scene directory, while the combined JSONL is rooted at the
dataset root.

Observed no-resampling distribution:

- `forward`: 12253
- `rotr`: 2995
- `rotl`: 2943
- `stop`: 914
- `descend`: 810
- `ascend`: 85

Observed LightningVLN run settings for the front-only experiment:

- `PEFT_METHOD=dora`
- `PER_DEVICE_TRAIN_BATCH_SIZE=8`
- `GRADIENT_ACCUMULATION_STEPS=1`
- `LEARNING_RATE=1e-4`
- `LR_SCHEDULER_TYPE=linear`
- `MAX_STEPS=1000`
- `SAVE_STEPS=200`
- `SAVE_TOTAL_LIMIT=5`
- `DATALOADER_NUM_WORKERS=4`
- `REPORT_TO=wandb`

Observed output paths:

```text
/root/autodl-fs/checkpoints/uavon_qwen3vl_action_front_rgb_3m_20k_dora_bs8_lr1e-4_linear_1k_workers4
/root/autodl-fs/logs/uavon_qwen3vl_action_front_rgb_bs8_lr1e-4_linear_1k_workers4.log
```

The first attempt used `DATALOADER_NUM_WORKERS=0`; it showed GPU starvation
and was stopped before checkpoint step 200. With 4 workers, a 180-second sample
showed average GPU utilization rising from about 81% to about 98%, and
`gpu_util=0` seconds dropping from 29 to 0.

PyTorch DataLoader workers appear in `top`/`ps` as child `python` processes
with the same command line as the trainer, not as processes named
`dataloader`. Verify them with:

```bash
ps --ppid <train_pid> -o pid,ppid,stat,etime,%cpu,%mem,rss,cmd
```

## Image Size And Qwen3-VL Token Cost

Observed with the local Qwen3-VL processor:

- Front-only RGB samples are `512x512` and occupy 256 image pad tokens.
- The stitched four-view grid is `1024x1024` when each AirSim RGB camera is
  `512x512`, and occupies 1024 image pad tokens.

Front-only training therefore uses about one quarter of the image tokens of the
stitched four-view training image.

## Interrupted Eval Observation

The 2026-05-11 nonforward80 closed-loop eval was manually stopped before a full
final report. The partial results seen during the run had no successes among
completed tasks; completed failures were ending by `step_limit`. Treat that
run as diagnostic only, not as a final benchmark.

## Full Valset Front-Only Eval

Observed on 2026-05-12: the full official validation metadata contains 1000
tasks across 14 scenes. When merging the per-scene JSON files from
`/root/autodl-fs/datasets/UAV-ON-dataset/UAV-ON-data/valset`, make
`episode_id` unique by prefixing the original id with `map_name`; otherwise
eval artifact lookup by `episode_id` can pick the wrong task after scenes are
merged. The local merged full-valset path is:

```text
/root/autodl-fs/datasets/uavon_qwen3vl_eval_full_valset_1000_unique_ids.json
```

All test environment zips were unpacked to:

```text
/root/autodl-fs/datasets/UAV-ON-test-unpacked
```

Some newly unzipped test scenes had `.sh` launchers and `*Linux-Shipping`
binaries without execute bits. Fix permissions after unpacking:

```bash
find /root/autodl-fs/datasets/UAV-ON-test-unpacked -type f \
  \( -name '*.sh' -o -name '*Linux-Shipping' \) -exec chmod a+rx {} +
```

Full front-only Qwen3-VL eval launch script:

```text
scripts/tmp/eval_qwen3vl_front_rgb_bs16_lr1e-5_2k_full_valset_1000_max40.sh
```

It uses `max_actions=40`, `UAV_ON_FRONT_VIEW_ONLY=1`, `UAV_ON_RGB_ONLY=1`,
`UAV_ON_KINEMATIC_ACTIONS=1`, and the bs16/lr1e-5/front-rgb DoRA adapter at:

```text
/root/autodl-fs/checkpoints/uavon_qwen3vl_action_front_rgb_3m_20k_dora_bs16_lr1e-5_linear_2k_workers4
```

The run started on 2026-05-12 used:

```text
/root/autodl-fs/evals/qwen3vl_front_rgb_bs16_lr1e-5_2k_full_valset_1000_max40_20260512_190107
/root/autodl-fs/logs/qwen3vl_front_rgb_bs16_lr1e-5_2k_full_valset_1000_max40_20260512_190107
```

Important: this eval was launched before action step sizes were centralized.
Its raw outputs show the legacy eval defaults (`forward=5`, `ascend/descend=2`,
`rotl/rotr=15`). Treat its closed-loop score as an invalid configuration for
front-RGB 3m/90deg action checkpoints, whose training rows use
`forward/ascend/descend=3.0` and mostly `rotl/rotr=90.0`.

## Trajectory-Mode Qwen3-VL Eval

Observed on 2026-05-13: trajectory-mode Qwen3-VL checkpoints trained by
LightningVLN use `sample_mode=trajectory` with multiple ordered front RGB
observations in one prompt. For the local traj32 full-finetune checkpoint:

```text
/root/autodl-fs/checkpoints/uavon_qwen3vl_action_front_rgb_3m_20k_fullft_traj32_bs1_lr1e-5_linear_2k_workers4
```

`launch_config.json` records:

- `sample_mode=trajectory`
- `trajectory_max_steps=32`
- `trajectory_stride=32`
- `trajectory_min_steps=2`
- `peft_method=none`

The UAV-ON eval wrapper supports this with
`--qwen3vl_eval_sample_mode trajectory`. At each closed-loop step it sends the
latest ordered observation window, capped by `--qwen3vl_trajectory_max_steps`,
using the same prompt shape as training: one action name per line, in the same
order as observations. Eval deduplicates overlapping simulator observation
windows before building the model trajectory, selects the parsed action offset
that corresponds to the current deduplicated observation, and logs
`trajectory_raw_source_steps`, `trajectory_source_steps`, `parsed_actions`,
`parsed_action_count`, and `selected_action_offset` in
`qwen3vl_action_raw_outputs.jsonl`. A healthy traj32 eval should show source
steps growing like `[0]`, `[0, 1]`, `[0, 1, 2]`, not
`[0]`, `[0, 0, 1]`, `[0, 0, 1, 0, 1, 2]` as the model input.

Observed on 2026-05-14: the historical traj32 checkpoint above was trained
before LightningVLN had prefix trajectory sampling. Its training samples were
32-step chunks, so early labels could attend to future observations that are
not available in closed-loop eval. Treat its low loss and score as not-clean
closed-loop evidence until retrained with `TRAJECTORY_SAMPLE_STRATEGY=prefixes`
in LightningVLN.

Trajectory parsing should remain strict. If the model emits fewer action names
than the current deduplicated prefix length, the wrapper records
`parse_failed`, logs `required_action_count` and `parse_error`, and executes the
fallback `stop` action. It should not silently select the last available action
for a later state.

For front-RGB 3m datasets, closed-loop eval must use action step sizes that
match the training labels:

- `UAV_ON_XOY_STEP_SIZE=3`
- `UAV_ON_Z_STEP_SIZE=3`
- `UAV_ON_ROTATE_ANGLE=90`

`scripts/eval_qwen3vl_front_rgb_full_valset.sh` now defaults to `3/3/90`
because it is specific to the front-RGB 3m eval flow. The lower-level generic
`scripts/eval_qwen3vl_action.sh` still exposes explicit step-size arguments,
so record resolved values from `runtime_config.json` for every run. Runs before
2026-05-14 often used legacy defaults (`5/2/15`) because the model outputs only
action names, not step sizes. Those scores are not comparable to proper
3m/90deg evals.

Set `QWEN3VL_TRAJECTORY_KV_CACHE=true` to reuse the prompt KV cache while the
trajectory window grows by appending one new observation. When the window slides
or a new task starts, eval rebuilds the prompt cache to avoid reusing KV entries
for different images. Raw outputs record `kv_cache_enabled`, `kv_cache_hit`,
`kv_cache_lcp_tokens`, `kv_cache_prompt_tokens`, and
`kv_cache_reset_reason`.

Trajectory eval must set `QWEN3VL_MAX_NEW_TOKENS` high enough to emit one action
name for each observation in the prefix. The front-RGB full-valset runner
defaults to 128 tokens for trajectory mode; the wrapper raises an error if
`qwen3vl_max_new_tokens < qwen3vl_trajectory_max_steps`.

Reusable full-valset runner:

```text
scripts/eval_qwen3vl_front_rgb_full_valset.sh
```

Example command for the traj32 full-finetune checkpoint:

```bash
cd /root/autodl-tmp/UAV-ON
RUN_ID=qwen3vl_front_rgb_fullft_traj32_bs1_lr1e-5_2k_full_valset_1000_max40_$(date +%Y%m%d_%H%M%S) \
QWEN3VL_MODEL_PATH=/root/autodl-fs/checkpoints/uavon_qwen3vl_action_front_rgb_3m_20k_fullft_traj32_bs1_lr1e-5_linear_2k_workers4 \
QWEN3VL_ADAPTER_PATH= \
QWEN3VL_EVAL_SAMPLE_MODE=trajectory \
QWEN3VL_TRAJECTORY_MAX_STEPS=32 \
QWEN3VL_TRAJECTORY_KV_CACHE=true \
QWEN3VL_MAX_NEW_TOKENS=128 \
UAV_ON_MAX_ACTIONS=40 \
UAV_ON_XOY_STEP_SIZE=3 \
UAV_ON_Z_STEP_SIZE=3 \
UAV_ON_ROTATE_ANGLE=90 \
UAV_ON_SCENE_BOOT_SECONDS=300 \
setsid bash scripts/eval_qwen3vl_front_rgb_full_valset.sh \
  >/root/autodl-fs/logs/qwen3vl_front_rgb_fullft_traj32_launcher.log 2>&1 < /dev/null &
```
