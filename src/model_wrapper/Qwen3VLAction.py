from __future__ import annotations

import json
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch

from common.uavon_action_schema import (
    QWEN3VL_ACTION_SYSTEM_PROMPT,
    QWEN3VL_TRAJECTORY_ACTION_SYSTEM_PROMPT,
    UAVON_VALID_ACTIONS,
    build_qwen3vl_action_user_prompt,
    build_qwen3vl_trajectory_action_user_prompt,
    extract_uavon_action,
    extract_uavon_actions,
)
from common.runtime_config import runtime_config_from_env
from model_wrapper.base_model import BaseModelWrapper
from src.common.param import args


def _latest_checkpoint(path: str) -> str:
    root = Path(path)
    checkpoints = [p for p in root.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return str(root)

    def step(checkpoint: Path) -> int:
        match = re.search(r"checkpoint-(\d+)$", checkpoint.name)
        return int(match.group(1)) if match else -1

    return str(max(checkpoints, key=step))


class Qwen3VLAction(BaseModelWrapper):
    def __init__(self, fixed: bool, batch_size: int):
        super().__init__()
        self.fixed = fixed
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = args.qwen3vl_model_path
        self.adapter_path = _latest_checkpoint(args.qwen3vl_adapter_path) if args.qwen3vl_adapter_path else ""
        self.max_new_tokens = int(args.qwen3vl_max_new_tokens)
        self.runtime_config = runtime_config_from_env()
        self.view_mode = "front_rgb" if self.runtime_config.front_view_only else "four_view"
        self.sample_mode = str(args.qwen3vl_eval_sample_mode)
        self.trajectory_max_steps = int(args.qwen3vl_trajectory_max_steps)
        self.trajectory_kv_cache = bool(args.qwen3vl_trajectory_kv_cache)
        if self.sample_mode not in {"single", "trajectory"}:
            raise ValueError(f"Unsupported qwen3vl_eval_sample_mode: {self.sample_mode}")
        if self.trajectory_max_steps < 0:
            raise ValueError("--qwen3vl_trajectory_max_steps must be >= 0")
        self.processor, self.model = self._load_model()
        self._trajectory_cache_by_batch: dict[int, dict[str, Any]] = {}
        self.raw_log_path = Path(args.eval_save_path) / "qwen3vl_action_raw_outputs.jsonl"
        self.raw_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_log_handle = self.raw_log_path.open("a", encoding="utf-8")

    def _load_model(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor_source = (
            self.adapter_path
            if self.adapter_path and (Path(self.adapter_path) / "preprocessor_config.json").exists()
            else self.model_path
        )
        processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            attn_implementation=args.qwen3vl_attn_implementation,
            trust_remote_code=True,
        )
        if self.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.to(self.device)
        model.eval()
        model.config.use_cache = True
        return processor, model

    def _image_from_obs(self, image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (bytes, bytearray)):
            return Image.open(BytesIO(image)).convert("RGB")
        array = np.asarray(image)
        if array.ndim == 3 and array.shape[-1] == 3:
            return Image.fromarray(array.astype(np.uint8)).convert("RGB")
        raise ValueError(f"Unsupported UAV-ON RGB observation type: {type(image)!r}")

    def _make_fourview_grid(self, images: list[Any]) -> Image.Image:
        if len(images) != 4:
            raise ValueError(f"Expected four UAV-ON RGB images, got {len(images)}")
        pil_images = [self._image_from_obs(image) for image in images]
        tile_size = min(min(image.size) for image in pil_images)
        labels = ["front", "left", "right", "down"]
        tiles = []
        for image, label in zip(pil_images, labels):
            tile = image.resize((tile_size, tile_size), Image.Resampling.BILINEAR)
            draw = ImageDraw.Draw(tile)
            draw.rectangle((0, 0, tile_size, 30), fill=(0, 0, 0))
            draw.text((8, 8), label, fill=(255, 255, 255))
            tiles.append(tile)
        grid = Image.new("RGB", (tile_size * 2, tile_size * 2), (0, 0, 0))
        grid.paste(tiles[0], (0, 0))
        grid.paste(tiles[1], (tile_size, 0))
        grid.paste(tiles[2], (0, tile_size))
        grid.paste(tiles[3], (tile_size, tile_size))
        return grid

    def _observation_image(self, images: list[Any]) -> Image.Image:
        if self.view_mode == "front_rgb":
            if len(images) != 1:
                raise ValueError(f"Expected one front UAV-ON RGB image, got {len(images)}")
            return self._image_from_obs(images[0])
        return self._make_fourview_grid(images)

    def _task_instruction(self, observation: dict[str, Any]) -> str:
        target_name = str(observation.get("object_name") or "target object").strip()
        size = str(observation.get("object_size") or "").strip()
        description = str(observation.get("description") or "").strip()
        parts = [f"Find and stop near the target object: {target_name}."]
        if size:
            parts.append(f"Target size: {size}.")
        if description:
            parts.append(f"Target description: {description}")
        return " ".join(parts)

    def _messages(self, instruction: str) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": QWEN3VL_ACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": build_qwen3vl_action_user_prompt(instruction, view_mode=self.view_mode)},
                ],
            },
        ]

    def _trajectory_messages(self, instruction: str, num_steps: int) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": build_qwen3vl_trajectory_action_user_prompt(
                    instruction,
                    num_steps=num_steps,
                    view_mode=self.view_mode,
                ),
            }
        ]
        for offset in range(1, num_steps + 1):
            content.append({"type": "text", "text": f"Step {offset} observation:"})
            content.append({"type": "image"})
        return [
            {"role": "system", "content": QWEN3VL_TRAJECTORY_ACTION_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def _select_trajectory_observations(self, episode: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not episode:
            raise ValueError("Expected at least one observation in episode")
        observations = self._dedupe_trajectory_observations(episode)
        if self.trajectory_max_steps <= 0 or len(observations) <= self.trajectory_max_steps:
            return observations
        return observations[-self.trajectory_max_steps :]

    def _dedupe_trajectory_observations(self, episode: list[dict[str, Any]]) -> list[dict[str, Any]]:
        observations: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for index, observation in enumerate(episode):
            key = self._trajectory_observation_key(observation, index)
            if key in seen:
                continue
            seen.add(key)
            observations.append(observation)
        return observations

    def _trajectory_observation_key(self, observation: dict[str, Any], index: int) -> tuple[Any, ...]:
        state = observation.get("sensors", {}).get("state", {})
        position = state.get("position")
        quaternion = state.get("quaternionr")
        step = observation.get("step", None)
        if position is None or quaternion is None:
            return ("fallback", step, index)

        def normalize(values: Any) -> tuple[float, ...]:
            return tuple(round(float(value), 6) for value in values)

        return ("pose", step, normalize(position), normalize(quaternion))

    def _prepare_trajectory_input(self, episode: list[dict[str, Any]]) -> dict[str, Any]:
        raw_source_steps = [observation.get("step", -1) for observation in episode]
        observations = self._select_trajectory_observations(episode)
        current = observations[-1]
        images = [self._observation_image(observation["rgb"]) for observation in observations]
        instruction = self._task_instruction(current)
        return {
            "images": images,
            "instruction": instruction,
            "step": current.get("step", -1),
            "trajectory_raw_num_steps": len(episode),
            "trajectory_raw_source_steps": raw_source_steps,
            "trajectory_num_steps": len(observations),
            "trajectory_source_steps": [observation.get("step", -1) for observation in observations],
        }

    def prepare_inputs(self, episodes, fixed):
        prompts = []
        inputs = []
        for episode in episodes:
            if self.sample_mode == "trajectory":
                item = self._prepare_trajectory_input(episode)
                prompts.append(item["instruction"])
                inputs.append(item)
                continue
            observation = episode[-1]
            image = self._observation_image(observation["rgb"])
            instruction = self._task_instruction(observation)
            prompts.append(instruction)
            inputs.append({"image": image, "instruction": instruction, "step": observation.get("step", -1)})
        return inputs, prompts

    def _default_step_size(self, action: str, parsed_step_size: float | None, fixed: bool) -> float:
        if parsed_step_size is not None:
            return float(parsed_step_size)
        if fixed:
            return 0.0
        if action in {"rotl", "rotr"}:
            return float(args.rotateAngle)
        if action in {"ascend", "descend"}:
            return float(args.z_step_size)
        if action in {"forward", "left", "right"}:
            return float(args.xOy_step_size)
        return 0.0

    @torch.inference_mode()
    def _encode_messages(
        self,
        messages: list[dict[str, Any]],
        images: list[Image.Image],
        *,
        add_generation_prompt: bool = True,
    ) -> dict[str, Any]:
        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        inputs = self.processor(text=[prompt_text], images=images, return_tensors="pt")
        return {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}

    @torch.inference_mode()
    def _generate_from_messages(self, messages: list[dict[str, Any]], images: list[Image.Image]) -> str:
        inputs = self._encode_messages(messages, images, add_generation_prompt=True)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        trimmed = generated_ids[:, inputs["input_ids"].shape[-1] :]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    def _generate(self, image: Image.Image, instruction: str) -> str:
        return self._generate_from_messages(self._messages(instruction), [image])

    def _generate_trajectory(self, images: list[Image.Image], instruction: str) -> str:
        messages = self._trajectory_messages(instruction, len(images))
        return self._generate_from_messages(messages, images)

    def _inner_qwen_model(self) -> Any | None:
        inner = getattr(self.model, "model", None)
        if inner is not None and hasattr(inner, "get_rope_index"):
            return inner
        base_model = getattr(self.model, "base_model", None)
        candidate = getattr(getattr(base_model, "model", None), "model", None)
        if candidate is not None and hasattr(candidate, "get_rope_index"):
            return candidate
        return None

    def _count_image_groups(self, mm_token_type_ids: torch.Tensor, end: int) -> int:
        values = mm_token_type_ids[0, :end].detach().cpu().tolist()
        count = 0
        in_image = False
        for value in values:
            if int(value) == 1:
                if not in_image:
                    count += 1
                    in_image = True
            else:
                in_image = False
        return count

    def _safe_cache_boundary(self, mm_token_type_ids: torch.Tensor, length: int) -> int:
        if length <= 0:
            return 0
        values = mm_token_type_ids[0].detach().cpu().tolist()
        length = min(length, len(values))
        while length > 0 and int(values[length - 1]) != 0:
            length -= 1
        return length

    def _longest_common_prefix(self, previous_ids: torch.Tensor, current_ids: torch.Tensor) -> int:
        limit = min(previous_ids.numel(), current_ids.numel())
        if limit <= 0:
            return 0
        prev = previous_ids[:limit].to(current_ids.device)
        matches = prev.eq(current_ids[:limit])
        mismatch = (~matches).nonzero(as_tuple=False)
        if mismatch.numel() == 0:
            return limit
        return int(mismatch[0].item())

    def _slice_image_inputs(
        self,
        inputs: dict[str, Any],
        *,
        start_image_index: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        image_grid_thw = inputs.get("image_grid_thw")
        pixel_values = inputs.get("pixel_values")
        if image_grid_thw is None or pixel_values is None:
            return None, None
        if start_image_index <= 0:
            return pixel_values, image_grid_thw
        if start_image_index >= int(image_grid_thw.shape[0]):
            return None, None
        patch_counts = image_grid_thw.prod(dim=1).detach().cpu().tolist()
        patch_start = int(sum(patch_counts[:start_image_index]))
        return pixel_values[patch_start:], image_grid_thw[start_image_index:]

    def _compute_position_ids(self, inputs: dict[str, Any]) -> torch.Tensor | None:
        inner = self._inner_qwen_model()
        if inner is None:
            return None
        input_ids = inputs.get("input_ids")
        mm_token_type_ids = inputs.get("mm_token_type_ids")
        if input_ids is None or mm_token_type_ids is None:
            return None
        image_grid_thw = inputs.get("image_grid_thw")
        video_grid_thw = inputs.get("video_grid_thw")
        if image_grid_thw is None and video_grid_thw is None:
            return None
        position_ids, rope_deltas = inner.get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=inputs.get("attention_mask"),
            mm_token_type_ids=mm_token_type_ids,
        )
        inner.rope_deltas = rope_deltas
        return position_ids

    def _model_forward_prefill(
        self,
        inputs: dict[str, Any],
        *,
        past_key_values: Any | None = None,
        start: int = 0,
        position_ids: torch.Tensor | None = None,
        image_start_index: int = 0,
    ) -> Any:
        input_ids = inputs["input_ids"][:, start:]
        if input_ids.shape[-1] <= 0:
            raise ValueError("Cannot prefill an empty Qwen3-VL suffix")
        full_attention_mask = inputs.get("attention_mask")
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": full_attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
            "return_dict": True,
            "logits_to_keep": 1,
        }
        if position_ids is not None:
            kwargs["position_ids"] = position_ids[:, :, start:]
        if "mm_token_type_ids" in inputs:
            kwargs["mm_token_type_ids"] = inputs["mm_token_type_ids"][:, start:]
        if past_key_values is not None:
            past_len = int(past_key_values.get_seq_length())
            kwargs["cache_position"] = torch.arange(
                past_len,
                past_len + input_ids.shape[-1],
                device=self.device,
                dtype=torch.long,
            )
        pixel_values, image_grid_thw = self._slice_image_inputs(inputs, start_image_index=image_start_index)
        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values
            kwargs["image_grid_thw"] = image_grid_thw
        return self.model(**kwargs)

    def _next_position_ids(self, cache_len: int) -> torch.Tensor | None:
        inner = self._inner_qwen_model()
        rope_deltas = getattr(inner, "rope_deltas", None) if inner is not None else None
        if rope_deltas is None:
            return None
        position_ids = torch.arange(cache_len, cache_len + 1, device=self.device, dtype=torch.long)
        position_ids = position_ids.view(1, 1, -1).expand(3, 1, -1)
        return position_ids + rope_deltas.to(device=self.device).view(1, 1, 1)

    def _decode_with_prompt_cache(self, cache: Any, first_logits: torch.Tensor, prompt_len: int) -> str:
        generated: list[int] = []
        next_token = torch.argmax(first_logits[:, -1, :], dim=-1, keepdim=True)
        eos_token_id = self.processor.tokenizer.eos_token_id
        for _ in range(self.max_new_tokens):
            token_id = int(next_token[0, 0].item())
            if token_id == eos_token_id:
                break
            generated.append(token_id)
            cache_len = int(cache.get_seq_length())
            attention_mask = torch.ones((1, cache_len + 1), device=self.device, dtype=torch.long)
            kwargs: dict[str, Any] = {
                "input_ids": next_token,
                "attention_mask": attention_mask,
                "past_key_values": cache,
                "cache_position": torch.arange(cache_len, cache_len + 1, device=self.device, dtype=torch.long),
                "use_cache": True,
                "return_dict": True,
                "logits_to_keep": 1,
            }
            position_ids = self._next_position_ids(cache_len)
            if position_ids is not None:
                kwargs["position_ids"] = position_ids
            outputs = self.model(**kwargs)
            cache = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        cache.crop(prompt_len)
        if not generated:
            return ""
        return self.processor.batch_decode(
            [generated],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def _generate_trajectory_with_kv_cache(self, batch_index: int, item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        messages = self._trajectory_messages(item["instruction"], len(item["images"]))
        inputs = self._encode_messages(messages, item["images"], add_generation_prompt=True)
        current_ids = inputs["input_ids"][0]
        source_steps = list(item.get("trajectory_source_steps", []))
        position_ids = self._compute_position_ids(inputs)
        cache_state = self._trajectory_cache_by_batch.get(batch_index)
        cache_hit = False
        cache_reset_reason = "empty"
        lcp_len = 0
        image_start_index = 0

        if cache_state is not None:
            previous_steps = list(cache_state.get("source_steps", []))
            can_append = (
                bool(previous_steps)
                and len(source_steps) > len(previous_steps)
                and source_steps[: len(previous_steps)] == previous_steps
            )
            if can_append:
                lcp_len = self._longest_common_prefix(cache_state["input_ids"], current_ids)
                if "mm_token_type_ids" in inputs:
                    lcp_len = self._safe_cache_boundary(inputs["mm_token_type_ids"], lcp_len)
                cache_state["cache"].crop(lcp_len)
                image_start_index = self._count_image_groups(inputs["mm_token_type_ids"], lcp_len) if "mm_token_type_ids" in inputs else 0
                cache_hit = lcp_len > 0
                cache_reset_reason = "append"
            else:
                cache_state = None
                cache_reset_reason = "window_changed"

        if cache_state is None or not cache_hit:
            outputs = self._model_forward_prefill(inputs, position_ids=position_ids)
            prompt_cache = outputs.past_key_values
            prompt_len = int(inputs["input_ids"].shape[-1])
            cache_hit = False
        else:
            outputs = self._model_forward_prefill(
                inputs,
                past_key_values=cache_state["cache"],
                start=lcp_len,
                position_ids=position_ids,
                image_start_index=image_start_index,
            )
            prompt_cache = outputs.past_key_values
            prompt_len = int(inputs["input_ids"].shape[-1])

        raw_text = self._decode_with_prompt_cache(prompt_cache, outputs.logits, prompt_len)
        self._trajectory_cache_by_batch[batch_index] = {
            "cache": prompt_cache,
            "input_ids": current_ids.detach().cpu(),
            "source_steps": source_steps,
        }
        return raw_text, {
            "kv_cache_enabled": True,
            "kv_cache_hit": cache_hit,
            "kv_cache_lcp_tokens": lcp_len,
            "kv_cache_prompt_tokens": prompt_len,
            "kv_cache_reset_reason": cache_reset_reason,
        }

    def _parse_action_item(self, item: dict[str, Any]) -> tuple[str, tuple[str, float | None] | None, dict[str, Any]]:
        if self.sample_mode == "trajectory":
            if self.trajectory_kv_cache:
                raw_text, cache_detail = self._generate_trajectory_with_kv_cache(
                    int(item.get("batch_index", 0)),
                    item,
                )
            else:
                raw_text = self._generate_trajectory(item["images"], item["instruction"])
                cache_detail = {"kv_cache_enabled": False}
            parsed_actions = extract_uavon_actions(raw_text)
            selected_action_offset = None
            if parsed_actions:
                trajectory_num_steps = int(item.get("trajectory_num_steps", 0))
                selected_action_offset = min(len(parsed_actions), max(trajectory_num_steps, 1)) - 1
                parsed = parsed_actions[selected_action_offset]
            else:
                parsed = None
            detail = {
                "raw_text": raw_text,
                "trajectory_raw_num_steps": item.get("trajectory_raw_num_steps", 0),
                "trajectory_raw_source_steps": item.get("trajectory_raw_source_steps", []),
                "trajectory_num_steps": item.get("trajectory_num_steps", 0),
                "trajectory_source_steps": item.get("trajectory_source_steps", []),
                "parsed_actions": [action for action, _step_size in parsed_actions],
                "parsed_action_count": len(parsed_actions),
                "selected_action_offset": selected_action_offset,
                **cache_detail,
            }
            return raw_text, parsed, detail

        raw_text = self._generate(item["image"], item["instruction"])
        parsed = extract_uavon_action(raw_text)
        return raw_text, parsed, {"raw_text": raw_text}

    def run(self, inputs, fixed, prompt_info_list=None):
        actions = []
        step_sizes = []
        dones = []
        for batch_index, item in enumerate(inputs):
            try:
                item["batch_index"] = batch_index
                raw_text, parsed, parse_detail = self._parse_action_item(item)
                status = "success" if parsed is not None else "parse_failed"
            except Exception as exc:
                raw_text = f"{type(exc).__name__}: {exc}"
                parsed = None
                parse_detail = {"raw_text": raw_text}
                status = "model_failed"
            if parsed is None:
                action, parsed_step_size = "stop", 0.0
            else:
                action, parsed_step_size = parsed
                if action not in UAVON_VALID_ACTIONS:
                    action, parsed_step_size = "stop", 0.0
                    status = "parse_failed"
            step_size = self._default_step_size(action, parsed_step_size, fixed)
            actions.append(action)
            step_sizes.append(step_size)
            dones.append(action == "stop")
            self.raw_log_handle.write(
                json.dumps(
                    {
                        "batch_index": batch_index,
                        "step": item.get("step", -1),
                        "status": status,
                        "action": action,
                        "step_size": step_size,
                        "view_mode": self.view_mode,
                        "sample_mode": self.sample_mode,
                        "raw_text": raw_text,
                        "instruction": item.get("instruction", ""),
                        **parse_detail,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            self.raw_log_handle.flush()
        return actions, step_sizes, dones

    def __del__(self):
        handle = getattr(self, "raw_log_handle", None)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
