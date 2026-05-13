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
    UAVON_VALID_ACTIONS,
    build_qwen3vl_action_user_prompt,
    extract_uavon_action,
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
        self.processor, self.model = self._load_model()
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

    def prepare_inputs(self, episodes, fixed):
        prompts = []
        inputs = []
        for episode in episodes:
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
    def _generate(self, image: Image.Image, instruction: str) -> str:
        messages = self._messages(instruction)
        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt_text], images=[image], return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        trimmed = generated_ids[:, inputs["input_ids"].shape[-1] :]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    def run(self, inputs, fixed, prompt_info_list=None):
        actions = []
        step_sizes = []
        dones = []
        for batch_index, item in enumerate(inputs):
            try:
                raw_text = self._generate(item["image"], item["instruction"])
                parsed = extract_uavon_action(raw_text)
                status = "success" if parsed is not None else "parse_failed"
            except Exception as exc:
                raw_text = f"{type(exc).__name__}: {exc}"
                parsed = None
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
                        "raw_text": raw_text,
                        "instruction": item.get("instruction", ""),
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
