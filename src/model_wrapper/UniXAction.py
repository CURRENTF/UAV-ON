from __future__ import annotations

import json
import math
import os
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch

from model_wrapper.base_model import BaseModelWrapper
from src.common.param import args


LIGHTNING_VLN_ROOT = Path(os.environ.get("LIGHTNING_VLN_ROOT", "/root/autodl-tmp/LightningVLN"))
if str(LIGHTNING_VLN_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(LIGHTNING_VLN_ROOT / "src"))

from lightning_vln.modeling.uni_x.modeling.uni_x_qwen3 import UniQwen3ForCausalLMInference  # noqa: E402
from lightning_vln.training.uavon_action import (  # noqa: E402
    build_uavon_action_prompt_ids,
    build_uavon_task_instruction,
    extract_uavon_action,
)
from lightning_vln.training.vq_encoding import ChameleonVQEncoder  # noqa: E402


VALID_ACTIONS = {"forward", "left", "right", "rotl", "rotr", "ascend", "descend", "stop"}


def _latest_checkpoint(path: str) -> str:
    root = Path(path)
    checkpoints = [p for p in root.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return str(root)

    def step(checkpoint: Path) -> int:
        match = re.search(r"checkpoint-(\d+)$", checkpoint.name)
        return int(match.group(1)) if match else -1

    return str(max(checkpoints, key=step))


class UniXAction(BaseModelWrapper):
    def __init__(self, fixed: bool, batch_size: int):
        super().__init__()
        self.fixed = fixed
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = args.unix_model_path
        self.adapter_path = _latest_checkpoint(args.unix_adapter_path) if args.unix_adapter_path else ""
        self.max_new_tokens = int(args.unix_max_new_tokens)
        self.tokenizer, self.model = self._load_model()
        self.vq_encoder = ChameleonVQEncoder(
            unix_repo_path=args.unix_repo_path or str(LIGHTNING_VLN_ROOT / "src/lightning_vln/modeling/uni_x"),
            vqgan_dir=args.unix_vqgan_dir,
            device=str(self.device),
            batch_size=max(1, int(args.unix_vq_batch_size)),
        )
        self.raw_log_path = Path(args.eval_save_path) / "unix_action_raw_outputs.jsonl"
        self.raw_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_log_handle = self.raw_log_path.open("a", encoding="utf-8")

    def _load_model(self):
        from transformers import AutoTokenizer

        tokenizer_source = self.adapter_path if self.adapter_path and (Path(self.adapter_path) / "tokenizer_config.json").exists() else self.model_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)
        model = UniQwen3ForCausalLMInference.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
        )
        if self.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.to(self.device)
        model.eval()
        model.config.use_cache = True
        return tokenizer, model

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

    def _task_instruction(self, observation: dict[str, Any]) -> str:
        task = {
            "true_name": observation.get("object_name", ""),
            "object_name": observation.get("object_name", ""),
            "size": observation.get("object_size", ""),
            "description": observation.get("description", ""),
        }
        return build_uavon_task_instruction(task)

    def prepare_inputs(self, episodes, fixed):
        prompts = []
        inputs = []
        for episode in episodes:
            observation = episode[-1]
            fourview = self._make_fourview_grid(observation["rgb"])
            vqcode = self.vq_encoder.encode_pil_images([fourview])[0]
            instruction = self._task_instruction(observation)
            prompt_ids = build_uavon_action_prompt_ids(
                tokenizer=self.tokenizer,
                instruction=instruction,
                current_vqcode=vqcode,
            )
            prompts.append(instruction)
            inputs.append({"input_ids": prompt_ids, "instruction": instruction, "step": observation.get("step", -1)})
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
    def _generate(self, prompt_ids: list[int]) -> tuple[str, str, str]:
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        generated: list[int] = []
        status = "parse_failed"
        action_text = "stop"
        for _ in range(self.max_new_tokens):
            attention_mask = torch.ones_like(input_ids)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            next_token = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
            generated.append(next_token)
            raw_text = self.tokenizer.decode(
                [token for token in generated if token < len(self.tokenizer)],
                skip_special_tokens=False,
            )
            parsed = extract_uavon_action(raw_text)
            if parsed is not None:
                action, _ = parsed
                action_text = raw_text
                status = "success" if action in VALID_ACTIONS else "parse_failed"
                break
            if next_token == self.tokenizer.eos_token_id:
                action_text = raw_text
                break
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token]], dtype=torch.long, device=self.device)],
                dim=1,
            )
        else:
            action_text = self.tokenizer.decode(
                [token for token in generated if token < len(self.tokenizer)],
                skip_special_tokens=False,
            )
        return action_text, status, self.tokenizer.decode(
            [token for token in generated if token < len(self.tokenizer)],
            skip_special_tokens=False,
        )

    def run(self, inputs, fixed, prompt_info_list=None):
        actions = []
        step_sizes = []
        dones = []
        for batch_index, item in enumerate(inputs):
            raw_text, status, generated_text = self._generate(item["input_ids"])
            parsed = extract_uavon_action(raw_text)
            if parsed is None:
                action, parsed_step_size = "stop", 0.0
            else:
                action, parsed_step_size = parsed
                if action not in VALID_ACTIONS:
                    action, parsed_step_size = "stop", 0.0
            step_size = self._default_step_size(action, parsed_step_size, fixed)
            done = action == "stop"
            actions.append(action)
            step_sizes.append(step_size)
            dones.append(done)
            self.raw_log_handle.write(
                json.dumps(
                    {
                        "batch_index": batch_index,
                        "step": item.get("step", -1),
                        "status": status,
                        "action": action,
                        "step_size": step_size,
                        "generated_text": generated_text,
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
