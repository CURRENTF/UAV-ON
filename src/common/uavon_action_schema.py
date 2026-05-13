from __future__ import annotations

import re
from typing import Final

UAVON_ACTION_PROMPT_VERSION: Final[str] = "qwen3vl_action_v1"
UAVON_FRONT_RGB_ACTION_PROMPT_VERSION: Final[str] = "qwen3vl_action_v1_front_rgb"
UAVON_FOUR_VIEW_IMAGES_ACTION_PROMPT_VERSION: Final[str] = "qwen3vl_action_v1_four_view_images"
UAVON_TRAJECTORY_ACTION_PROMPT_VERSION: Final[str] = "qwen3vl_action_v1_trajectory"

UAVON_ACTIONS: Final[tuple[str, ...]] = (
    "forward",
    "left",
    "right",
    "rotl",
    "rotr",
    "ascend",
    "descend",
    "stop",
)
UAVON_VALID_ACTIONS: Final[frozenset[str]] = frozenset(UAVON_ACTIONS)

# The A* oracle rotates then moves forward instead of emitting lateral actions.
UAVON_ASTAR_LABEL_ACTIONS: Final[tuple[str, ...]] = (
    "forward",
    "rotl",
    "rotr",
    "ascend",
    "descend",
    "stop",
)

QWEN3VL_ACTION_SYSTEM_PROMPT: Final[str] = "You are a UAV navigation policy. Return only one action name."
QWEN3VL_TRAJECTORY_ACTION_SYSTEM_PROMPT: Final[str] = "You are a UAV navigation policy. Return only action names."
QWEN3VL_ACTION_OUTPUT_FORMAT: Final[str] = "Return only the action name."
QWEN3VL_ACTION_CHOICE_INSTRUCTION: Final[str] = "Choose the A* action for the current state."
QWEN3VL_ALLOWED_ACTIONS_TEXT: Final[str] = ", ".join(UAVON_ACTIONS)

_ACTION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(action) for action in UAVON_ACTIONS) + r")\b"
    r"(?:\s+([-+]?\d+(?:\.\d+)?))?"
)


def build_qwen3vl_action_user_prompt(instruction: str, *, view_mode: str = "four_view") -> str:
    if view_mode == "front_rgb":
        observation_text = "Current observation is one front camera RGB image."
    elif view_mode == "four_view_images":
        observation_text = (
            "Current observation is provided as four RGB images in this order: "
            "front, left, right, down."
        )
    elif view_mode == "four_view":
        observation_text = (
            "Current four-view observation is provided as one 2x2 image grid "
            "(front, left, right, down)."
        )
    else:
        raise ValueError(f"Unsupported UAV-ON action prompt view_mode: {view_mode}")
    return (
        f"{observation_text}\n"
        f"Task instruction:\n{instruction.strip()}\n"
        f"{QWEN3VL_ACTION_CHOICE_INSTRUCTION} "
        f"Allowed actions: {QWEN3VL_ALLOWED_ACTIONS_TEXT}.\n"
        f"{QWEN3VL_ACTION_OUTPUT_FORMAT}"
    )


def build_qwen3vl_trajectory_action_user_prompt(instruction: str, *, num_steps: int, view_mode: str = "front_rgb") -> str:
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if view_mode == "front_rgb":
        observation_text = "You are given an ordered UAV trajectory of front camera RGB observations."
    elif view_mode == "four_view":
        observation_text = "You are given an ordered UAV trajectory of four-view 2x2 RGB image-grid observations."
    else:
        raise ValueError(f"Unsupported UAV-ON trajectory action prompt view_mode: {view_mode}")
    return (
        f"{observation_text}\n"
        f"Task instruction:\n{instruction.strip()}\n"
        f"Number of steps: {num_steps}.\n"
        "For each observation, choose the A* action for that state.\n"
        f"Allowed actions: {QWEN3VL_ALLOWED_ACTIONS_TEXT}.\n"
        "Return exactly one action name per line, in the same order as the observations."
    )


def extract_uavon_action(text: str) -> tuple[str, float | None] | None:
    match = _ACTION_PATTERN.search(text)
    if match is None:
        return None
    step_size = float(match.group(2)) if match.group(2) is not None else None
    return match.group(1), step_size


def extract_uavon_actions(text: str) -> list[tuple[str, float | None]]:
    actions: list[tuple[str, float | None]] = []
    for match in _ACTION_PATTERN.finditer(text):
        step_size = float(match.group(2)) if match.group(2) is not None else None
        actions.append((match.group(1), step_size))
    return actions
