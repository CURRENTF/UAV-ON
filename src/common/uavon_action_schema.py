from __future__ import annotations

import re
from typing import Final

UAVON_ACTION_PROMPT_VERSION: Final[str] = "qwen3vl_action_v1"

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
QWEN3VL_ACTION_OUTPUT_FORMAT: Final[str] = "Return only the action name."
QWEN3VL_ACTION_CHOICE_INSTRUCTION: Final[str] = "Choose the A* action for the current state."
QWEN3VL_ALLOWED_ACTIONS_TEXT: Final[str] = ", ".join(UAVON_ACTIONS)

_ACTION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(action) for action in UAVON_ACTIONS) + r")\b"
    r"(?:\s+([-+]?\d+(?:\.\d+)?))?"
)


def build_qwen3vl_action_user_prompt(instruction: str) -> str:
    return (
        "Current four-view observation is provided as one 2x2 image grid "
        "(front, left, right, down).\n"
        f"Task instruction:\n{instruction.strip()}\n"
        f"{QWEN3VL_ACTION_CHOICE_INSTRUCTION} "
        f"Allowed actions: {QWEN3VL_ALLOWED_ACTIONS_TEXT}.\n"
        f"{QWEN3VL_ACTION_OUTPUT_FORMAT}"
    )


def extract_uavon_action(text: str) -> tuple[str, float | None] | None:
    match = _ACTION_PATTERN.search(text)
    if match is None:
        return None
    step_size = float(match.group(2)) if match.group(2) is not None else None
    return match.group(1), step_size
