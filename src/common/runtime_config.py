from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off"}


def str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else str_to_bool(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(default if value is None else value)


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None else str(value)


@dataclass(frozen=True)
class UavOnRuntimeConfig:
    inline_vector_env: bool = False
    vector_env_start_method: str = "forkserver"
    kinematic_actions: bool = False
    scene_boot_seconds: float = 0.2
    action_timeout_seconds: float = 12.0
    verbose_pose: bool = False
    set_pose_settle_seconds: float = 0.2
    rgb_only: bool = False
    rgb_compress: bool = True
    image_settle_seconds: float = 0.2
    jpeg_optimize: bool = True
    astar_plan_cache: bool = True

    @classmethod
    def from_env(cls, *, scene_boot_seconds_default: float = 0.2) -> "UavOnRuntimeConfig":
        return cls(
            inline_vector_env=env_bool("UAV_ON_INLINE_VECTOR_ENV", False),
            vector_env_start_method=env_str("UAV_ON_VECTOR_ENV_START_METHOD", "forkserver"),
            kinematic_actions=env_bool("UAV_ON_KINEMATIC_ACTIONS", False),
            scene_boot_seconds=env_float("UAV_ON_SCENE_BOOT_SECONDS", scene_boot_seconds_default),
            action_timeout_seconds=env_float("UAV_ON_ACTION_TIMEOUT_SECONDS", 12.0),
            verbose_pose=env_bool("UAV_ON_VERBOSE_POSE", False),
            set_pose_settle_seconds=env_float("UAV_ON_SET_POSE_SETTLE_SECONDS", 0.2),
            rgb_only=env_bool("UAV_ON_RGB_ONLY", False),
            rgb_compress=env_bool("UAV_ON_RGB_COMPRESS", True),
            image_settle_seconds=env_float("UAV_ON_IMAGE_SETTLE_SECONDS", 0.2),
            jpeg_optimize=env_bool("UAV_ON_JPEG_OPTIMIZE", True),
            astar_plan_cache=env_bool("UAV_ON_ASTAR_PLAN_CACHE", True),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["env_present"] = {
            "inline_vector_env": "UAV_ON_INLINE_VECTOR_ENV" in os.environ,
            "vector_env_start_method": "UAV_ON_VECTOR_ENV_START_METHOD" in os.environ,
            "kinematic_actions": "UAV_ON_KINEMATIC_ACTIONS" in os.environ,
            "scene_boot_seconds": "UAV_ON_SCENE_BOOT_SECONDS" in os.environ,
            "action_timeout_seconds": "UAV_ON_ACTION_TIMEOUT_SECONDS" in os.environ,
            "verbose_pose": "UAV_ON_VERBOSE_POSE" in os.environ,
            "set_pose_settle_seconds": "UAV_ON_SET_POSE_SETTLE_SECONDS" in os.environ,
            "rgb_only": "UAV_ON_RGB_ONLY" in os.environ,
            "rgb_compress": "UAV_ON_RGB_COMPRESS" in os.environ,
            "image_settle_seconds": "UAV_ON_IMAGE_SETTLE_SECONDS" in os.environ,
            "jpeg_optimize": "UAV_ON_JPEG_OPTIMIZE" in os.environ,
            "astar_plan_cache": "UAV_ON_ASTAR_PLAN_CACHE" in os.environ,
        }
        data["env_var_names"] = {
            "inline_vector_env": "UAV_ON_INLINE_VECTOR_ENV",
            "vector_env_start_method": "UAV_ON_VECTOR_ENV_START_METHOD",
            "kinematic_actions": "UAV_ON_KINEMATIC_ACTIONS",
            "scene_boot_seconds": "UAV_ON_SCENE_BOOT_SECONDS",
            "action_timeout_seconds": "UAV_ON_ACTION_TIMEOUT_SECONDS",
            "verbose_pose": "UAV_ON_VERBOSE_POSE",
            "set_pose_settle_seconds": "UAV_ON_SET_POSE_SETTLE_SECONDS",
            "rgb_only": "UAV_ON_RGB_ONLY",
            "rgb_compress": "UAV_ON_RGB_COMPRESS",
            "image_settle_seconds": "UAV_ON_IMAGE_SETTLE_SECONDS",
            "jpeg_optimize": "UAV_ON_JPEG_OPTIMIZE",
            "astar_plan_cache": "UAV_ON_ASTAR_PLAN_CACHE",
        }
        return data

    def to_env(self) -> dict[str, str]:
        return {
            "UAV_ON_INLINE_VECTOR_ENV": _bool_env(self.inline_vector_env),
            "UAV_ON_VECTOR_ENV_START_METHOD": self.vector_env_start_method,
            "UAV_ON_KINEMATIC_ACTIONS": _bool_env(self.kinematic_actions),
            "UAV_ON_SCENE_BOOT_SECONDS": str(self.scene_boot_seconds),
            "UAV_ON_ACTION_TIMEOUT_SECONDS": str(self.action_timeout_seconds),
            "UAV_ON_VERBOSE_POSE": _bool_env(self.verbose_pose),
            "UAV_ON_SET_POSE_SETTLE_SECONDS": str(self.set_pose_settle_seconds),
            "UAV_ON_RGB_ONLY": _bool_env(self.rgb_only),
            "UAV_ON_RGB_COMPRESS": _bool_env(self.rgb_compress),
            "UAV_ON_IMAGE_SETTLE_SECONDS": str(self.image_settle_seconds),
            "UAV_ON_JPEG_OPTIMIZE": _bool_env(self.jpeg_optimize),
            "UAV_ON_ASTAR_PLAN_CACHE": _bool_env(self.astar_plan_cache),
        }

    def apply_to_environ(self) -> None:
        os.environ.update(self.to_env())


def _bool_env(value: bool) -> str:
    return "1" if value else "0"


def runtime_config_from_env(*, scene_boot_seconds_default: float = 0.2) -> UavOnRuntimeConfig:
    return UavOnRuntimeConfig.from_env(scene_boot_seconds_default=scene_boot_seconds_default)


def write_runtime_config(path: str | Path, *, scene_boot_seconds_default: float = 0.2, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "runtime_config": runtime_config_from_env(
            scene_boot_seconds_default=scene_boot_seconds_default
        ).to_dict()
    }
    if extra:
        payload.update(extra)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print or write the resolved UAV-ON runtime config.")
    parser.add_argument("--output")
    parser.add_argument("--scene_boot_seconds_default", type=float, default=0.2)
    args = parser.parse_args()
    payload = {
        "runtime_config": runtime_config_from_env(
            scene_boot_seconds_default=args.scene_boot_seconds_default
        ).to_dict()
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
