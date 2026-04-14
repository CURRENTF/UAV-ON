from enum import Enum
from typing import Dict, Union
import attr
import argparse
from common.param import args


class AirsimActions(Enum):
    STOP = "stop"
    FORWARD = "forward"
    LEFT = "left"
    RIGHT = "right"
    ROTATE_LEFT = "rotl"
    ROTATE_RIGHT = "rotr"
    ASCEND = "ascend"
    DESCEND = "descend"


class Singleton(type):
    _instances: Dict["Singleton", "Singleton"] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[cls]

class _DefaultAirsimActionSettings(Dict):
    FORWARD_STEP_SIZE = args.xOy_step_size
    UP_DOWN_STEP_SIZE = args.z_step_size
    LEFT_RIGHT_STEP_SIZE = args.xOy_step_size
    TURN_ANGLE = args.rotateAngle
   
AirsimActionSettings: _DefaultAirsimActionSettings = _DefaultAirsimActionSettings()
