"""Simple next-version CoppeliaSim controller.

V2 intentionally keeps rail motion, arm joint motion, and arm-only IK separate.
It uses sampled collision prevalidation for linear moves and arm-only OMPL paths.
"""

from .pick_place import PickPlaceControllerV2
from .robot import RobotControllerV2

__all__ = ["PickPlaceControllerV2", "RobotControllerV2"]
