"""
Dexterous hands for GR1 robot.
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class InspireLeftHand(GripperModel):
    """
    Dexterous left hand of GR1 robot

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/inspire_left_hand.xml"), idn=idn)

    def format_action(self, action):
        return action  # * 0.25

    @property
    def init_qpos(self):
        # return np.array([
        # 0.6908888300000045, 0.33040944897938984, 0.33040944897938984, 0.373352,
        # 1.8373, 1.6262810813257051,
        # 1.3970, 1.2363174066611564,
        # 1.3649, 1.263856745671346,
        # 1.13936, 0.9907592979638666
        # ])
        return np.array([0.0] * 12)

    @property
    def speed(self):
        return 0.1

    @property
    def dof(self):
        return 6


class InspireRightHand(GripperModel):
    """
    Dexterous right hand of GR1 robot

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/inspire_right_hand.xml"), idn=idn)

    def format_action(self, action):
        return action  # * 0.25

    @property
    def init_qpos(self):
        # return np.array([0.9239287812762349, 0.5213948124043338, 0.5213948124043338, 0.58873,
        # 0.73337, 0.6491059936647702,
        # 0.60794, 0.538189296341709,
        # 0.57132, 0.5291087508786502,
        # 0.50945, 0.44390332861756177,])

        return np.array([0.0] * 12)

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 6
