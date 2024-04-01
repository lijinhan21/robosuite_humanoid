import numpy as np

from robosuite.models.robots.manipulators.humanoid_model import HumanoidModel
from robosuite.utils.mjcf_utils import xml_path_completion


class GR1(HumanoidModel):
    """
    GR1 is a humanoid designed by Fourier Intelligence.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/gr1/gr1.xml"), idn=idn)

    @property
    def default_gripper(self):
        """
        Since this is bimanual robot, returns dict with `'right'`, `'left'` keywords corresponding to their respective
        values

        Returns:
            dict: Dictionary containing arm-specific gripper names
        """
        return {"right": "InspireRightHand", "left": "InspireLeftHand"}

    @property
    def default_controller_config(self):
        """
        Since this is bimanual robot, returns dict with `'right'`, `'left'` keywords corresponding to their respective
        values

        Returns:
            dict: Dictionary containing arm-specific default controller config names
        """
        return {"right": "default_gr1", "left": "default_gr1"}

    @property
    def init_qpos(self):
        """
        Since this is bimanual robot, returns [right, left] array corresponding to respective values

        Returns:
            np.array: default initial qpos for the right, left arms
        """
        init_qpos = np.array([0.0] * 32)
        return init_qpos

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0.92),
            "empty": (-0.29, 0, 0.92),
            "table": lambda table_length: (-0.26 - table_length / 2, 0, 0.92),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "bimanual"

    @property
    def _eef_name(self):
        """
        Since this is bimanual robot, returns dict with `'right'`, `'left'` keywords corresponding to their respective
        values

        Returns:
            dict: Dictionary containing arm-specific eef names
        """
        return {"right": "right_eef", "left": "left_eef"}