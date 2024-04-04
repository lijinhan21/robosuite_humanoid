import numpy as np

from robosuite.models.robots.manipulators.humanoid_upperbody_model import HumanoidUpperBodyModel
from robosuite.utils.mjcf_utils import xml_path_completion


class GR1UpperBody(HumanoidUpperBodyModel):
    """
    GR1 is a humanoid designed by Fourier Intelligence.

    Here we seperates GR1's upper and lower bodies, and use its lower body as mount.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/gr1/gr1_upperbody.xml"), idn=idn)

    @property
    def default_mount(self):
        return "GR1LowerBodyMount"

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
            np.array: default initial qpos for the right, left arms and torso
        """

        init_qpos = np.array(
            [
                0.0014040807723666444,
                0.0329435867253274,
                -0.024655615771712422,
                0.0,
                0.0,
                0.0,
                0.7161998741312713,
                1.210619552204434,
                -0.6153317664476635,
                -0.6177820316479151,
                -0.23335661355637657,
                -0.25603128456888447,
                0.011129393205483024,
                -0.6450654829774324,
                -1.2630308009844142,
                0.5779023301794867,
                0.509870018771001,
                0.11822843901171841,
                -0.15207494115114661,
                0.10339960954616242,
            ]
        )

        init_qpos = np.array([0.0] * 20)
        return init_qpos

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.29, 0, 0),
            "table": lambda table_length: (-0.26 - table_length / 2 + 0.1, 0, 0),  # meaning: 0.26(+0.4m) behind table
        }

    @property
    def top_offset(self):
        return np.array((0.0, 0, 1.0))

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
