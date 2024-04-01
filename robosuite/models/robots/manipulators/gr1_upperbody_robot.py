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
            np.array: default initial qpos for the right, left arms
        """
        init_qpos = np.array([0.0] * 14)
        init_qpos[0] = -0.5
        init_qpos[2] = 1.57
        init_qpos[7] = 0.5
        init_qpos[9] = -1.57

        init_qpos = np.array(
            [
                0.5753504639936845,
                -1.2880741550806842,
                -0.49831703339314676,
                -0.44860417328978697,
                -0.28278333799344413,
                -0.25934162343941286,
                -0.0699300618377962,
                0.9012313467817487,
                1.2331900425827245,
                0.7956200444029548,
                -0.6185976731583682,
                -0.03100568565622645,
                0.112091621446073,
                0.04413014930725423,
            ]
        )
        return init_qpos

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.29, 0, 0),
            "table": lambda table_length: (-0.26 - table_length / 2 - 0.4, 0, 0),  # meaning: 0.26(+0.4m) behind table
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
