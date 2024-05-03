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
        init_qpos = np.array([0.0] * 20)

        init_qpos[6] = -2
        init_qpos[9] = -0.8

        init_qpos = np.array(
            [
                -0.0035970834452681436,
                0.011031227286351492,
                -0.01311470003464996,
                0.0,
                0.0,
                0.0,
                0.8511509067155127,
                1.310805039853726,
                -0.7118440391862395,
                -0.536551596928798,
                0.02341464067352966,
                -0.23317144423063796,
                -0.0803808564555934,
                0.18086797377837605,
                -1.5034221574091646,
                -0.15101789788918812,
                0.00014316406250000944,
                -0.07930486850248092,
                -0.1222325540688668,
                -0.2801763429367678,
            ]
        )

        return init_qpos

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.29, 0, 0),
            "table": lambda table_length: (-0.26 - table_length / 2, 0, 0),  # meaning: 0.26(+0.4m) behind table
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
