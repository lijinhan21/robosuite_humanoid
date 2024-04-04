from robosuite.environments.base import make

# Manipulation environments
from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.manipulation.stack import Stack
from robosuite.environments.manipulation.nut_assembly import NutAssembly
from robosuite.environments.manipulation.pick_place import PickPlace
from robosuite.environments.manipulation.door import Door
from robosuite.environments.manipulation.wipe import Wipe
from robosuite.environments.manipulation.tool_hang import ToolHang
from robosuite.environments.manipulation.two_arm_lift import TwoArmLift
from robosuite.environments.manipulation.two_arm_peg_in_hole import TwoArmPegInHole
from robosuite.environments.manipulation.two_arm_handover import TwoArmHandover
from robosuite.environments.manipulation.two_arm_transport import TwoArmTransport
from robosuite.environments.manipulation.humanoid_transport import HumanoidTransport
from robosuite.environments.manipulation.humanoid_lift import HumanoidLift
from robosuite.environments.manipulation.humanoid_hci import HumanoidHCI
from robosuite.environments.manipulation.humanoid_toy_assembly import HumanoidToyAssembly
from robosuite.environments.manipulation.humanoid_return_book import HumanoidReturnBook
from robosuite.environments.manipulation.humanoid_pour import HumanoidPour
from robosuite.environments.manipulation.humanoid_handover import HumanoidHandover

from robosuite.environments import ALL_ENVIRONMENTS
from robosuite.controllers import ALL_CONTROLLERS, load_controller_config
from robosuite.robots import ALL_ROBOTS
from robosuite.models.grippers import ALL_GRIPPERS

__version__ = "1.4.1"
__logo__ = """
      ;     /        ,--.
     ["]   ["]  ,<  |__**|
    /[_]\  [~]\/    |//  |
     ] [   OOO      /o|__|
"""
