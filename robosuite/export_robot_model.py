from xml.etree import ElementTree as ET

import robosuite
from robosuite.robots import Bimanual

robot = Bimanual(robot_type="GR1UpperBody")
# robot = FixedBaseRobot(robot_type="Z1")
robot.load_model()
robot_model = robot.robot_model
robot_root = robot_model.tree.getroot()

# indent on root
# ET.indent(robot_root, space="\t", level=0)

with open("exported_robot_model.xml", "w") as f:
    f.write(ET.tostring(robot_root).decode())
