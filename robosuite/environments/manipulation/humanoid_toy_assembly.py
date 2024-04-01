from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.humanoid_env import HumanoidEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import CrateObject, IkeaBoxObject, ToyBoatObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler


class HumanoidToyAssembly(HumanoidEnv):  # TODO: check which env to inherit
    """
    # TODO: write comments
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.7, 0.7, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.95))  # a higher table to accommodate humanoid robot 1.15m

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):  # TODO: implement
        """
        Reward function for the task.

        TODO: implement

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0
        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose(s) accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # set camera pose for this task
        mujoco_arena.set_camera(camera_name="taskview", pos=[1.0, 0.0, 1.48], quat=[0.56, 0.43, 0.43, 0.56])

        # initialize objects of interest
        self.toy_boat = ToyBoatObject(name="toy-boat")
        self.crate = CrateObject(name="crate")  # IkeaBoxObject(name="ikea_box")

        # Create placement initializer
        self._get_placement_initializer()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.toy_boat, self.crate],
        )

    def _get_placement_initializer(self):  # TODO: change for this task
        """
        Helper function for defining placement initializer and object sampling bounds
        """
        # Create placement initializer
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        obj_names = ["toy-boat", "crate"]
        objs = [self.toy_boat, self.crate]
        x_ranges = [[-0.25, -0.1], [-0.25, -0.15]]  # TODO: change to reasonable range
        y_ranges = [[0.05, 0.15], [-0.15, -0.05]]  # TODO: change to reasonable range
        rotation_ranges = [[-0.1, 0.1], [-1.57, -1.57]]  # TODO: change to reasonable range
        # TODO: change site in xml files after the sizes are determined!
        for obj_name, obj, x_range, y_range, rotation_range in zip(
            obj_names, objs, x_ranges, y_ranges, rotation_ranges
        ):
            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name=f"{obj_name}ObjectSampler",
                    mujoco_objects=obj,
                    x_range=x_range,
                    y_range=y_range,
                    rotation=rotation_range,
                    rotation_axis="z",
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                    reference_pos=self.table_offset,  # this is the center of the table
                    z_offset=0.00,
                )
            )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.toy_boat_body_id = self.sim.model.body_name2id(self.toy_boat.root_body)
        self.crate_body_id = self.sim.model.body_name2id(self.crate.root_body)
        self.table_top_id = self.sim.model.site_name2id("table_top")

        # TODO: add other useful site id

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:

            modality = "object"

            # position and rotation of objects
            @sensor(modality=modality)
            def toy_boat_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.toy_boat_body_id])

            @sensor(modality=modality)
            def toy_boat_quat(obs_cache):
                return T.convert_quat(self.sim.data.body_xquat[self.toy_boat_body_id], to="xyzw")

            @sensor(modality=modality)
            def crate_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.crate_body_id])

            @sensor(modality=modality)
            def crate_quat(obs_cache):
                return T.convert_quat(self.sim.data.body_xquat[self.crate_body_id], to="xyzw")

            # TODO: may need to add other observables if needed

            sensors = [toy_boat_pos, toy_boat_quat, crate_pos, crate_quat]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def _check_success(self):
        """
        TODO: implement
        """
        return False
