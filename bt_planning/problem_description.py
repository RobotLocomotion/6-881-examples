import numpy as np

from iiwa_utils import (open_wsg50_gripper, get_door_positions, DOOR_CLOSED,
                        DOOR_OPEN)
from motion import get_difference_fn
from utils import (get_relative_transform, get_model_bodies, get_bodies,
                   set_joint_position, set_world_pose, get_world_pose,
                   get_configuration, get_model_name, get_joint_positions,
                   get_parent_joints, get_movable_joints)

class Pose(object):
    def __init__(self, mbp, parent, child, transform, surface=None):
        self.mbp = mbp
        self.parent = parent  # body_frame
        self.child = child  # model_index
        self.transform = transform
        self.surface = surface

    @property
    def bodies(self):
        return get_model_bodies(self.mbp, self.child)

    def assign(self, context):
        parent_pose = get_relative_transform(self.mbp, context, self.parent)
        child_pose = parent_pose.multiply(self.transform)
        set_world_pose(self.mbp, context, self.child, child_pose)

    def __repr__(self):
        return '{}({}->{})'.format(self.__class__.__name__, get_model_name(
            self.mbp, self.child), self.parent.name())


class Conf(object):
    def __init__(self, joints, positions):
        assert len(joints) == len(positions)
        self.joints = joints
        self.positions = tuple(positions)

    @property
    def bodies(self):
        return {joint.child_body() for joint in self.joints}

    def assign(self, context):
        for joint, position in zip(self.joints, self.positions):
            set_joint_position(joint, context, position)

    def __repr__(self):
        return '{}({},{})'.format(
            self.__class__.__name__, len(self.joints), id(self) % 1000)


class Trajectory(object):
    def __init__(self, path, attachments=[], force_control=False,
                 speed=np.pi/32):
        self.path = tuple(path)
        self.attachments = attachments
        self.force_control = force_control
        self.speed = speed

    @property
    def joints(self):
        return self.path[0].joints

    @property
    def bodies(self):
        joint_bodies = {joint.child_body() for joint in self.joints}
        for attachment in self.attachments:
            joint_bodies.update(attachment.bodies)
        return joint_bodies

    def reverse(self):
        return self.__class__(
            self.path[::-1], self.attachments, self.force_control)

    def iterate(self, context):
        for conf in self.path[1:]:
            conf.assign(context)
            for attach in self.attachments:
                attach.assign(context)
            yield

    def distance(self):
        distance_fn = get_distance_fn(self.joints)
        return sum(distance_fn(q1.positions, q2.positions)
                   for q1, q2 in zip(self.path, self.path[1:]))

    def retime(self):
        path = [q.positions[:len(self.joints)] for q in self.path]
        q_knots = np.vstack(path).T
        difference_fn = get_difference_fn(self.joints)
        distances = [0.] + [np.max(np.abs(difference_fn(q1, q2)))
                            for q1, q2 in zip(path, path[1:])]
        t_knots = np.cumsum(distances) / self.speed
        return q_knots, t_knots

    def __repr__(self):
        return '{}({},{})'.format(
            self.__class__.__name__, len(self.joints), len(self.path))


class Task(object):
    def __init__(self, diagram, mbp, scene_graph, robot, gripper,
                 movable=[], surfaces=[], doors=[],
                 initial_positions={}, initial_poses={},
                 goal_poses={}, goal_holding=[], goal_on=[],
                 reset_robot=True, reset_doors=True):
        # Drake systems
        self.diagram = diagram
        self.mbp = mbp
        self.scene_graph = scene_graph
        self.diagram_context = diagram.CreateDefaultContext()
        self.plant_context = diagram.GetMutableSubsystemContext(
            mbp, self.diagram_context)

        # Semantic information about models
        self.robot = robot
        self.gripper = gripper
        self.movable = tuple(movable)
        self.surfaces = tuple(surfaces)
        self.doors = tuple(doors)

        # Initial values
        self.initial_positions = initial_positions
        self.initial_poses = initial_poses

        # Goal conditions
        self.goal_poses = goal_poses
        self.goal_holding = goal_holding
        self.goal_on = goal_on
        self.reset_robot = reset_robot
        self.reset_doors = reset_doors

        # initialize at the start of the problem, will be checked
        # and mutated throughout execution
        # Preposition dictionaries
        robot_joints = get_movable_joints(self.mbp, robot)
        robot_conf = Conf(robot_joints, get_configuration(
            self.mbp, self.plant_context, robot))
        self.can_move = {robot: True}
        self.hand_empty = {robot: True}
        self.conf = {robot: {robot_conf: True}}
        self.at_conf = {robot: robot_conf}
        self.init_conf = {robot: robot_conf}

        self.graspable = {}
        self.at_pose ={}
        self.pose = {}
        self.holding = {robot: {}}
        self.stackable = {}
        self.on = {}
        self.at_grasp = {robot: {}}
        for obj in self.movable:
            obj_name = get_model_name(self.mbp, obj)
            obj_pose = Pose(self.mbp, self.mbp.world_frame(), obj,
                            get_world_pose(self.mbp, self.plant_context, obj))
            self.holding[robot][obj_name] = False
            self.graspable[obj_name] = True
            self.at_pose[obj_name] = obj_pose
            self.pose[obj_name] = {obj_pose: True}

            for surface in self.surfaces:
                self.stackable[(obj_name, surface)] = True

        for door in self.doors:
            door_body = self.mbp.get_body(door)
            door_name = door_body.name()
            door_joints = get_parent_joints(self.mbp, door_body)
            door_conf = Conf(door_joints, get_joint_positions(
                door_joints, self.plant_context))
            self.conf[door_name] = {door_conf: True}
            self.at_conf[door_name] = door_conf
            self.init_conf[door_name] = door_conf
            for positions in [get_door_positions(door_body, DOOR_OPEN)]:
                conf = Conf(door_joints, positions)
                self.conf[door_name][conf] = True

    @property
    def plant(self):
        return self.mbp

    def movable_bodies(self):
        movable = {self.mbp.get_body(index) for index in self.doors}
        for model in [self.robot, self.gripper] + list(self.movable):
            movable.update(get_model_bodies(self.mbp, model))
        return movable

    def fixed_bodies(self):
        fixed = set(get_bodies(self.mbp)) - self.movable_bodies()
        return fixed

    def set_initial(self):
        for joint, position in self.initial_positions.items():
            set_joint_position(joint, self.plant_context, position)
        for model, pose in self.initial_poses.items():
            set_world_pose(self.plant, self.plant_context, model, pose)
        open_wsg50_gripper(self.plant, self.plant_context, self.gripper)

    def publish(self):
        self.diagram.Publish(self.diagram_context)

    def complete(self):
        '''
        Checks if all of the goal conditions have been met.

        @return True if all conditions have been met, otherwise False
        '''
        for obj in self.goal_holding:
            if not self.holding[self.robot][get_model_name(self.mbp, obj)]:
                return False

        for obj, surface in self.goal_on:
            if not self.on[get_model_name(self.mbp, obj)][surface]:
                return False

        if self.reset_robot:
            if not self.at_conf[self.robot] == self.init_conf[self.robot]:
                return False

        if self.reset_doors:
            for door in self.doors:
                door_body = self.mbp.get_body(door)
                door_name = door_body.name()
                if not self.at_conf[door_name] == self.init_conf[door_name]:
                    return False

        return True

    def __repr__(self):
        return '{}(robot={}, gripper={}, movable={}, surfaces={})'.format(
            self.__class__.__name__,
            get_model_name(self.mbp, self.robot),
            get_model_name(self.mbp, self.gripper),
            [get_model_name(self.mbp, model) for model in self.movable],
            self.surfaces)