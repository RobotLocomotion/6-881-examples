from __future__ import print_function

from itertools import product, cycle

import numpy as np

from pddl_planning.iiwa_utils import open_wsg50_gripper, get_box_grasps, get_cylinder_grasps, \
    get_close_wsg50_positions, get_open_wsg50_positions, get_door_positions, DOOR_CLOSED, DOOR_OPEN
from pddl_planning.motion import plan_joint_motion, plan_waypoints_joint_motion, get_difference_fn, \
    get_extend_fn, interpolate_translation, plan_workspace_motion, get_collision_fn, get_distance_fn
from pddl_planning.utils import get_relative_transform, set_world_pose, set_joint_position, get_body_pose, \
    get_base_body, sample_aabb_placement, get_movable_joints, get_model_name, set_joint_positions, get_box_from_geom, \
    exists_colliding_pair, get_model_bodies, bodies_from_models, get_world_pose, get_state

RADIANS_PER_SECOND = np.pi / 32

class Pose(object):
    def __init__(self, mbp, parent, child, transform, surface=None):
        self.mbp = mbp
        self.parent = parent # body_frame
        self.child = child # model_index
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
        return '{}({}->{})'.format(self.__class__.__name__, get_model_name(self.mbp, self.child), self.parent.name())


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
        return '{}({},{})'.format(self.__class__.__name__, len(self.joints), id(self) % 1000)


class Trajectory(object):
    def __init__(self, path, attachments=[], force_control=False):
        self.path = tuple(path)
        self.attachments = attachments
        self.force_control = force_control

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
        return self.__class__(self.path[::-1], self.attachments, self.force_control)

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
        t_knots = np.cumsum(distances) / RADIANS_PER_SECOND
        return q_knots, t_knots

    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, len(self.joints), len(self.path))


def get_open_trajectory(plant, gripper):
    gripper_joints = get_movable_joints(plant, gripper)
    gripper_extend_fn = get_extend_fn(gripper_joints)
    gripper_closed_conf = get_close_wsg50_positions(plant, gripper)
    gripper_path = list(gripper_extend_fn(gripper_closed_conf, get_open_wsg50_positions(plant, gripper)))
    gripper_path.insert(0, gripper_closed_conf)
    return Trajectory(Conf(gripper_joints, q) for q in gripper_path)

##################################################


def get_pose_gen(task, context, collisions=True, shrink=0.025):
    mbp = task.mbp
    world = mbp.world_frame()
    box_from_geom = get_box_from_geom(task.scene_graph)
    fixed = task.fixed_bodies() if collisions else []
    world_from_robot = get_world_pose(task.mbp, context, task.robot)
    max_xy_distance = 0.85

    def gen(obj_name, surface):
        obj = mbp.GetModelInstanceByName(obj_name)
        obj_aabb, obj_from_box, _ = box_from_geom[int(obj), get_base_body(mbp, obj).name(), 0]
        collision_pairs = set(product(get_model_bodies(mbp, obj), fixed))

        surface_body = mbp.GetBodyByName(surface.body_name, surface.model_index)
        surface_pose = get_body_pose(context, surface_body)
        surface_aabb, surface_from_box, _ = box_from_geom[int(surface.model_index), surface.body_name, surface.visual_index]

        for surface_box_from_obj_box in sample_aabb_placement(obj_aabb, surface_aabb, shrink=shrink):
            world_from_obj = surface_pose.multiply(surface_from_box).multiply(
                surface_box_from_obj_box).multiply(obj_from_box.inverse())
            robot_from_obj = world_from_robot.inverse().multiply(world_from_obj)
            if max_xy_distance < np.linalg.norm(robot_from_obj.translation()[:2], ord=np.inf):
                continue
            pose = Pose(mbp, world, obj, world_from_obj, surface=surface)
            pose.assign(context)
            if not exists_colliding_pair(task.diagram, task.diagram_context,
                                         task.mbp, task.scene_graph, collision_pairs):
                yield (pose,)
    return gen


def get_grasp_gen_fn(task):
    plant = task.mbp
    gripper_frame = get_base_body(plant, task.gripper).body_frame()
    box_from_geom = get_box_from_geom(task.scene_graph)
    pitch = 4*np.pi / 9
    assert abs(pitch) <= np.pi / 2

    def gen(obj_name):
        obj = plant.GetModelInstanceByName(obj_name)
        obj_aabb, obj_from_box, obj_shape = box_from_geom[int(obj), get_base_body(plant, obj).name(), 0]
        if obj_shape == 'cylinder':
            grasp_gen = get_cylinder_grasps(obj_aabb, pitch_range=(pitch, pitch))
        elif obj_shape == 'box':
            grasp_gen = get_box_grasps(obj_aabb, pitch_range=(pitch, pitch))
        else:
            raise NotImplementedError(obj_shape)
        for gripper_from_box in grasp_gen:
            gripper_from_obj = gripper_from_box.multiply(obj_from_box.inverse())
            grasp = Pose(plant, gripper_frame, obj, gripper_from_obj)
            yield (grasp,)
    return gen

##################################################


def plan_frame_motion(plant, joints, frame, frame_path,
                      initial_guess=None, resolutions=None, collision_fn=lambda q: False):
    waypoints = plan_workspace_motion(plant, joints, frame, frame_path,
                                      initial_guess=initial_guess, collision_fn=collision_fn)
    if waypoints is None:
        return None
    return plan_waypoints_joint_motion(joints, waypoints, resolutions=resolutions, collision_fn=collision_fn)


def get_ik_gen_fn(task, context, collisions=True, max_failures=10, step_size=0.05):
    approach_vector = 0.15 * np.array([0, -1, 0])
    world_vector = 0.05 * np.array([0, 0, +1])
    gripper_frame = get_base_body(task.mbp, task.gripper).body_frame()
    door_bodies = {task.mbp.tree().get_body(door_index) for door_index in task.doors}
    fixed = (task.fixed_bodies() | door_bodies) if collisions else []
    initial_guess = get_state(task.mbp, context)[:task.mbp.num_positions()]
    # Above shelves prevent some placements

    def fn(robot_name, obj_name, obj_pose, obj_grasp):
        robot = task.mbp.GetModelInstanceByName(robot_name)
        joints = get_movable_joints(task.mbp, robot)
        collision_pairs = set(product(bodies_from_models(task.mbp, [robot, task.gripper]), fixed))
        collision_fn = get_collision_fn(task.diagram, task.diagram_context, task.mbp, task.scene_graph,
                                        joints, collision_pairs=collision_pairs) # TODO: while holding

        gripper_pose = obj_pose.transform.multiply(obj_grasp.transform.inverse())
        end_position = gripper_pose.multiply(approach_vector) + world_vector
        translation = end_position - gripper_pose.translation()
        gripper_path = list(interpolate_translation(gripper_pose, translation, step_size=step_size))

        attempts = 0
        last_success = 0
        while (attempts - last_success) < max_failures:
            attempts += 1
            obj_pose.assign(context)

            if door_bodies:
                for door_body in door_bodies:
                    positions = get_door_positions(door_body, DOOR_OPEN)
                    for door_joint in get_movable_joints(task.plant, door_body.model_instance()):
                        if door_joint.child_body() == door_body:
                            set_joint_positions([door_joint], context, positions)

            path = plan_frame_motion(task.plant, joints, gripper_frame, gripper_path,
                                     initial_guess=initial_guess, collision_fn=collision_fn)
            if path is None:
                continue
            traj = Trajectory([Conf(joints, q) for q in path], attachments=[obj_grasp])
            conf = traj.path[-1]
            yield (conf, traj)
            last_success = attempts
    return fn

##################################################


def get_reachable_grasp_gen_fn(task, context, collisions=True, max_failures=50, **kwargs):
    grasp_gen_fn = get_grasp_gen_fn(task)
    ik_gen_fn = get_ik_gen_fn(task, context, collisions=collisions, max_failures=1, **kwargs)
    def gen(r, o, p):
        failures = 0
        for (g,) in cycle(grasp_gen_fn(o)):
            if max_failures < failures:
                break
            try:
                (q, t) = next(ik_gen_fn(r, o, p, g))
                yield (g, q, t)
                failures = 0
            except StopIteration:
                failures += 1
    return gen


def get_reachable_pose_gen_fn(task, context, collisions=True, max_failures=100, **kwargs):
    pose_gen_fn = get_pose_gen(task, context, collisions=collisions)
    ik_gen_fn = get_ik_gen_fn(task, context, collisions=collisions, max_failures=1, **kwargs)
    def gen(r, o, g, s):
        failures = 0
        for (p,) in cycle(pose_gen_fn(o, s)):
            if max_failures < failures:
                break
            try:
                (q, t) = next(ik_gen_fn(r, o, p, g))
                yield (p, q, t)
                failures = 0
            except StopIteration:
                failures += 1
    return gen

##################################################

def get_door_grasp(door_body, box_from_geom):
    pitch = np.pi/3 # np.pi/2
    grasp_length = 0.02
    target_shape, target_ori = 'cylinder', 1  # Second grasp is np.pi/2, corresponding to +y
    for i in range(2):
        handle_aabb, handle_from_box, handle_shape = box_from_geom[int(door_body.model_instance()), door_body.name(), i]
        if handle_shape == target_shape:
            break
    else:
        raise RuntimeError(target_shape)
    [gripper_from_box] = list(get_box_grasps(handle_aabb, orientations=[target_ori],
                                             pitch_range=(pitch, pitch), grasp_length=grasp_length))
    return gripper_from_box.multiply(handle_from_box.inverse())


def get_body_path(body, context, joints, joint_path):
    body_path = []
    for conf in joint_path:
        set_joint_positions(joints, context, conf)
        body_path.append(get_body_pose(context, body))
    return body_path


def get_pull_fn(task, context, collisions=True, max_attempts=25, step_size=np.pi / 16, approach_distance=0.05):
    box_from_geom = get_box_from_geom(task.scene_graph)
    gripper_frame = get_base_body(task.mbp, task.gripper).body_frame()
    fixed = task.fixed_bodies() if collisions else []

    def fn(robot_name, door_name, door_conf1, door_conf2):
        """
        :param robot_name: The name of the robot (should be iiwa)
        :param door_name: The name of the door (should be left_door or right_door)
        :param door_conf1: The initial door configuration
        :param door_conf2: The final door configuration
        :return: A triplet composed of the initial robot configuration, final robot configuration,
                 and combined robot & door position trajectory to execute the pull
        """
        robot = task.mbp.GetModelInstanceByName(robot_name)
        robot_joints = get_movable_joints(task.mbp, robot)
        collision_pairs = set(product(bodies_from_models(task.mbp, [robot, task.gripper]), fixed))
        collision_fn = get_collision_fn(task.diagram, task.diagram_context, task.mbp, task.scene_graph,
                                        robot_joints, collision_pairs=collision_pairs)

        door_body = task.mbp.GetBodyByName(door_name)
        door_joints = door_conf1.joints
        combined_joints = robot_joints + door_joints
        # The transformation from the door frame to the gripper frame that corresponds to grasping the door handle
        gripper_from_door = get_door_grasp(door_body, box_from_geom)

        extend_fn = get_extend_fn(door_joints, resolutions=step_size*np.ones(len(door_joints)))
        door_joint_path = [door_conf1.positions] + list(extend_fn(door_conf1.positions, door_conf2.positions))
        door_body_path = get_body_path(door_body, context, door_joints, door_joint_path)
        gripper_body_path = [door_pose.multiply(gripper_from_door.inverse()) for door_pose in door_body_path]

        for _ in range(max_attempts):
            robot_joint_waypoints = plan_workspace_motion(task.mbp, robot_joints, gripper_frame,
                                                          gripper_body_path, collision_fn=collision_fn)
            if robot_joint_waypoints is None:
                continue
            combined_waypoints = [list(rq) + list(dq) for rq, dq in zip(robot_joint_waypoints, door_joint_path)]
            combined_joint_path = plan_waypoints_joint_motion(combined_joints, combined_waypoints,
                                                              collision_fn=lambda q: False)
            if combined_joint_path is None:
                continue

            # combined_joint_path is a joint position path for the concatenated robot and door joints.
            # It should be a list of 8 DOF configurations (7 robot DOFs + 1 door DOF).
            # Additionally, combined_joint_path[0][len(robot_joints):] should equal door_conf1.positions
            # and combined_joint_path[-1][len(robot_joints):] should equal door_conf2.positions.

            robot_conf1 = Conf(robot_joints, combined_joint_path[0][:len(robot_joints)])
            robot_conf2 = Conf(robot_joints, combined_joint_path[-1][:len(robot_joints)])
            traj = Trajectory(Conf(combined_joints, combined_conf) for combined_conf in combined_joint_path)
            yield (robot_conf1, robot_conf2, traj)
    return fn


def get_force_pull_fn(task, context, collisions=True):
    home_conf = np.array([0, -0.2136, 0, -2.094, 0, 0.463, 0])

    def fn(robot_name, door_name, door_conf1, door_conf2):
        robot = task.mbp.GetModelInstanceByName(robot_name)
        robot_joints = get_movable_joints(task.mbp, robot)

        door_body = task.mbp.GetBodyByName(door_name)
        door_joints = door_conf1.joints
        combined_joints = robot_joints + door_joints
        if not np.allclose(get_door_positions(door_body, DOOR_CLOSED), door_conf1.positions):
            return

        combined_waypoints = [list(home_conf) + list(door_conf.positions)
                              for door_conf in [door_conf1, door_conf2]]
        combined_joint_path = plan_waypoints_joint_motion(combined_joints, combined_waypoints,
                                                          collision_fn=lambda q: False)
        if combined_joint_path is None:
            return

        robot_conf1 = Conf(robot_joints, home_conf)
        robot_conf2 = robot_conf1
        traj = Trajectory([Conf(combined_joints, combined_conf)
                           for combined_conf in combined_joint_path], force_control=True)
        yield (robot_conf1, robot_conf2, traj)
    return fn

##################################################

def parse_fluents(fluents, context, obstacles):
    attachments = []
    for fact in fluents:
        predicate = fact[0]
        if predicate == 'AtConf'.lower():
            name, conf = fact[1:]
            conf.assign(context)
            obstacles.update(conf.bodies)
        elif predicate == 'AtPose'.lower():
            name, pose = fact[1:]
            pose.assign(context)
            obstacles.update(pose.bodies)
        elif predicate == 'AtGrasp'.lower():
            robot, name, grasp = fact[1:]
            attachments.append(grasp)
        else:
            raise ValueError(predicate)
    return attachments


def get_motion_fn(task, context, collisions=True, teleport=False):
    gripper = task.gripper

    def fn(robot_name, conf1, conf2, fluents=[]):
        robot = task.mbp.GetModelInstanceByName(robot_name)
        joints = get_movable_joints(task.mbp, robot)

        moving = bodies_from_models(task.mbp, [robot, gripper])
        obstacles = set(task.fixed_bodies())
        attachments = parse_fluents(fluents, context, obstacles)
        for grasp in attachments:
            moving.update(grasp.bodies)
        obstacles -= moving

        if teleport:
            traj = Trajectory([conf1, conf2], attachments=attachments)
            return (traj,)

        collision_pairs = set(product(moving, obstacles)) if collisions else set()
        collision_fn = get_collision_fn(task.diagram, task.diagram_context, task.mbp, task.scene_graph,
                                        joints, collision_pairs=collision_pairs, attachments=attachments)
        #sample_fn = get_sample_fn(joints, start_conf=conf1.positions,
        #                          end_conf=conf2.positions, collision_fn=collision_fn)
        sample_fn = None

        open_wsg50_gripper(task.mbp, context, gripper)
        path = plan_joint_motion(joints, conf1.positions, conf2.positions,
                                 sample_fn=sample_fn, collision_fn=collision_fn,
                                 restarts=10, iterations=75, smooth=50)
                                 #restarts=25, iterations=25, smooth=0) # Disabled smoothing to see the path

        if path is None:
            return None
        # Sample within the ellipsoid
        traj = Trajectory([Conf(joints, q) for q in path], attachments=attachments)
        return (traj,)
    return fn

##################################################

def get_collision_test(task, context, collisions=True):
    def test(traj, obj_name, pose):
        if not collisions:
            return False
        moving = bodies_from_models(task.mbp, [task.robot, task.gripper])
        moving.update(traj.bodies)
        obstacles = set(pose.bodies) - moving
        collision_pairs = set(product(moving, obstacles))
        if not collision_pairs:
            return False
        pose.assign(context)
        for _ in traj.iterate(context):
            if exists_colliding_pair(task.diagram, task.diagram_context,
                                     task.mbp, task.scene_graph, collision_pairs):
                return True
        return False
    return test
