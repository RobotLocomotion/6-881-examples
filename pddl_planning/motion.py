import numpy as np

from pddl_planning.utils import get_random_positions, get_joint_limits, set_joint_positions, exists_colliding_pair, \
    create_transform, solve_inverse_kinematics
from examples.pybullet.utils.motion.motion_planners.rrt_connect import birrt

DEFAULT_WEIGHT = 1.0
DEFAULT_RESOLUTION = 0.01*np.pi

##################################################


def get_sample_fn(joints, collision_fn=lambda q: False):
    def fn():
        while True:
            q = get_random_positions(joints)
            if not collision_fn(q):
                return q
    return fn


def get_difference_fn(joints):

    def fn(q2, q1):
        assert len(joints) == len(q2)
        assert len(joints) == len(q1)
        return np.array(q2) - np.array(q1)
    return fn


def get_distance_fn(joints, weights=None):
    if weights is None:
        weights = DEFAULT_WEIGHT*np.ones(len(joints))
    difference_fn = get_difference_fn(joints)

    def fn(q1, q2):
        diff = np.array(difference_fn(q2, q1))
        return np.sqrt(np.dot(weights, diff * diff))
    return fn


def get_extend_fn(joints, resolutions=None):
    if resolutions is None:
        resolutions = DEFAULT_RESOLUTION*np.ones(len(joints))
    assert len(joints) == len(resolutions)
    difference_fn = get_difference_fn(joints)

    def fn(q1, q2):
        num_steps = int(np.max(np.abs(np.divide(difference_fn(q2, q1), resolutions)))) + 1
        q = q1
        for i in range(num_steps):
            q = (1. / (num_steps - i)) * np.array(difference_fn(q2, q)) + q
            yield q
    return fn


def within_limits(joint, position):
    lower, upper = get_joint_limits(joint)
    return lower <= position <= upper


def get_collision_fn(diagram, diagram_context, plant, scene_graph,
                     joints, collision_pairs=set(), attachments=[]):
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

    def fn(q):
        if any(not within_limits(joint, position) for joint, position in zip(joints, q)):
            return True
        if not collision_pairs:
            return False
        set_joint_positions(joints, plant_context, q)
        for attachment in attachments:
            attachment.assign(plant_context)
        colliding = exists_colliding_pair(diagram, diagram_context, plant, scene_graph, collision_pairs)
        return colliding
    return fn

##################################################


def plan_workspace_motion(plant, joints, frame, frame_path, initial_guess=None,
                          collision_fn=lambda q: False, **kwargs):
    solution = initial_guess
    waypoints = []
    for frame_pose in frame_path:
        solution = solve_inverse_kinematics(plant, frame, frame_pose, initial_guess=solution, **kwargs)
        if solution is None:
            return None
        positions = [solution[j.position_start()] for j in joints]
        if collision_fn(positions):
            return None
        waypoints.append(positions)
    return waypoints


def plan_waypoints_joint_motion(joints, waypoints, resolutions=None,
                                collision_fn=lambda q: False):
    if not waypoints:
        return []
    for waypoint in waypoints:
        assert len(joints) == len(waypoint)
        if collision_fn(waypoint):
            return None

    extend_fn = get_extend_fn(joints, resolutions=resolutions)
    path = [waypoints[0]]
    for waypoint in waypoints[1:]:
        for q in extend_fn(path[-1], waypoint):
            if collision_fn(q):
                return None
            path.append(q)
    return path


def plan_joint_motion(joints, start_positions, end_positions,
                      weights=None, resolutions=None,
                      sample_fn=None, distance_fn=None,
                      collision_fn=lambda q: False,
                      **kwargs):
    assert len(joints) == len(start_positions)
    assert len(joints) == len(end_positions)
    if collision_fn(start_positions):
        print('Warning! Start positions in collision')
        return None
    if collision_fn(end_positions):
        print('Warning! End positions in collision')
        return None
    if distance_fn is None:
        distance_fn = get_distance_fn(joints, weights=weights)
    if sample_fn is None:
        sample_fn = get_sample_fn(joints, collision_fn=collision_fn)
    extend_fn = get_extend_fn(joints, resolutions=resolutions)
    return birrt(start_positions, end_positions, distance=distance_fn, sample=sample_fn,
                 extend=extend_fn, collision=collision_fn, **kwargs)


##################################################


def interpolate_translation(transform, translation, step_size=0.01):
    distance = np.linalg.norm(translation)
    if distance == 0:
        yield transform
        return
    direction = np.array(translation) / distance
    for t in list(np.arange(0, distance, step_size)) + [distance]:
        yield create_transform(translation=t * direction).multiply(transform)
