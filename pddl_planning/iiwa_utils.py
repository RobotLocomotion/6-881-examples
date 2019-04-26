import random
import math
import numpy as np
from itertools import count

from pddl_planning.utils import create_transform, set_joint_positions, get_movable_joints


WSG50_LEFT_FINGER = 'left_finger_sliding_joint'
WSG50_RIGHT_FINGER = 'right_finger_sliding_joint'

def get_close_wsg50_positions(mbp, model_index):
    left_joint = mbp.GetJointByName(WSG50_LEFT_FINGER, model_index)
    right_joint = mbp.GetJointByName(WSG50_RIGHT_FINGER, model_index)
    return [left_joint.position_upper_limits()[0], right_joint.position_lower_limits()[0]]


def get_open_wsg50_positions(mbp, model_index):
    left_joint = mbp.GetJointByName(WSG50_LEFT_FINGER, model_index)
    right_joint = mbp.GetJointByName(WSG50_RIGHT_FINGER, model_index)
    return [left_joint.position_lower_limits()[0], right_joint.position_upper_limits()[0]]


def close_wsg50_gripper(mbp, context, model_index): # 0.05
    set_joint_positions(get_movable_joints(mbp, model_index), context,
                        get_close_wsg50_positions(mbp, model_index))


def open_wsg50_gripper(mbp, context, model_index):
    set_joint_positions(get_movable_joints(mbp, model_index), context,
                        get_open_wsg50_positions(mbp, model_index))

##################################################

TOOL_Z = 0.025 + 0.075/2 - (-0.049133) # Difference between WSG50 finger tip and body link
DEFAULT_MAX_WIDTH = np.inf
FINGER_WIDTH = 0.02

def get_box_grasps(aabb, max_width=DEFAULT_MAX_WIDTH, grasp_length=0.03,
                   orientations=list(range(4)), pitch_range=(-np.pi/2, np.pi/2)): # y is out of gripper initially
    center, extent = aabb
    dx, dy, dz = extent
    reflect_z = create_transform(rotation=[np.pi / 2, 0, 0])
    aabb_from_body = create_transform(translation=center).inverse()
    min_pitch, max_pitch = pitch_range
    if min_pitch == max_pitch:
        generator = ((min_pitch, ori) for ori in orientations)
    else:
        generator = ((random.uniform(min_pitch, max_pitch), random.choice(orientations)) for _ in count())
    for pitch, orientation in generator:
        d1, d2 = (dx, dy) if (orientation % 2) == 1 else (dy, dx)
        if 2*d2 <= max_width:
            rotate_z = create_transform(rotation=[0, 0, orientation*np.pi/2])
            rotate_x = create_transform(rotation=[pitch, 0, 0])
            threshold = math.atan2(d1, dz)
            distance = dz / math.cos(pitch) if abs(pitch) < threshold else d1 / abs(math.sin(pitch))
            translate_z = create_transform(translation=[0, 0, - TOOL_Z - distance + grasp_length])
            yield reflect_z.multiply(translate_z).multiply(rotate_x).multiply(rotate_z).multiply(aabb_from_body)


def get_cylinder_grasps(aabb, max_width=DEFAULT_MAX_WIDTH, grasp_length=0.02,
                        pitch_range=(-np.pi / 2, np.pi / 2)):  # y is out of gripper initially
    center, extent = aabb
    dx, dy, dz = extent
    radius = (dx + dy) / 2
    if max_width < 2*radius:
        return
    reflect_z = create_transform(rotation=[np.pi / 2, 0, 0])
    aabb_from_body = create_transform(translation=center).inverse()
    translate_axis = create_transform(translation=[0, 0, -dz + FINGER_WIDTH]) # FINGER_WIDTH/2
    while True:
        yaw = random.uniform(0, 2 * np.pi)
        rotate_z = create_transform(rotation=[0, 0, yaw])
        pitch = random.uniform(*pitch_range)
        rotate_x = create_transform(rotation=[pitch, 0, 0])
        translate_z = create_transform(translation=[0, 0, - TOOL_Z + grasp_length])
        yield reflect_z.multiply(translate_z).multiply(rotate_x).multiply(rotate_z). \
            multiply(translate_axis).multiply(aabb_from_body)


##################################################

DOOR_CLOSED = 0.001
DOOR_OPEN = 0.4 * np.pi
# np.pi/2 is the physical max

def get_door_positions(door_body, abs_position):
    name = door_body.name()
    if name == 'left_door':
        return [-abs(abs_position)]
    elif name == 'right_door':
        return [abs(abs_position)]
    raise ValueError(name)
