from __future__ import print_function

import numpy as np
import os
from pydrake.examples.manipulation_station import ManipulationStation, IiwaCollisionModel
from pydrake.multibody.parsing import Parser
from pydrake.common.eigen_geometry import Isometry3
from pydrake.common import FindResourceOrThrow

from pddl_planning.systems import build_manipulation_station
from pddl_planning.iiwa_utils import DOOR_CLOSED, open_wsg50_gripper
from pddl_planning.utils import get_model_name, create_transform, get_movable_joints, get_body_boxes, get_base_body, \
    get_model_bodies, get_bodies, set_joint_position, set_world_pose, get_box_from_geom, \
    get_aabb_z_placement, get_body_pose, weld_to_world


class Surface(object):
    def __init__(self, plant, model_index, body_name, visual_index):
        self.plant = plant
        self.model_index = model_index
        self.body_name = body_name
        self.visual_index = visual_index
    def __repr__(self):
        return '{}({},{},{})'.format(self.__class__.__name__,
                                     get_model_name(self.plant, self.model_index),
                                     self.body_name, self.visual_index)


class Task(object):
    def __init__(self, diagram, mbp, scene_graph, robot, gripper,
                 movable=[], surfaces=[], doors=[],
                 initial_positions={}, initial_poses={},
                 goal_poses={}, goal_holding=[], goal_on=[], goal_cooked=[],
                 reset_robot=True, reset_doors=True):
        # Drake systems
        self.diagram = diagram
        self.mbp = mbp
        self.scene_graph = scene_graph
        self.diagram_context = diagram.CreateDefaultContext()
        self.plant_context = diagram.GetMutableSubsystemContext(mbp, self.diagram_context)

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
        self.goal_cooked = goal_cooked
        self.reset_robot = reset_robot
        self.reset_doors = reset_doors
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
    def __repr__(self):
        return '{}(robot={}, gripper={}, movable={}, surfaces={})'.format(
            self.__class__.__name__,
            get_model_name(self.mbp, self.robot),
            get_model_name(self.mbp, self.gripper),
            [get_model_name(self.mbp, model) for model in self.movable],
            self.surfaces)

##################################################

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DOPE_PATH = os.path.join(FILE_DIR, "poses.txt")
dz_table_top_robot_base = 0.0127

def read_poses_from_file(filename):
    pose_dict = {}
    row_num = 0
    cur_matrix = np.eye(4)
    with open(filename, "r") as f:
        for line in f:
            line = line.rstrip()
            if not line.lstrip(" ").startswith("["):
                object_name = line
            else:
                row = np.matrix(line)
                cur_matrix[row_num, :] = row
                row_num += 1
                if row_num == 4:
                    pose_dict[object_name] = Isometry3(cur_matrix)
                    translation = pose_dict[object_name].translation()
                    translation[2] += dz_table_top_robot_base
                    pose_dict[object_name].set_translation(translation)
                    pose_dict[object_name].set_rotation(
                        np.array([
                            [1, 0, 0],
                            [0, 0, 1],
                            [0, -1, 0]
                        ])
                    )
                    cur_matrix = np.eye(4)
                row_num %= 4
    return pose_dict

##################################################

CUPBOARD_SHELVES = [
    'bottom',
    'shelf_lower',
    'shelf_upper',
    'top',
]


# https://github.com/kmuhlrad/models/tree/master/ycb_objects
SDF_PATH_FROM_NAME = {
    'soup': "models/ycb/sdf/005_tomato_soup_can.sdf",
    # 'soup': "../../models/ycb_objects/soup_can.sdf",
    'meat': "models/ycb_objects/potted_meat_can.sdf",
    'gelatin': "models/ycb_objects/gelatin_box.sdf",
    'mustard': "models/ycb_objects/mustard_bottle.sdf",
    'cracker': "models/ycb_objects/cracker_box.sdf",
}

PLANE_FILE_PATH = os.path.join(FILE_DIR, "plane.sdf")


def get_sdf_path(model_name):
    return FindResourceOrThrow("drake/manipulation/" + SDF_PATH_FROM_NAME[model_name])
    # return os.path.join(FILE_DIR, SDF_PATH_FROM_NAME[model_name])


def get_z_placement(plant, box_from_geom, item_body, surface_body, surface_index):
    plant_context = plant.CreateDefaultContext()
    surface_aabb, surface_body_from_box, _ = get_body_boxes(surface_body, box_from_geom)[surface_index]
    [(item_aabb, item_body_from_box, _)] = get_body_boxes(get_base_body(plant, item_body), box_from_geom)
    dz = get_aabb_z_placement(item_aabb, surface_aabb, z_epsilon=0)
    surface_box_from_obj_box = create_transform(translation=[0, 0, dz])
    world_pose = get_body_pose(plant_context, surface_body).multiply(surface_body_from_box).multiply(
        surface_box_from_obj_box).multiply(item_body_from_box.inverse())
    _, _, item_z = world_pose.translation()
    return item_z

def load_station(time_step=0.0, **kwargs):
    station = ManipulationStation(time_step)
    plant = station.get_mutable_multibody_plant()
    scene_graph = station.get_mutable_scene_graph()
    station.SetupDefaultStation(IiwaCollisionModel.kBoxCollision)
    robot = plant.GetModelInstanceByName('iiwa')
    gripper = plant.GetModelInstanceByName('gripper')
    table = plant.GetModelInstanceByName('table')
    cupboard = plant.GetModelInstanceByName('cupboard')

    model_name = 'soup'
    parser = Parser(plant=plant)
    item = parser.AddModelFromFile(file_name=get_sdf_path(model_name), model_name=model_name)
    ceiling = parser.AddModelFromFile(file_name=PLANE_FILE_PATH, model_name="ceiling")
    weld_to_world(plant, ceiling, create_transform(translation=[0.3257, 0, 1.0]))
    station.Finalize()

    diagram, state_machine = build_manipulation_station(station, **kwargs)
    box_from_geom = get_box_from_geom(scene_graph)

    table_body = plant.GetBodyByName('amazon_table', table)
    table_index = 0
    #table_surface = Surface(plant, table, table_body.name(), table_index),
    start_z = get_z_placement(plant, box_from_geom, item, table_body, table_index)

    shelf_body = plant.GetBodyByName('top_and_bottom', cupboard)
    shelf_index = CUPBOARD_SHELVES.index('shelf_lower')
    goal_surface = Surface(plant, cupboard, shelf_body.name(), shelf_index)

    door_names = [
        'left_door',
        #'right_door',
    ]
    doors = [plant.GetBodyByName(name).index() for name in door_names]

    initial_positions = {
        plant.GetJointByName('left_door_hinge'): -DOOR_CLOSED,
        plant.GetJointByName('right_door_hinge'): DOOR_CLOSED,
    }

    initial_conf = [0, 0, 0, -1.75, 0, 1.0, 0]
    initial_positions.update(zip(get_movable_joints(plant, robot), initial_conf))

    start_x, start_y, start_theta = 0.4, -0.2, np.pi/2
    initial_poses = {
        item: create_transform(translation=[start_x, start_y, start_z], rotation=[0, 0, start_theta]),
    }

    surfaces = [
        #table_surface,
        goal_surface,
    ]

    task = Task(diagram, plant, scene_graph, robot, gripper,
                movable=[item], surfaces=surfaces, doors=doors,
                initial_positions=initial_positions, initial_poses=initial_poses,
                #goal_holding=[item],
                goal_on=[(item, goal_surface)],
                #goal_poses=goal_poses,
                reset_robot=True, reset_doors=False)
    task.set_initial()
    task.station = station

    return task, diagram, state_machine

##################################################

def load_dope(time_step=0.0, dope_path=DOPE_PATH, goal_name='soup', is_visualizing=True, **kwargs):
    station = ManipulationStation(time_step)
    station.SetupDefaultStation(IiwaCollisionModel.kBoxCollision)
    plant = station.get_mutable_multibody_plant()
    scene_graph = station.get_mutable_scene_graph()
    robot = plant.GetModelInstanceByName('iiwa')
    gripper = plant.GetModelInstanceByName('gripper')
    cupboard = plant.GetModelInstanceByName('cupboard')

    parser = Parser(plant=plant)

    ceiling = parser.AddModelFromFile(file_name=PLANE_FILE_PATH, model_name="ceiling")
    weld_to_world(plant, ceiling, create_transform(translation=[0.3257, 0, 1.1]))

    pose_from_name = read_poses_from_file(dope_path)
    model_from_name = {}
    for name in pose_from_name:
        model_from_name[name] = parser.AddModelFromFile(file_name=get_sdf_path(name), model_name=name)
    station.Finalize()

    door_names = [
        'left_door',
    ]
    doors = [plant.GetBodyByName(name).index() for name in door_names]

    initial_positions = {
        plant.GetJointByName('left_door_hinge'): -DOOR_CLOSED,
        plant.GetJointByName('right_door_hinge'): DOOR_CLOSED,
    }
    initial_conf = [0, 0, 0, -1.75, 0, 1.0, 0]
    initial_positions.update(zip(get_movable_joints(plant, robot), initial_conf))

    initial_poses = {model_from_name[name]: pose_from_name[name] for name in pose_from_name}

    goal_shelf = 'shelf_lower'
    print('Goal shelf:', goal_shelf)
    goal_surface = Surface(plant, cupboard, 'top_and_bottom', CUPBOARD_SHELVES.index(goal_shelf))
    surfaces = [
        goal_surface,
    ]
    item = model_from_name[goal_name]

    diagram, state_machine = build_manipulation_station(station, visualize=is_visualizing, **kwargs)
    task = Task(diagram, plant, scene_graph, robot, gripper,
                movable=[item], surfaces=surfaces, doors=doors,
                initial_positions=initial_positions, initial_poses=initial_poses,
                #goal_holding=[item],
                goal_on=[(item, goal_surface)],
                reset_robot=True, reset_doors=False)
    task.set_initial()
    task.station = station

    return task, diagram, state_machine
