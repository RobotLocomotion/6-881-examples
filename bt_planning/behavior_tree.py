import argparse
import numpy as np

from pydrake.examples.manipulation_station import ManipulationStation, _xyz_rpy
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import AbstractValue, DiagramBuilder, LeafSystem, BasicVector
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer, MeshcatPointCloudVisualizer
from pydrake.systems.sensors import PixelType
import pydrake.perception as mut
from pydrake.systems.rendering import PoseBundle

from perception_tools.file_utils import LoadCameraConfigFile

from py_trees.blackboard import Blackboard
from behaviors import *
import py_trees
from py_trees.composites import Selector, Sequence
from py_trees.meta import inverter

import time

class BehaviorTree(LeafSystem):

    def __init__(self, root, object_names, door_names, surface_names, plant, plant_context):
        """
        # TODO(kmuhlrad): eventually remove the plant argument, but it makes
        # things easy to start running now

        @system{
          @input_port{point_cloud_id0}
          .
          .
          .
          @input_port{point_cloud_P_idN}
          @output_port{combined_point_cloud_W}
        }
        """
        LeafSystem.__init__(self)

        self.blackboard = Blackboard()
        self.root = root
        self.root.setup_with_descendents(timeout=15)

        self.tick_counter = 0

        self.object_names = object_names
        self.door_names = door_names
        self.surface_names = surface_names

        self.plant = plant
        self.plant_context = plant_context

        self._cupboard_shelves = [
            'bottom', 'shelf_lower', 'shelf_upper', 'top']

        # TODO(kmuhlrad): figure out size of pose_bundle
        self.pose_bundle_input_port = self.DeclareAbstractInputPort(
            "pose_bundle_W", AbstractValue.Make(PoseBundle(num_poses=0)))

        # iiwa position input port
        self.iiwa_position_input_port = \
            self.DeclareVectorInputPort(
                "iiwa_position", BasicVector(7))

        # iiwa velocity input port
        self.iiwa_velocity_input_port = \
            self.DeclareVectorInputPort(
                "iiwa_velocity", BasicVector(7))

        self.gripper_position_input_port = \
            self.DeclareVectorInputPort(
                "gripper_position", BasicVector(1))

        self.gripper_force_input_port = \
            self.DeclareVectorInputPort(
                "gripper_force", BasicVector(1))

        # TODO(kmuhlrad): add output ports

    def UpdateBlackboard(self, pose_bundle, iiwa_q, iiwa_v, wsg_q, wsg_F):
        """
        Update the blackboard according to the input port values.
        This is so conditions can be checked only by reading the blackboard
        and don't need to know about ports at all. This method should always
        be called before ticking the tree.

        @param iiwa_q: The current iiwa joint angles.
        @param iiwa_v: The curernt iiwa joint velocities.
        @param wsg_q: The current gripper setpoint.
        @param wsg_F: The current gripper force.
        """

        # robot_moving
        if iiwa_v == 0:
            self.blackboard.set("robot_moving", False)
        else:
            self.blackboard.set("robot_moving", True)

        # TODO(kmuhlrad): remove plant, make less specific
        # obj_on, surface
        for obj in self.object_names:
            for surface in self.surface_names:
                cupboard = self.plant.GetModelInstanceByName("cupboard")
                shelf_name = self.plant.GetBodyByName("top_and_bottom", cupboard)
                shelf_index = self._cupboard_shelves.index(surface)
                shelf_body = self.plant.GetBodyByName(shelf_name, shelf_index)
                shelf_pose = self.plant.EvalBodyPoseInWorld(
                    self.plant_context, shelf_body)

                obj_body = self.plant.GetBodyByName(obj, 0)
                obj_pose = self.plant.EvalBodyPoseInWorld(
                    self.plant_context, obj_body)

                z_diff = obj_pose.translation()[2] - shelf_pose.translation()[2]
                if abs(z_diff) < 0.01:
                    self.blackboard.set("{}_on".format(obj), surface)

        # door_open
        for door in self.door_names:
            door_angle = self.plant.GetJointByName(
                "{}_hinge".format(door)).get_angle(self.plant_context)
            if abs(door_angle) >= np.pi / 3:
                self.blackboard.set("{}_open".format(door), True)
            else:
                self.blackboard.set("{}_open".format(door), False)

        # robot_holding, obj
        for obj in self.object_names:
            if wsg_q > 0 and wsg_F > 0:
                obj_body = self.plant.GetBodyByName(obj, 0)
                obj_pose = self.plant.EvalBodyPoseInWorld(
                    self.plant_context, obj_body)

                gripper_body = self.plant.GetBodyByName("gripper", 0)
                gripper_pose = self.plant.EvalBodyPoseInWorld(
                    self.plant_context, gripper_body)

                distance = np.linalg.norm(
                    obj_pose.translation() - gripper_pose.translation())
                if distance < 0.01:
                    self.blackboard.set("robot_holding", obj)


    def Tick(self, context, output):
        # Evaluate ports
        pose_bundle = self.EvalAbstractInput(
            context, self.pose_bundle_input_port.get_index()).get_value()
        iiwa_q = self.EvalAbstractInput(
            context, self.iiwa_position_input_port.get_index()).get_value()
        iiwa_v = self.EvalAbstractInput(
            context, self.iiwa_velocity_input_port.get_index()).get_value()
        wsg_q = self.EvalAbstractInput(
            context, self.gripper_position_input_port.get_index()).get_value()
        wsg_F = self.EvalAbstractInput(
            context, self.gripper_force_input_port.get_index()).get_value()

        self.UpdateBlackboard(pose_bundle, iiwa_q, iiwa_v, wsg_q, wsg_F)

        print("\n--------- Tick {0} ---------\n".format(self.tick_counter))
        self.root.tick_once()
        print("\n")
        print("{}".format(py_trees.display.print_ascii_tree(self.root, show_status=True)))
        print(self.blackboard)
        time.sleep(1.0)

        self.tick_counter += 1

        # output.get_mutable_value()


def make_root():
    # conditions
    holding_soup = Holding("soup")
    soup_on_shelf = On("soup", "bottom_shelf")
    left_door_open = DoorOpen("left_door")
    moving_inverter = inverter(RobotMoving)("MovingInverter")
    gripper_empty = inverter(Holding)("soup", "GripperEmpty")

    # actions
    pick_soup = PickDrake("soup")
    place_soup = PlaceDrake("soup", "bottom_shelf")
    open_left_door = OpenDoorDrake("left_door")

    root = Selector(name="Root")

    soup_on_shelf_seq = Sequence(name="SoupOnShelfSeq")

    open_door_sel = Selector(name="OpenDoorSel")
    open_door_seq = Sequence(name="OpenDoorSeq")

    pick_soup_sel = Selector(name="PickSoupSel")
    pick_soup_seq = Sequence(name="PickSoupSeq")

    soup_on_shelf_seq.add_children([open_door_sel, pick_soup_sel, place_soup])
    open_door_sel.add_children([left_door_open, open_door_seq])
    open_door_seq.add_children(
        [gripper_empty, moving_inverter, open_left_door])

    pick_soup_sel.add_children([holding_soup, pick_soup_seq])
    pick_soup_seq.add_children([gripper_empty, moving_inverter, pick_soup])

    root.add_children([soup_on_shelf, soup_on_shelf_seq])

    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    MeshcatVisualizer.add_argparse_argument(parser)
    args = parser.parse_args()

    py_trees.logging.level = py_trees.logging.Level.DEBUG
    root = make_root()

    builder = DiagramBuilder()

    # Create the ManipulationStation.
    station = builder.AddSystem(ManipulationStation())
    station.SetupDefaultStation()
    soup_file = ""
    X_WSoup = None
    station.AddManipulandFromFile(soup_file, X_WSoup)
    station.Finalize()

    meshcat = builder.AddSystem(MeshcatVisualizer(
        station.get_scene_graph(), zmq_url=args.meshcat,
        open_browser=args.open_browser))
    builder.Connect(station.GetOutputPort("pose_bundle"),
                    meshcat.get_input_port(0))

    object_names = ["soup"]
    door_names = ["left_door"]
    surface_names = ["shelf_lower"]

    # Create the BehaviorTree
    plant = station.get_multibody_plant()
    bt = BehaviorTree(root,
                      object_names,
                      door_names,
                      surface_names,
                      plant,
                      plant.get_mutable_context())

    builder.Connect(station.GetOutputPort("pose_bundle"),
                    bt.GetInputPort("pose_bundle"))
    builder.Connect(station.GetOutputPort("iiwa_position_measured"),
                    bt.GetInputPort("iiwa_position"))
    builder.Connect(station.GetOutputPort("iiwa_velocity_estimated"),
                    bt.GetInputPort("iiwa_velocity"))
    builder.Connect(station.GetOutputPort("wsg_state_measured"),
                    bt.GetInputPort("gripper_position"))
    builder.Connect(station.GetOutputPort("wsg_force_measured"),
                    bt.GetInputPort("gripper_force"))

    # TODO(kmuhlrad): hook up the beginning of the manipulation station

    # build diagram
    diagram = builder.Build()

    # construct simulator
    simulator = Simulator(diagram)

    station_context = diagram.GetMutableSubsystemContext(
        station, simulator.get_mutable_context())

    q0 = station.GetIiwaPosition(station_context)
    station_context.FixInputPort(station.GetInputPort(
        "iiwa_position").get_index(), q0)
    station_context.FixInputPort(station.GetInputPort(
        "iiwa_feedforward_torque").get_index(), np.zeros(7))
    station_context.FixInputPort(station.GetInputPort(
        "wsg_position").get_index(), np.array([0.1]))
    station_context.FixInputPort(station.GetInputPort(
        "wsg_force_limit").get_index(), np.array([40.0]))

    simulator.set_publish_every_time_step(False)
    simulator.set_target_realtime_rate(1.0)
    simulator.StepTo(0.1)
