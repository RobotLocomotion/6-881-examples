import argparse
import numpy as np

from pydrake.examples.manipulation_station import ManipulationStation
from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw
from pydrake.manipulation.robot_plan_runner import RobotPlanRunner
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import (
    AbstractValue, DiagramBuilder, LeafSystem, BasicVector)
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.systems.rendering import PoseBundle

import py_trees
from py_trees.blackboard import Blackboard
from behaviors import (
    RobotMoving, On, Holding, DoorOpen, PickDrake, PlaceDrake, OpenDoorDrake)
from py_trees.common import Status
from py_trees.composites import Selector, Sequence
from py_trees.meta import inverter

from bt_planning.planning import Calc_p_WQ, MakeIKGuess, MakeReturnHomePlan


class BehaviorTree(LeafSystem):

    def __init__(self, root, object_names, door_names, surface_names):
        """
        A System that takes in a py_trees.Behaviour object, names of objects
        in the world, and runs the encoded behavior tree. The system writes
        the state of the world represented by the context to the tree's
        py_trees.Blackboard object. The underlying py_trees.Behaviour object
        does not need any knowledge about the Systems framework. The tree is
        ticked once every second. Be aware that this System is customized
        towards the stowing task of opening a door, picking an object up from
        the table, and putting that object on a shelf of the door cabinet. In
        order to do different tasks, such as pouring a glass of water, a this
        System needs to be modified in addition to the given root
        py_trees.Behaviour.

        @param root: A py_trees.Behaviour object representing the root of a
            py_trees.BehaviourTree
        @param object_names: A list of strings representing all of the
            pick-able objects in the world, such as ["soup", "gelatin"].
        @param door_names: A list of all of the doors, such as ["left_door",
            "right_door"].
        @param surface_names: A list of all of the surfaces such as ["bottom",
            "shelf_lower"].

        @system{
          @input_port{iiwa_position}
          @input_port{iiwa_velocity}
          @input_port{iiwa_torque}
          @input_port{gripper_position}
          @input_port{gripper_force}
          @input_port{pose_bundle_W}
          @output_port{status}
          @output_port{plan_data}
          @output_port{gripper_setpoint}
        }
        """
        LeafSystem.__init__(self)

        self.blackboard = Blackboard()
        self.root = root
        self.root.setup(timeout=15)

        self.tick_counter = 0

        self.object_names = object_names
        self.door_names = door_names
        self.surface_names = surface_names

        self._cupboard_shelves = [
            'bottom', 'shelf_lower', 'shelf_upper', 'top']

        self.pose_bundle_input_port = self.DeclareAbstractInputPort(
            "pose_bundle_W", AbstractValue.Make(PoseBundle(num_poses=0)))

        self.iiwa_position_input_port = \
            self.DeclareVectorInputPort(
                "iiwa_position", BasicVector(7))

        self.iiwa_velocity_input_port = \
            self.DeclareVectorInputPort(
                "iiwa_velocity", BasicVector(7))

        self.iiwa_torque_input_port = \
            self.DeclareVectorInputPort(
                "iiwa_torque", BasicVector(7))

        self.gripper_position_input_port = \
            self.DeclareVectorInputPort(
                "gripper_position", BasicVector(2))

        self.gripper_force_input_port = \
            self.DeclareVectorInputPort(
                "gripper_force", BasicVector(1))

        self.status_output_port = self.DeclareVectorOutputPort(
            "status", BasicVector(1), self.DoCalcRootStatus)

        q_start = np.array([0, 0, 0, -1.75, 0, 1.0, 0])
        self.DeclareAbstractOutputPort("plan_data",
                                       lambda: AbstractValue.Make(
                                           MakeReturnHomePlan(q_start)),
                                       self.DoCalcNextPlan)

        self.gripper_setpoint_output_port = self.DeclareVectorOutputPort(
            "gripper_setpoint", BasicVector(1), self.CalcGripperSetpoint,
            prerequisites_of_calc=set([self.all_state_ticket()]))

        self.DeclarePeriodicPublish(1, 1.0)

    def _ParsePoses(self, pose_bundle):
        pose_dict = {}

        for i in range(pose_bundle.get_num_poses()):
            if pose_bundle.get_name(i):
                pose_dict[pose_bundle.get_name(i).split("::")[-1]] = (
                    pose_bundle.get_pose(i))

        return pose_dict

    def UpdateBlackboard(
            self, pose_dict, iiwa_q, iiwa_v, iiwa_T, wsg_q, wsg_F):
        """
        Update the blackboard according to the input port values.
        This is so conditions can be checked only by reading the blackboard
        and don't need to know about ports at all. This method should always
        be called before ticking the tree.

        @param pose_dict: A dictionary of 4x4 poses keyed by body name.
        @param iiwa_q: The current iiwa joint angles.
        @param iiwa_v: The current iiwa joint velocities.
        @param iiwa_T: The current external torque on the iiwa joints.
        @param wsg_q: The current gripper [position, velocity] vector.
        @param wsg_F: The current gripper force.
        """

        # iiwa_q
        self.blackboard.set("iiwa_q", iiwa_q.get_value())
        self.blackboard.set("p_WQ", Calc_p_WQ(iiwa_q.get_value()))
        self.blackboard.set("prev_q_full", MakeIKGuess(iiwa_q.get_value()))

        # robot_moving
        if np.any(iiwa_v.get_value() > 0.001) or wsg_q.GetAtIndex(1) > 0.001:
            self.blackboard.set("robot_moving", True)
        else:
            self.blackboard.set("robot_moving", False)

        # iiwa_T
        self.blackboard.set("iiwa_T", iiwa_T.get_value())

        # obj_on, surface
        shelf_thickness = 0.016 / 2.
        surface_translations = {
            'bottom': -0.3995,
            'shelf_lower': -0.13115,
            'shelf_upper': 0.13155,
            'top': 0.3995
        }
        obj_dims = {
            'cracker': 0.21 / 2.,
            'sugar': 0.18 / 2.,
            'soup': 0.1 / 2.,
            'mustard': 0.19 / 2.,
            'gelatin': 0.073 / 2.,
            'meat': 0.084 / 2.
        }
        for obj in self.object_names:
            for surface in self.surface_names:
                cupboard_pose = pose_dict["top_and_bottom"]
                shelf_z = (cupboard_pose.translation()[2]
                           + surface_translations[surface])

                try:
                    obj_dim = obj_dims[obj]
                    obj_pose = pose_dict["base_link_{}".format(obj)]
                except:
                    continue

                self.blackboard.set("{}_pose".format(obj), obj_pose)
                z_diff = obj_pose.translation()[2] - shelf_z
                if abs(z_diff) < shelf_thickness + obj_dim + 0.01:
                    self.blackboard.set("{}_on".format(obj), surface)

        # door_open
        for door in self.door_names:
            yaw_angle = RollPitchYaw(pose_dict[door].rotation()).yaw_angle()
            door_angle = (yaw_angle + 2*np.pi) % (2 * np.pi ) - np.pi
            if abs(door_angle) >= np.pi / 3:
                self.blackboard.set("{}_open".format(door), True)
            else:
                self.blackboard.set("{}_open".format(door), False)

        # robot_holding, obj
        gripper_dim = 0.0725
        holding = False
        for obj in self.object_names:
            if 0.005 < wsg_q.GetAtIndex(0) < 0.1 and wsg_F.GetAtIndex(0) > 3:
                try:
                    obj_dim = obj_dims[obj]
                    obj_pose = pose_dict["base_link_{}".format(obj)]
                except:
                    continue
                gripper_pose = pose_dict["body"]

                distance = np.linalg.norm(
                    obj_pose.translation() - gripper_pose.translation())
                if distance <= obj_dim + gripper_dim + 0.01:
                    holding = True
                    self.blackboard.set("robot_holding", obj)
                    break
        if not holding:
            self.blackboard.set("robot_holding", None)

    def DoPublish(self, context, events):
        self.TickOnce(context)

    def TickOnce(self, context):
        # Evaluate ports
        pose_bundle = self.EvalAbstractInput(
            context, self.pose_bundle_input_port.get_index()).get_value()
        iiwa_q = self.EvalAbstractInput(
            context, self.iiwa_position_input_port.get_index()).get_value()
        iiwa_v = self.EvalAbstractInput(
            context, self.iiwa_velocity_input_port.get_index()).get_value()
        iiwa_T = self.EvalAbstractInput(
            context, self.iiwa_torque_input_port.get_index()).get_value()
        wsg_q = self.EvalAbstractInput(
            context, self.gripper_position_input_port.get_index()).get_value()
        wsg_F = self.EvalAbstractInput(
            context, self.gripper_force_input_port.get_index()).get_value()

        self.UpdateBlackboard(self._ParsePoses(pose_bundle),
                              iiwa_q, iiwa_v, iiwa_T, wsg_q, wsg_F)

        print("\n--------- Tick {0} ---------\n".format(self.tick_counter))
        self.root.tick_once()
        print("\n")
        print("{}".format(py_trees.display.print_ascii_tree(
            self.root, show_status=True)))
        print(self.blackboard)

        self.tick_counter += 1

    def DoCalcRootStatus(self, context, output):
        if self.root.status == Status.SUCCESS:
            output.get_mutable_value()[0] = 0
        elif self.root.status == Status.RUNNING:
            output.get_mutable_value()[0] = 1
        else:
            output.get_mutable_value()[0] = 2

    def DoCalcNextPlan(self, context, output):
        if self.blackboard.get("sent_new_plan"):
            next_plan_data = self.blackboard.get("next_plan_data")
            output.set_value(next_plan_data)

    def CalcGripperSetpoint(self, context, output):
        output.get_mutable_value()[0] = self.blackboard.get("gripper_setpoint")


def make_root(
        obj_name="soup", shelf_name="shelf_lower", door_name="left_door"):
    """
    Constructs the root py_trees.Behaviour of a py_trees.BehaviourTree that
    lets the robot open door_name, pick up obj_name, and put it on shelf_name.

    @param obj_name str. The name of the object to pick, such as "soup".
    @param shelf_name str. The name of the shelf to place the object on, such
        as "shelf_lower".
    @param door_name str. The name of the door to open, such as "left_door".
    @return a py_trees.Behaviour object.
    """

    # conditions
    holding_soup = Holding(obj_name)
    soup_on_shelf = On(obj_name, shelf_name)
    left_door_open = DoorOpen(door_name)
    moving_inverter = inverter(RobotMoving)("MovingInverter")
    gripper_empty = inverter(Holding)(obj=obj_name, name="GripperEmpty")

    # actions
    pick_soup = PickDrake(obj_name)
    place_soup = PlaceDrake(obj_name, shelf_name)
    open_left_door = OpenDoorDrake(door_name)

    root = Selector(name="Root")

    soup_on_shelf_seq = Sequence(name="ObjOnShelfSeq")

    open_door_sel = Selector(name="OpenDoorSel")
    open_door_seq = Sequence(name="OpenDoorSeq")

    pick_soup_sel = Selector(name="PickObjSel")
    pick_soup_seq = Sequence(name="PickObjSeq")

    soup_on_shelf_seq.add_children([open_door_sel, pick_soup_sel, place_soup])
    open_door_sel.add_children([left_door_open, open_door_seq])
    open_door_seq.add_children(
        [gripper_empty, moving_inverter, open_left_door])

    pick_soup_sel.add_children([holding_soup, pick_soup_seq])
    pick_soup_seq.add_children([gripper_empty, moving_inverter, pick_soup])

    root.add_children([soup_on_shelf, soup_on_shelf_seq])

    return root


def open_door_test():
    """
    Constructs the root py_trees.Behaviour of a py_trees.BehaviourTree that
    tests the robot just opening the left door.

    @return a py_trees.Behaviour object.
    """

    # conditions
    left_door_open = DoorOpen("left_door")
    moving_inverter = inverter(RobotMoving)("MovingInverter")
    gripper_empty = inverter(Holding)("soup", name="GripperEmpty")

    # actions
    open_left_door = OpenDoorDrake("left_door")

    root = Selector(name="Root")

    open_door_seq = Sequence(name="OpenDoorSeq")

    open_door_seq.add_children(
        [gripper_empty, moving_inverter, open_left_door])

    root.add_children([left_door_open, open_door_seq])

    return root


def pick_test(obj_name="soup"):
    """
    Constructs the root py_trees.Behaviour of a py_trees.BehaviourTree that
    tests the robot just grabbing obj_name on the table.

    @param obj_name str. The name of the object to grab, such as "soup".
    @return a py_trees.Behaviour object.
    """

    # conditions
    holding_soup = Holding(obj_name)
    moving_inverter = inverter(RobotMoving)("MovingInverter")
    gripper_empty = inverter(Holding)(obj_name, name="GripperEmpty")

    # actions
    pick_soup = PickDrake(obj_name)

    root = Selector(name="Root")

    pick_soup_seq = Sequence(name="PickObjSeq")

    pick_soup_seq.add_children(
        [gripper_empty, moving_inverter, pick_soup])

    root.add_children([holding_soup, pick_soup_seq])

    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    MeshcatVisualizer.add_argparse_argument(parser)
    args = parser.parse_args()

    py_trees.logging.level = py_trees.logging.Level.DEBUG
    root = pick_test(obj_name="gelatin")

    # Initialize plan list to empty
    blackboard = Blackboard()
    blackboard.set("sent_new_plan", False)
    blackboard.set("gripper_setpoint", 0.07)

    builder = DiagramBuilder()

    plan_runner = builder.AddSystem(RobotPlanRunner(is_discrete=True))

    # Create the ManipulationStation.
    station = builder.AddSystem(ManipulationStation())
    station.SetupDefaultStation()

    obj_file = "drake/manipulation/models/ycb/sdf/009_gelatin_box.sdf"
    X_WObject = RigidTransform(
        RotationMatrix(np.array([[1, 0,  0],
                                 [ 0, 0, -1 ],
                                 [ 0, 1,  0 ]])),
        np.array([0.53777627, -0.17532787, 0.03030285]))
    station.AddManipulandFromFile(obj_file, X_WObject)

    station.Finalize()

    meshcat = builder.AddSystem(MeshcatVisualizer(
        station.get_scene_graph(), zmq_url=args.meshcat,
        open_browser=args.open_browser))
    builder.Connect(station.GetOutputPort("pose_bundle"),
                    meshcat.get_input_port(0))

    object_names = ["gelatin"]
    door_names = ["left_door", "right_door"]
    surface_names = ["shelf_lower"]

    # Create the BehaviorTree
    bt = builder.AddSystem(
        BehaviorTree(root, object_names, door_names, surface_names))

    builder.Connect(station.GetOutputPort("pose_bundle"),
                    bt.GetInputPort("pose_bundle_W"))
    builder.Connect(station.GetOutputPort("iiwa_position_measured"),
                    bt.GetInputPort("iiwa_position"))
    builder.Connect(station.GetOutputPort("iiwa_velocity_estimated"),
                    bt.GetInputPort("iiwa_velocity"))
    builder.Connect(station.GetOutputPort("iiwa_torque_external"),
                    bt.GetInputPort("iiwa_torque"))
    builder.Connect(station.GetOutputPort("wsg_state_measured"),
                    bt.GetInputPort("gripper_position"))
    builder.Connect(station.GetOutputPort("wsg_force_measured"),
                    bt.GetInputPort("gripper_force"))

    builder.Connect(station.GetOutputPort("iiwa_position_measured"),
                    plan_runner.GetInputPort("iiwa_position_measured"))
    builder.Connect(station.GetOutputPort("iiwa_velocity_estimated"),
                    plan_runner.GetInputPort("iiwa_velocity_estimated"))
    builder.Connect(station.GetOutputPort("iiwa_torque_external"),
                    plan_runner.GetInputPort("iiwa_torque_external"))

    builder.Connect(bt.GetOutputPort("plan_data"),
                    plan_runner.GetInputPort("plan_data"))
    builder.Connect(bt.GetOutputPort("gripper_setpoint"),
                    station.GetInputPort("wsg_position"))

    builder.Connect(plan_runner.GetOutputPort("iiwa_position_command"),
                    station.GetInputPort("iiwa_position"))
    builder.Connect(plan_runner.GetOutputPort("iiwa_torque_command"),
                    station.GetInputPort("iiwa_feedforward_torque"))

    # build diagram
    diagram = builder.Build()

    # construct simulator
    simulator = Simulator(diagram)

    station_context = diagram.GetMutableSubsystemContext(
        station, simulator.get_mutable_context())

    q_start = np.array([0, 0, 0, -1.75, 0, 1.0, 0])
    station.SetIiwaPosition(station_context, q_start)

    blackboard.set("prev_q_full", MakeIKGuess(q_start))

    station_context.FixInputPort(station.GetInputPort(
        "wsg_force_limit").get_index(), np.array([40.0]))

    # Open both doors for testing
    hinge_joint = station.get_multibody_plant().GetJointByName(
        "right_door_hinge")
    hinge_joint.set_angle(
        context=station.GetMutableSubsystemContext(
            station.get_multibody_plant(), station_context), angle=-np.pi/2)

    hinge_joint = station.get_multibody_plant().GetJointByName(
        "left_door_hinge")
    hinge_joint.set_angle(
        context=station.GetMutableSubsystemContext(
            station.get_multibody_plant(), station_context), angle=-np.pi/2)

    simulator.set_publish_every_time_step(False)
    simulator.set_target_realtime_rate(0.5)
    simulator.StepTo(15)

if __name__ == "__main__":
    main()