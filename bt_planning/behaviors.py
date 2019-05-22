from py_trees.behaviour import Behaviour
from py_trees.common import Status
from py_trees.blackboard import Blackboard

from planning import (
    GenerateApproachHandlePlans, InterpolateYawAngle, InverseKinPointwise,
    InterpolateStraightLine, ReturnConstantOrientation, MakePlanData,
    GetShelfPose, MakeZeroOrderHold, GetKukaQKnots, GRIPPER_OPEN,
    GRIPPER_CLOSED, MakeReturnHomePlan, p_WQ_home)

import numpy as np

########################## Conditions ##########################


class RobotMoving(Behaviour):
    def __init__(self, name="RobotMoving"):
        """
        Checks if the robot is currently to moving. Returns SUCCESS if true,
        else returns FAILURE.

        @param name str. The name of the BT node.
        """
        super(RobotMoving, self).__init__(name)
        self.blackboard = Blackboard()


    def update(self):
        if self.blackboard.get("robot_moving"):
            self.feedback_message = "The robot is moving"
            return Status.SUCCESS
        self.feedback_message = "The robot is not moving"
        return Status.FAILURE


class Holding(Behaviour):
    def __init__(self, obj, name="Holding"):
        """
        Checks if the robot is holding the specified object. Returns SUCCESS if
        true, else returns FAILURE.

        @param obj str. The name of the object of interest, such as "soup".
        @param name str. The name of the BT node.
        """
        super(Holding, self).__init__(name)
        self.blackboard = Blackboard()
        self.obj = obj

    def update(self):
        if self.blackboard.get("robot_holding") == self.obj:
            self.feedback_message = "The robot is holding {}".format(self.obj)
            return Status.SUCCESS
        self.feedback_message = "The robot is not holding {}".format(self.obj)
        return Status.FAILURE


class On(Behaviour):
    def __init__(self, obj, surface, name="On"):
        """
        Checks if the given object is on the given surface. Returns SUCCESS if
        true, else returns FAILURE.

        @param obj str. The name of the object, such as "soup".
        @param surace str. The name of the desired surface, such as
            "shelf_lower".
        @param name str. The name of the BT node.
        """
        super(On, self).__init__(name)
        self.blackboard = Blackboard()
        self.obj = obj
        self.surface = surface

    def update(self):
        if self.blackboard.get("{}_on".format(self.obj)) == self.surface:
            self.feedback_message = "{} is on {}".format(
                self.obj, self.surface)
            return Status.SUCCESS
        self.feedback_message = "{} is not on {}".format(
            self.obj, self.surface)
        return Status.FAILURE


class DoorOpen(Behaviour):
    """
    Checks if the given door is open. Returns SUCCESS if true, else returns
    FAILURE.

    @param door str. The name of the door, such as "left_door".
    @param name str. The name of the BT node.
    """
    def __init__(self, door, name="DoorOpen"):
        super(DoorOpen, self).__init__(name)
        self.blackboard = Blackboard()
        self.door = door

    def update(self):
        if self.blackboard.get("{}_open".format(self.door)):
            self.feedback_message = "{} is open.".format(self.door)
            return Status.SUCCESS
        self.feedback_message = "{} is not open.".format(self.door)
        return Status.FAILURE

########################## Light Actions ##########################


class Pick(Behaviour):
    def __init__(self, obj, name="Pick"):
        """
        Pick up the specified object. When the robot finishes and is
        holding the object, the node returns SUCCESS. While picking up the
        object, the node returns RUNNING. If the robot is unable to pick up the
        object, it will stop and the node will return FAILURE.

        @param obj str. The name of the object to pick, such as "soup".
        @param name str. The name of the BT node.
        """
        super(Pick, self).__init__(name)
        self.blackboard = Blackboard()
        self.obj = obj

    def initialise(self):
        self.counter = 0

    def update(self):
        if self.counter < 1:
            self.blackboard.set("robot_moving", True)
            self.blackboard.set("robot_holding", None)
            self.counter += 1
            return Status.RUNNING

        self.blackboard.set("robot_moving", False)
        self.blackboard.set("robot_holding", self.obj)

        return Status.SUCCESS


class Place(Behaviour):
    def __init__(self, obj, surface, name="Place"):
        """
        Place up the specified object on the given surface. When the robot
        finishes and is no longer holding the object, the node returns SUCCESS.
        While placing the object, the node returns RUNNING. If the robot is
        unable to place the object, it will stop and the node will return
        FAILURE.

        @param obj str. The name of the object to place, such as "soup".
        @param surface str. The surface on which to place the object such as
            "bottom_shelf".
        @param name str. The name of the BT node.
        """
        super(Place, self).__init__(name)
        self.blackboard = Blackboard()
        self.obj = obj
        self.surface = surface

    def initialise(self):
        self.counter = 0

    def update(self):
        if self.counter < 1:
            self.blackboard.set("robot_moving", True)
            self.blackboard.set("robot_holding", self.obj)
            self.counter += 1
            return Status.RUNNING

        self.blackboard.set("robot_moving", False)
        self.blackboard.set("robot_holding", None)
        self.blackboard.set("{}_on".format(self.obj), self.surface)

        return Status.SUCCESS


class OpenDoor(Behaviour):
    def __init__(self, door, name="OpenDoor"):
        """
        Open the specified door. When the robot finishes opening the door, the
        node returns SUCCESS. While opening the door, the node returns RUNNING.
        If the robot is unable to open the door, it will stop and the node will
        return FAILURE.

        @param door str. The name of the door to open, such as "left_door".
        @param name str. The name of the BT node.
        """
        super(OpenDoor, self).__init__(name)
        self.blackboard = Blackboard()
        self.door = door


    def initialise(self):
        self.counter = 0

    def update(self):
        if self.counter < 1:
            self.blackboard.set("robot_moving", True)
            self.blackboard.set("robot_holding", self.door)
            self.counter += 1
            return Status.RUNNING

        self.blackboard.set("{}_open".format(self.door), True)
        self.blackboard.set("robot_moving", False)
        self.blackboard.set("robot_holding", None)

        return Status.SUCCESS


########################## Drake Actions ##########################


class PickDrake(Behaviour):
    def __init__(self, obj, name="Pick"):
        """
        Pick up the specified object. When the robot finishes and is
        holding the object, the node returns SUCCESS. While picking up the
        object, the node returns RUNNING. If the robot is unable to pick up the
        object, it will stop and the node will return FAILURE.

        @param obj str. The name of the object to pick, such as "soup".
        @param name str. The name of the BT node.
        """
        super(PickDrake, self).__init__(name)
        self.blackboard = Blackboard()
        self.obj = obj

    def initialise(self):
        self.sent = False
        self.plans = []
        self.gripper_setpoints = []
        self.blackboard.set("sent_new_plan", False)

    def update(self):
        if not self.blackboard.get("robot_moving"):
            # if self.blackboard.get("robot_holding") == self.obj:
            #     self.feedback_message = "Successfully picked up {}".format(
            #         self.obj)
            #     return Status.SUCCESS
            if not self.sent:
                X_WObj = self.blackboard.get("{}_pose".format(self.obj))
                #p_WQ_end = X_WObj.multiply(np.array([0.015, -0.03, 0.04]))
                # TODO(kmuhlrad): generalize this a bit more
                p_WQ_end = X_WObj.multiply(np.array([-0.035, -0.01, -0.004]))
                angle_start = np.pi * 150 / 180.

                qtraj, q_knots = InverseKinPointwise(
                    self.blackboard.get("p_WQ"), p_WQ_end, angle_start,
                    angle_start, 5.0, 15,
                    q_initial_guess=self.blackboard.get("prev_q_full"),
                    InterpolatePosition=InterpolateStraightLine,
                    InterpolateOrientation=ReturnConstantOrientation,
                    is_printing=True)

                q_knots_kuka = GetKukaQKnots(q_knots[-1])

                plan_go_home = MakeReturnHomePlan(q_knots_kuka.squeeze(), duration=3)

                self.plans = [MakePlanData(qtraj),
                              MakeZeroOrderHold(q_knots_kuka),
                              plan_go_home]
                self.gripper_setpoints = [
                    self.blackboard.get("gripper_setpoint"),
                    GRIPPER_CLOSED,
                    GRIPPER_CLOSED]

                self.blackboard.set("prev_q_full", q_knots[-1])
                self.blackboard.set("next_plan_data", self.plans.pop(0))
                self.blackboard.set("gripper_setpoint",
                                    self.gripper_setpoints.pop(0))
                self.blackboard.set("sent_new_plan", True)

                self.sent = True
                self.feedback_message = "Sent plan to pick up {}".format(
                    self.obj)
                return Status.RUNNING
            elif len(self.plans):
                self.blackboard.set("next_plan_data",
                                    self.plans.pop(0))
                self.blackboard.set("gripper_setpoint",
                                    self.gripper_setpoints.pop(0))
                self.blackboard.set("sent_new_plan", True)

                self.feedback_message = (
                    "Sent the next plan to pick up {}".format(self.obj))
                return Status.RUNNING
            elif not self.blackboard.get("robot_holding") == self.obj:
                self.blackboard.set("sent_new_plan", False)
                self.feedback_message = "Could not pick up {}".format(self.obj)
                return Status.FAILURE
            else:
                self.feedback_message = "Successfully picked up {}".format(
                    self.obj)
                return Status.SUCCESS

        self.blackboard.set("sent_new_plan", False)
        self.feedback_message = "Picking up {}".format(self.obj)
        return Status.RUNNING


class PlaceDrake(Behaviour):
    def __init__(self, obj, surface, name="Place"):
        """
        Place up the specified object on the given surface. When the robot
        finishes and is no longer holding the object, the node returns SUCCESS.
        While placing the object, the node returns RUNNING. If the robot is
        unable to place the object, it will stop and the node will return
        FAILURE.

        @param obj str. The name of the object to place, such as "soup".
        @param surface str. The surface on which to place the object such as
            "shelf_lower".
        @param name str. The name of the BT node.
        """
        super(PlaceDrake, self).__init__(name)
        self.blackboard = Blackboard()
        self.obj = obj
        self.surface = surface

    def initialise(self):
        self.sent = False
        self.plans = []
        self.gripper_setpoints = []
        self.blackboard.set("sent_new_plan", False)

    def update(self):
        if (self.blackboard.get("{}_on".format(self.obj)) == self.surface
                and not self.blackboard.get("robot_moving")):
            return Status.SUCCESS

        if not self.blackboard.get("robot_moving"):
            if not self.sent:
                plan_go_home = MakeReturnHomePlan(
                    self.blackboard.get("iiwa_q"), duration=5)

                p_WQ_end = GetShelfPose(self.surface)
                angle_start = np.pi * 150 / 180.

                qtraj, q_knots = InverseKinPointwise(
                    p_WQ_home, p_WQ_end, angle_start,
                    angle_start, 10.0, 15,
                    q_initial_guess=self.blackboard.get("prev_q_full"),
                    InterpolatePosition=InterpolateStraightLine,
                    InterpolateOrientation=ReturnConstantOrientation,
                    is_printing=True)

                q_knots_kuka = GetKukaQKnots(q_knots[-1])
                self.plans = [plan_go_home,
                              MakePlanData(qtraj),
                              MakeZeroOrderHold(q_knots_kuka)]
                self.gripper_setpoints = [
                    self.blackboard.get("gripper_setpoint"),
                    self.blackboard.get("gripper_setpoint"),
                    GRIPPER_OPEN]

                self.blackboard.set("prev_q_full", q_knots[-1])
                self.blackboard.set("next_plan_data", self.plans.pop(0))
                self.blackboard.set(
                    "gripper_setpoint", self.gripper_setpoints.pop(0))
                self.blackboard.set("sent_new_plan", True)

                self.sent = True
                self.feedback_message = "Sent plan to place {} on {}".format(
                    self.obj, self.surface)
                return Status.RUNNING
            elif len(self.plans):
                self.blackboard.set("next_plan_data", self.plans.pop(0))
                self.blackboard.set(
                    "gripper_setpoint", self.gripper_setpoints.pop(0))
                self.blackboard.set("sent_new_plan", True)

                self.feedback_message = (
                    "Sent the next plan to place {} on {}".format(
                        self.obj, self.obj))
                return Status.RUNNING
            elif not self.blackboard.get("{}_on".format(self.obj)) == self.surface:
                self.blackboard.set("sent_new_plan", False)
                self.feedback_message = "Could not place {} on {}".format(
                    self.obj, self.surface)
                return Status.FAILURE

        self.blackboard.set("sent_new_plan", False)
        return Status.RUNNING


class OpenDoorDrake(Behaviour):
    def __init__(self, door, name="OpenDoor"):
        """
        Open the specified door. When the robot finishes opening the door, the
        node returns SUCCESS. While opening the door, the node returns RUNNING.
        If the robot is unable to open the door, it will stop and the node will
        return FAILURE.

        @param door str. The name of the door to open, such as "left_door".
        @param name str. The name of the BT node.
        """
        super(OpenDoorDrake, self).__init__(name)
        self.blackboard = Blackboard()
        self.door = door

    def initialise(self):
        self.sent = False
        self.plans = []
        self.gripper_setpoints = []
        self.blackboard.set("sent_new_plan", False)

    def update(self):
        if (self.blackboard.get("{}_open".format(self.door))
                and not self.blackboard.get("robot_moving")):
            return Status.SUCCESS

        if not self.blackboard.get("robot_moving"):
            if not self.sent:
                plan_data_list, gripper_setpoint_list, q_final_full = (
                    GenerateApproachHandlePlans(
                        ReturnConstantOrientation, np.pi/4, is_printing=True))

                self.plans = plan_data_list
                self.gripper_setpoints = gripper_setpoint_list

                self.blackboard.set("prev_q_full", q_final_full[-1])
                self.blackboard.set("next_plan_data", self.plans.pop(0))
                self.blackboard.set(
                    "gripper_setpoint", self.gripper_setpoints.pop(0))
                self.blackboard.set("sent_new_plan", True)

                self.feedback_message = "Sent plan to open {}".format(
                    self.door)
                self.sent = True
                return Status.RUNNING
            elif len(self.plans):
                self.blackboard.set("next_plan_data", self.plans.pop(0))
                self.blackboard.set(
                    "gripper_setpoint", self.gripper_setpoints.pop(0))
                self.blackboard.set("sent_new_plan", True)

                self.feedback_message = (
                    "Sent the next plan to open {}".format(self.door))
                return Status.RUNNING
            elif not self.blackboard.get("{}_open".format(self.door)):
                self.blackboard.set("sent_new_plan", False)
                self.feedback_message = "Could not open {}".format(self.door)
                return Status.FAILURE

        self.blackboard.set("sent_new_plan", False)
        return Status.RUNNING
