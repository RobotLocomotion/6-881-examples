from py_trees.behaviour import Behaviour
from py_trees.common import Status
from py_trees.blackboard import Blackboard

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
            "bottom_shelf".
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

########################## Actions ##########################


# TODO(kmuhlrad): finish implementations of things and figure out if all
# params are actually needed for the actions and not just condition checking
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
        # TODO(kmuhlrad): figure this out

        # if not holding object:
        #   set robot_moving to True
        #   return RUNNING

        # if some error occurs, return FAILURE

        # if holding object:
        #   set conditions below
        #   return SUCCESS

        if self.counter < 1:
            self.blackboard.set("robot_moving", True)
            # TODO(kmuhlrad): replace with actually picking up object
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
        # TODO(kmuhlrad): figure this out

        # if holding object:
        #   set robot_moving to True
        #   return RUNNING

        # if some error occurs, return FAILURE

        # if not holding object:
        #   set conditions below
        #   return SUCCESS

        if self.counter < 1:
            self.blackboard.set("robot_moving", True)
            # TODO(kmuhlrad): replace with actually placing object
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
        # TODO(kmuhlrad): figure this out

        # if the door is not open:
        #   set robot_moving to True
        #   set gripper_empty to False
        #   return RUNNING

        # if some error occurs, return FAILURE

        # if the door is open:
        #   set the conditions below
        #   return SUCCESS

        if self.counter < 1:
            self.blackboard.set("robot_moving", True)
            # TODO(kmuhlrad): replace with actually opening the door
            self.blackboard.set("robot_holding", self.door)
            self.counter += 1
            return Status.RUNNING

        self.blackboard.set("{}_open".format(self.door), True)
        self.blackboard.set("robot_moving", False)
        self.blackboard.set("robot_holding", None)

        return Status.SUCCESS
