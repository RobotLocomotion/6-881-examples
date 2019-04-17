from py_trees.behaviour import Behaviour
from py_trees.common import Status

from problem_description import Task

########################## Conditions ##########################


class CanMove(Behaviour):
    def __init__(self, task, name="CanMove"):
        """
        Checks if the robot is allowed to move. Returns SUCCESS if true, else
        returns FAILURE.

        @param task Task. A description of the problem.
        @param name str. The name of the BT node.
        """
        super(CanMove, self).__init__(name)
        self.task = task

    def update(self):
        if self.task.can_move[self.task.robot]:
            self.feedback_message = "{} can move".format(self.task.robot)
            return Status.SUCCESS
        self.feedback_message = "{} can't move".format(self.task.robot)
        return Status.FAILURE


class HandEmpty(Behaviour):
    def __init__(self, task, name="HandEmpty"):
        """
        Checks if the robot is holding anything. Returns FAILURE if true, else
        returns SUCCESS.

        @param task Task. A description of the problem.
        @param name str. The name of the BT node.
        """
        super(HandEmpty, self).__init__(name)
        self.task = task

    def update(self):
        if self.task.hand_empty[self.task.robot]:
            self.feedback_message = "{} is not holding anything".format(
                self.task.robot)
            return Status.SUCCESS
        self.feedback_message = "{} is holding something".format(
            self.task.robot)
        return Status.FAILURE


class Holding(Behaviour):
    def __init__(self, task, obj, name="Holding"):
        """
        Checks if the robot is holding the specified object. Returns SUCCESS if
        true, else returns FAILURE.

        @param task Task. A description of the problem.
        @param obj str. The name of the object of interest, such as "soup".
        @param name str. The name of the BT node.
        """
        super(Holding, self).__init__(name)
        self.task = task
        self.obj = obj

    def update(self):
        if self.task.holding[self.task.robot][self.obj]:
            self.feedback_message = "{} is holding {}".format(
                self.task.robot, self.obj)
            return Status.SUCCESS
        self.feedback_message = "{} is not holding {}".format(
            self.task.robot, self.obj)
        return Status.FAILURE


class AtConf(Behaviour):
    def __init__(self, task, agent, conf, name="AtConf"):
        """
        Checks if the given agent is at the given configuration. The agent
        is either a robot or a door. Returns SUCCESS if true, else returns
        FAILURE.

        @param task Task. A description of the problem.
        @param agent str. The name of the agent, such as "iiwa" or "left_door".
        @param conf Conf. The desired configuration.
        @param name str. The name of the BT node.
        """
        super(AtConf, self).__init__(name)
        self.task = task
        self.agent = agent
        self.conf = conf

    def update(self):
        if self.task.at_conf[self.agent] == self.conf:
            self.feedback_message = "{} is at configuration {}".format(
                self.agent, self.conf)
            return Status.SUCCESS
        self.feedback_message = "{} is not at configuration {}".format(
            self.agent, self.conf)
        return Status.FAILURE


class AtPose(Behaviour):
    def __init__(self, task, obj, pose, name="AtPose"):
        """
        Checks if the given object is at the given pose. Returns SUCCESS if
        true, else returns FAILURE.

        @param task Task. A description of the problem.
        @param obj str. The name of the object, such as "soup".
        @param pose Pose. The desired pose.
        @param name str. The name of the BT node.
        """
        super(AtPose, self).__init__(name)
        self.task = task
        self.obj = obj
        self.pose = pose

    def update(self):
        if self.task.at_pose[self.obj] == self.pose:
            self.feedback_message = "{} is at pose {}".format(
                self.obj, self.pose)
            return Status.SUCCESS
        self.feedback_message = "{} is not at pose {}".format(
            self.obj, self.pose)
        return Status.FAILURE


class AtGrasp(Behaviour):
    def __init__(self, task, obj, grasp, name="AtGrasp"):
        """
        Checks if the robot is holding the given object with the given grasp.
        Returns SUCCESS if true, else returns FAILURE.

        @param task Task. A description of the problem.
        @param obj str. The name of the object, such as "soup".
        # TODO(kmuhlrad): is Grasp actually an object? Is it a pose?
        @param grasp Grasp. The desired grasp.
        @param name str. The name of the BT node.
        """
        super(AtGrasp, self).__init__(name)
        self.task = task
        self.obj = obj
        self.grasp = grasp

    def update(self):
        if self.task.at_grasp[self.task.robot][self.obj] == self.grasp:
            self.feedback_message = "{} is holding {} with grasp {}".format(
                self.task.robot, self.obj, self.grasp)
            return Status.SUCCESS
        self.feedback_message = "{} is not holding {} with grasp {}".format(
            self.task.robot, self.obj, self.grasp)
        return Status.FAILURE


# TODO(kmuhlrad): finish implementing
class UnsafeTraj(Behaviour):
    def __init__(self, task, traj, name="UnsafeTraj"):
        """
        Checks if the trajectory is not unsafe for the robot to follow. Returns
        SUCCESS if true, else returns FAILURE.

        @param task Task. A description of the problem.
        @param traj Trajectory. The trajectory to check.
        @param name str. The name of the BT node.
        """
        super(UnsafeTraj, self).__init__(name)
        self.task = task
        self.traj = traj

    def update(self):
        # TODO(kmuhlrad): figure this out
        return Status.FAILURE


class On(Behaviour):
    def __init__(self, task, obj, surface, name="On"):
        """
        Checks if the given object is on the given surface. Returns SUCCESS if
        true, else returns FAILURE.

        @param task Task. A description of the problem.
        @param obj str. The name of the object, such as "soup".
        @param surace str. The name of the desired surface, such as "bottom".
        @param name str. The name of the BT node.
        """
        super(On, self).__init__(name)
        self.task = task
        self.obj = obj
        self.surface = surface

    def update(self):
        if self.task.on[self.obj] == self.surface:
            self.feedback_message = "{} is on {}".format(
                self.obj, self.surface)
            return Status.SUCCESS
        self.feedback_message = "{} is not on {}".format(
            self.obj, self.surface)
        return Status.FAILURE


########################## Actions ##########################


# TODO(kmuhlrad): finish implementations of things and figure out if all
# params are actually needed for the actions and not just condition checking


class Move(Behaviour):
    def __init__(self, task, conf1, conf2, traj, name="Move"):
        """
        Move the robot from conf1 to conf2 along the specified trajectory. When
        the robot finishes the trajectory and is at conf2, the node returns
        SUCCESS. While following the path, the node returns RUNNING. If the
        robot is unable to complete the trajectory, it will stop and the node
        will return FAILURE.

        @param task Task. A description of the problem.
        @param conf1 Conf. The start configuration of the robot.
        @param conf2 Conf. The end configuration of the robot.
        @param traj Trajectory. The trajectory for the robot to follow.
        @param name str. The name of the BT node.
        """
        super(Move, self).__init__(name)
        self.task = task
        self.conf1 = conf1
        self.conf2 = conf2
        self.traj = traj

    def update(self):
        # TODO(kmuhlrad): figure this out


        # check pose of robot
        # if at conf2, return SUCCESS
        # update conditions
        #   AtConf(robot, conf2)
        #   !CanMove(robot)

        # else start/continue running and return RUNNING

        # if some error occurs, return FAILURE

        self.task.at_conf[self.task.robot] = self.conf2
        self.can_move[self.task.robot] = False

        return Status.SUCCESS


class Pick(Behaviour):
    def __init__(self, task, obj, pose, grasp, conf, traj, name="Pick"):
        """
        Pick up the specified object by following the given trajectory. When
        the robot finishes the trajectory and has grasped the object, the node
        returns SUCCESS. While following the trajectory, the node returns
        RUNNING. If the robot is unable to complete the trajectory or pick up
        the object, it will stop and the node will return FAILURE.

        @param task Task. A description of the problem.
        @param obj str. The name of the object to pick, such as "soup".
        @param pose Pose. The start pose of the object.
        @param grasp Grasp. The grasp to pick up the object.
        @param conf Conf. The start configuration of the robot.
        @param traj Trajectory. The trajectory for the robot to follow.
        @param name str. The name of the BT node.
        """
        super(Pick, self).__init__(name)
        self.task = task
        self.obj = obj
        self.pose = pose
        self.grasp = grasp
        self.conf = conf
        self.traj = traj

    def update(self):
        # TODO(kmuhlrad): figure this out

        # check trajectory status
        # if complete, return SUCCESS
        # update conditions:
        #   !AtPose(obj, pose)
        #   !HandEmpty(robot)
        #   AtGrasp(robot, obj, grasp)
        #   CanMove(robot)

        # else continue running and return RUNNING

        # if some error occurs, return FAILURE

        self.task.can_move[self.task.robot] = True
        self.task.at_grasp[self.task.robot][self.obj] = self.grasp
        self.task.hand_empty[self.task.robot] = False
        self.task.at_pose[self.obj] = False

        return Status.SUCCESS


class Place(Behaviour):
    def __init__(self, task, obj, pose, grasp, conf, traj, name="Place"):
        """
        Place up the specified object by following the given trajectory. When
        the robot finishes the trajectory and has released the object, the node
        returns SUCCESS. While following the trajectory, the node returns
        RUNNING. If the robot is unable to complete the trajectory or place
        the object, it will stop and the node will return FAILURE.

        @param task Task. A description of the problem.
        @param obj str. The name of the object to place, such as "soup".
        @param pose Pose. The end pose of the object.
        @param grasp Grasp. The start grasp of the object.
        @param conf Conf. The start configuration of the robot.
        @param traj Trajectory. The trajectory for the robot to follow.
        @param name str. The name of the BT node.
        """
        super(Place, self).__init__(name)
        self.task = task
        self.obj = obj
        self.pose = pose
        self.grasp = grasp
        self.conf = conf
        self.traj = traj

    def update(self):
        # TODO(kmuhlrad): figure this out

        # check trajectory and gripper status
        # if complete, return SUCCESS
        # update conditions:
        #   AtPose(obj, pose)
        #   HandEmpty(robot)
        #   !AtGrasp(robot, obj, grasp)
        #   CanMove(robot)

        # else continue running and return RUNNING

        # if some error occurs, return FAILURE

        self.task.at_pose[self.obj] = self.pose
        self.task.hand_empty[self.task.robot] = True
        self.task.at_grasp[self.task.robot][self.obj] = False
        self.task.can_move[self.robot] = True

        return Status.SUCCESS


class Pull(Behaviour):
    def __init__(self, task, door, r_conf1, r_conf2, d_conf1, d_conf2, traj,
                 name="Pull"):
        """
        Pull open the specified door by following the specified trajectory.
        When the robot finishes the trajectory, is at r_conf2, and the door is
        at d_conf2, the node returns SUCCESS. While following the trajectory,
        the node returns RUNNING. If the robot is unable to complete the
        trajectory or open the door, it will stop and the node will return
        FAILURE.

        @param task Task. A description of the problem.
        @param door str. The name of the door to open, such as "left_door".
        @param r_conf1 Conf. The start configuration of the robot.
        @param r_conf2 Conf. The end configuration of the robot.
        @param d_conf1 Conf. The start configuration of the door.
        @param d_conf1 Conf. The end configuration of the door.
        @param traj Trajectory. The trajectory for the robot to follow.
        @param name str. The name of the BT node.
        """
        super(Pull, self).__init__(name)
        self.task = task
        self.door = door
        self.r_conf1 = r_conf1
        self.r_conf2 = r_conf2
        self.d_conf1 = d_conf1
        self.d_conf2 = d_conf2
        self.traj = traj

    def update(self):
        # TODO(kmuhlrad): figure this out

        # check trajectory and configurations
        # if complete, return SUCCESS
        # update conditions:
        #   AtConf(robot, r_conf2)
        #   AtConf(door, d_conf2)
        #   CanMove(robot)

        # else continue running and return RUNNING

        # if some error occurs, return FAILURE

        self.task.at_conf[self.task.robot] = self.r_conf2
        self.task.at_conf[self.door] = self.d_conf2
        self.task.can_move[self.task.robot] = True
        
        return Status.SUCCESS
