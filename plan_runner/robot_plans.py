import numpy as np
from pydrake.trajectories import PiecewisePolynomial
from pydrake.math import RollPitchYaw
from pydrake.common.eigen_geometry import Isometry3, Quaternion
from pydrake.multibody.plant import MultibodyPlant
from plan_utils import *

plan_type_strings = [
    "JointSpacePlan",
    "JointSpacePlanRelative",
    "IiwaTaskSpacePlan",
    "PlanarTaskSpacePlan",
    "PlanarHybridPositionForcePlan",
    "OpenLeftDoorPositionPlan",
    "OpenLeftDoorImpedancePlan",
    "JointSpacePlanGoToTarget",
]

PlanTypes = dict()
for plan_types_string in plan_type_strings:
    PlanTypes[plan_types_string] = plan_types_string


class PlanBase:
    def __init__(self,
                 type = None,
                 trajectory = None,):
        self.type = type
        self.traj = trajectory
        self.traj_d = None
        self.duration = None
        self.start_time = None
        if trajectory is not None:
            self.traj_d = trajectory.derivative(1)
            self.duration = trajectory.end_time()
            self.nq = self.traj.rows()

    def get_duration(self):
        return self.duration

    def CalcPositionCommand(
            self, q_iiwa, v_iiwa, tau_iiwa, t_plan, control_period, **kwargs):
        pass

    def CalcTorqueCommand(
            self, q_iiwa, v_iiwa, tau_iiwa, t_plan, control_period, **kwargs):
        pass


class JointSpacePlan(PlanBase):
    def __init__(self,
                 trajectory=None):
        PlanBase.__init__(self,
                          type=PlanTypes["JointSpacePlan"],
                          trajectory=trajectory)

    def CalcPositionCommand(
            self, q_iiwa, v_iiwa, tau_iiwa, t_plan, control_period, **kwargs):
        return self.traj.value(t_plan).flatten()

    def CalcTorqueCommand(
            self, q_iiwa, v_iiwa, tau_iiwa, t_plan, control_period, **kwargs):
        return np.zeros(self.nq)


'''
The robot goes to q_target from its configuration when this plan starts. 
'''
class JointSpacePlanGoToTarget(PlanBase):
    def __init__(self, duration, q_target):
        PlanBase.__init__(self,
                          type=PlanTypes["JointSpacePlanGoToTarget"],
                          trajectory=None)
        self.q_target = q_target
        self.duration = duration

    def UpdateTrajectory(self, q_start):
        self.traj = ConnectPointsWithCubicPolynomial(
            q_start, self.q_target, self.duration)
        self.traj_d = self.traj.derivative(1)

    def CalcPositionCommand(
            self, q_iiwa, v_iiwa, tau_iiwa, t_plan, control_period, **kwargs):
        if self.traj is None:
            self.UpdateTrajectory(q_start=q_iiwa)
        return self.traj.value(t_plan).flatten()

    def CalcTorqueCommand(
            self, q_iiwa, v_iiwa, tau_iiwa, t_plan, control_period, **kwargs):
        return np.zeros(len(q_iiwa))


'''
The robot goes from its configuration when this plan starts (q_current) by
delta_q to reach the final configuration (q_current + delta_q).
'''
class JointSpacePlanRelative(PlanBase):
    def __init__(self, duration, delta_q):
        PlanBase.__init__(self,
                          type=PlanTypes["JointSpacePlanRelative"],
                          trajectory=None)
        self.delta_q = delta_q
        self.duration = duration

    def UpdateTrajectory(self, q_start):
        self.traj = ConnectPointsWithCubicPolynomial(
            q_start, self.delta_q + q_start, self.duration)
        self.traj_d = self.traj.derivative(1)

    def CalcPositionCommand(
            self, q_iiwa, v_iiwa, tau_iiwa, t_plan, control_period, **kwargs):
        if self.traj is None:
            self.UpdateTrajectory(q_start=q_iiwa)
        return self.traj.value(t_plan).flatten()

    def CalcTorqueCommand(
            self, q_iiwa, v_iiwa, tau_iiwa, t_plan, control_period, **kwargs):
        return np.zeros(len(q_iiwa))


class JacobianBasedPlan(PlanBase):
    def __init__(self, plan_type, trajectory, Q_WT_ref, p_TQ, plant,
                 task_frame_name):
        PlanBase.__init__(self,
                          type=plan_type,
                          trajectory=trajectory)
        # Declare stuff related to rigid body computations.
        self.plant_robot = plant
        self.context_robot = self.plant_robot.CreateDefaultContext()
        self.task_frame = self.plant_robot.GetFrameByName(task_frame_name)
        self.nq = plant.num_positions()

        # Store EE rotation reference as a quaternion.
        self.Q_WT_ref = Q_WT_ref
        self.p_TQ = p_TQ

        # data members updated by CalcKinematics
        self.X_WT = None
        self.p_WQ = None
        self.Q_WT = None
        self.Jv_WTq = None

    def CalcKinematics(self, q_robot, v_robot):
        """
        @param q_robot: robot configuration.
        @param v_robot: robot velocity.
        Updates the following data members:
        - Jv_WTq: geometric jacboain of point Q in frame T. T is the "task frame",
            which can be the end effector frame or the body frame of the last link.
        - p_WQ: position of point Q in world frame.
        - Q_WT: orientation of frame T in world frame as a quaternion.
        - X_WT: pose of frame T relative to world frame.
        """
        # Update context
        self.context_robot.SetDiscreteState(np.hstack((q_robot, v_robot)))

        # Pose of frame L7 in world frame
        self.X_WT = self.plant_robot.CalcRelativeTransform(
            self.context_robot, frame_A=self.plant_robot.world_frame(),
            frame_B=self.task_frame).GetAsIsometry3()

        # Position of Q in world frame
        self.p_WQ = self.X_WT.multiply(self.p_TQ)

        # Orientation of frame T in world frame
        self.Q_WT = self.X_WT.quaternion()

        # calculate Geometric jacobian (6 by self.nq matrix) of point Q in frame T.
        self.Jv_WTq = self.plant_robot.CalcFrameGeometricJacobianExpressedInWorld(
            context=self.context_robot, frame_B=self.task_frame,
            p_BoFo_B=self.p_TQ)

    def CalcPositionError(self, t_plan):
        pass

    def CalcOrientationError(self, t_plan):
        # must be called after calling CalcKinematics
        Q_TTr = self.Q_WT.inverse().multiply(self.Q_WT_ref)
        return Q_TTr


class IiwaTaskSpacePlan(JacobianBasedPlan):
    def __init__(self,
                 plant,
                 xyz_traj,
                 Q_WT_ref,
                 task_frame_name,
                 p_TQ=np.zeros(3)):
        """
        :param xyz_traj (3-dimensional PiecewisePolynomial): desired
            trajectory of point Q in world frame,
            RELATIVE TO ITS POSITION AT THE BEGINNING OF THE PLAN.
        :param Q_WL7_ref (Quaternion): fixed orientation of the task frame T while Q
             tracks the given trajectory. If set to None, then only position of Q is
             controlled.
        :param p_TQ: the point in frame T that tracks xyz_traj. Its default value is the
            origin of T.
        :param plant: a MultibodyPlant that contains only the robot. The body
            frame of the robot's link 0 is coincident with the world frame.
        :param task_frame_name (string): name of the task frame T as defined in plant.
        """
        JacobianBasedPlan.__init__(
            self,
            plan_type=PlanTypes["IiwaTaskSpacePlan"],
            trajectory=xyz_traj,
            Q_WT_ref=Q_WT_ref,
            p_TQ=p_TQ,
            plant=plant,
            task_frame_name=task_frame_name)

        assert xyz_traj.rows() == 3
        self.xyz_offset = None
        self.q_iiwa_previous = np.zeros(self.nq)

        # final target position
        self.xyz_goal = None
        self.t_goal = xyz_traj.duration(0)

    def _UpdateXyzOffset(self, xyz_offset):
        assert len(xyz_offset) == 3
        self.xyz_offset = np.copy(xyz_offset)
        self.xyz_goal = self.xyz_offset + self.traj.value(self.t_goal).ravel()

    def CalcPositionError(self, t_plan):
        # must be called after calling CalcKinematics
        return self.traj.value(t_plan).ravel() + self.xyz_offset - self.p_WQ

    def CalcPositionReference(self, t_plan, q_iiwa, v_iiwa):
        self.CalcKinematics(q_iiwa, v_iiwa)

        if self.xyz_offset is None:
            self._UpdateXyzOffset(self.p_WQ)

        # position and orientation errors.
        err_xyz = self.CalcPositionError(t_plan)
        if self.Q_WT_ref is not None:
            Q_TTr = self.CalcOrientationError(t_plan)

        # first 3: angular velocity, last 3: translational velocity
        v_ee_desired = np.zeros(6)

        # Translation
        kp_translation = np.array([100., 100., 100])
        v_ee_desired[3:6] = kp_translation * err_xyz

        if self.Q_WT_ref is not None:
            # Rotation
            kp_rotation = np.array([20., 20, 20])
            v_ee_desired[0:3] = self.Q_WT.multiply(kp_rotation * Q_TTr.xyz())
            result = np.linalg.lstsq(self.Jv_WTq, v_ee_desired, rcond=None)
        else:
            # no rotation
            v_ee_desired = v_ee_desired[3:]
            result = np.linalg.lstsq(self.Jv_WTq[3:6], v_ee_desired, rcond=None)

        # this saturation could keep the robot from reaching the desired
        # configuration.
        return np.clip(result[0], -10, 10)

    def CalcPositionCommand(
            self, q_iiwa, v_iiwa, tau_iiwa, t_plan, control_period, **kwargs):

        if t_plan < self.duration:
            qdot_desired = self.CalcPositionReference(t_plan, q_iiwa, v_iiwa)
            self.q_iiwa_previous[:] = q_iiwa
            return q_iiwa + qdot_desired * control_period
        else:
            return self.q_iiwa_previous


## FROM OLD PLAN RUNNER, FIGURE OUT IF NEEDED
# X_EEa = GetEndEffectorWorldAlignedFrame()
# X_L7E = GetL7EeTransform()

# '''
# trajectory is a 3-dimensional PiecewisePolynomial. It describes the trajectory
#     of a point fixed to the ee frame in world frame, 
#     RELATIVE TO WHERE THE POINT IS AT THE BEGINNING OF THE PLAN.
# R_WE_ref is the fixed pose (RotationMatrix) of the end effector while it its origin moves along
#     the given trajectory. 
# the value of duration will be overwritten by trajectory.end_time() if trajectory
#     is a valid PiecewisePolynomial.
# The axes of L7 and Ea (End Effector World Frane Aligned) are actually aligned. 
#     So R_EaL7 is identity. 
# p_EQ: the point in frame L7 that tracks trajectory. Its default value is the origin of L7.
# '''
# class IiwaTaskSpacePlan(PlanBase):
#     def __init__(self,
#                  duration=None,
#                  trajectory=None,
#                  R_WEa_ref=None,
#                  p_EQ=np.zeros(3)):
#         self.xyz_offset = None

#         self.plant_iiwa = station.get_controller_plant()
#         self.context_iiwa = self.plant_iiwa.CreateDefaultContext()
#         self.l7_frame = self.plant_iiwa.GetFrameByName('iiwa_link_7')

#         PlanBase.__init__(self,
#                           type=PlanTypes["IiwaTaskSpacePlan"],
#                           trajectory=trajectory)
#         self.duration = duration

#         self.q_iiwa_previous = np.zeros(7)
#         if trajectory is not None and R_WEa_ref is not None:
#             assert trajectory.rows() == 3
#             assert np.allclose(self.duration, trajectory.end_time())
#             self.traj = trajectory
#             self.p_L7Q = X_L7E.multiply(p_EQ)

#             # L7 and Ea are already aligned.
#             # This is the identity matrix.
#             self.R_EaL7 = X_EEa.inverse().rotation().dot(X_L7E.inverse().rotation().matrix())

#             # Store EE rotation reference as a quaternion.
#             self.Q_WL7_ref = R_WEa_ref.ToQuaternion()
#             self.p_EQ = p_EQ

#     def UpdateXyzOffset(self, xyz_offset):
#         assert len(xyz_offset) == 3
#         self.xyz_offset = np.copy(xyz_offset)

#     def CalcXyzReference(self, t_plan):
#         return self.traj.value(t_plan).flatten() + self.xyz_offset

#     def CalcPositionCommand(self, t_plan, q_iiwa, control_period):
#         if t_plan < self.duration * 2:
#             # Update context
#             x_iiwa_mutable = \
#                 self.plant_iiwa.GetMutablePositionsAndVelocities(self.context_iiwa)
#             x_iiwa_mutable[:7] = q_iiwa

#             # calculate Geometric jacobian (6 by 7 matrix) of point Q in frame L7.
#             Jv_WL7q = self.plant_iiwa.CalcFrameGeometricJacobianExpressedInWorld(
#                 context=self.context_iiwa, frame_B=self.l7_frame,
#                 p_BoFo_B=self.p_L7Q)

#             # Pose of frame L7 in world frame
#             X_WL7 = self.plant_iiwa.CalcRelativeTransform(
#                 self.context_iiwa, frame_A=self.plant_iiwa.world_frame(),
#                 frame_B=self.l7_frame)

#             # position and orientation errors.
#             p_WQ = X_WL7.multiply(self.p_L7Q)
#             err_xyz = self.CalcXyzReference(t_plan) - p_WQ
#             Q_WL7 = X_WL7.quaternion()
#             Q_L7L7r = Q_WL7.inverse().multiply(self.Q_WL7_ref)

#             # first 3: angular velocity, last 3: translational velocity
#             v_ee_desired = np.zeros(6)

#             # Translation
#             kp_translation = np.array([100., 100., 200])
#             v_ee_desired[3:6] = kp_translation * err_xyz

#             # Rotation
#             kp_rotation = np.array([20., 20, 20])
#             v_ee_desired[0:3] = Q_WL7.multiply(kp_rotation * Q_L7L7r.xyz())

#             result = np.linalg.lstsq(Jv_WL7q, v_ee_desired, rcond=None)
#             qdot_desired = np.clip(result[0], -2, 2)

#             self.q_iiwa_previous[:] = q_iiwa
#             return q_iiwa + qdot_desired * control_period
#         else:
#             return self.q_iiwa_previous


# #------------------------------------------------------------------------------
# #TODO: The following plan types are not supported yet.
# '''
# trajectory is a 3-dimensional PiecewisePolynomial.
# trajectory[0] is the x-position of the gripper in world frame.
# trajectory[1] is the y-position of the gripper in world frame.
# trajectory[2] is the angle between the ee frame and the world frame. 
# '''
# class PlanarTaskSpacePlan(PlanBase):
#     def __init__(self,
#                  trajectory=None):
#         PlanBase.__init__(self,
#                           type=PlanTypes[2],
#                           trajectory=trajectory)


# '''
# The end effector of a planar robot has three degrees of freedom:
#     translation along y, z and rotation about x. 
# x_ee_traj is a k-dimensional (k<=3) PiecewisePolynomial that describes the desired trajectory
#     of the position-controlled DOFs.
# f_ee_traj is a (3-k)-dimensional PiecewisePolynomial that describes the desired trajecotry
#     of the force-controlled DOFs (applied by the robot on the world).
# The "Task frame" or "constrained frame" (as in Raibert and Craig, 1981) is aligned 
#     with the world frame, the origin of the task frame is coincident with the body 
#     frame of the end effector.
# selector is a (3,) numpy array that whether the i-th DOF is force-controlled (selector[i] == 1)
#     or position-controlled(selector[i] == 0).
# '''
# class PlanarHybridPositionForcePlan(PlanBase):
#     def __init__(self,
#                  x_ee_traj=None,
#                  f_ee_traj=None,
#                  selector=None):
#         PlanBase.__init__(self,
#                           type=PlanTypes[3],
#                           trajectory=x_ee_traj)
#         assert np.isclose(x_ee_traj.end_time(), f_ee_traj.end_time())
#         assert x_ee_traj.rows() + f_ee_traj.rows() == 3
#         assert selector.shape == (3,)
#         self.f_ee_traj = f_ee_traj
#         self.selector = selector

#         # using notations from De Luca's notes on hybrid force/motion control
#         k = x_ee_traj.rows()
#         self.T = np.zeros((3, k))
#         j = 0
#         for i in range(3):
#             if selector[i] == 0:
#                 self.T[i,j] = 1
#                 j += 1

#         j I= 0
#         self.Y = np.zeros((3, 3-k))
#         for i in range(3):
#             if selector[i] == 1:
#                 self.Y[i,j] = 1
#                 j+= 1
