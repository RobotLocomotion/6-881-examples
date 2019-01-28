import numpy as np
from pydrake.trajectories import PiecewisePolynomial
from pydrake.math import RollPitchYaw
from pydrake.common.eigen_geometry import Isometry3, Quaternion
from pydrake.examples.manipulation_station import ManipulationStation
from pydrake.multibody.multibody_tree.multibody_plant import MultibodyPlant
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

    def get_duration(self):
        return self.duration

    def CalcPositionCommand(self, q_iiwa, v_iiwa, tau_iiwa, t_plan, control_period):
        pass

    def CalcTorqueCommand(self, q_iiwa, v_iiwa, tau_iiwa, t_plan, control_period):
        return np.zeros(7)


class JointSpacePlan(PlanBase):
    def __init__(self,
                 trajectory=None):
        PlanBase.__init__(self,
                          type=PlanTypes["JointSpacePlan"],
                          trajectory=trajectory)

    def CalcPositionCommand(self, q_iiwa, v_iiwa, tau_iiwa, t_plan, control_period):
        return self.traj.value(t_plan).flatten()

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

    def CalcPositionCommand(self, q_iiwa, v_iiwa, tau_iiwa, t_plan, control_period):
        if self.traj is None:
            self.UpdateTrajectory(q_start=q_iiwa)
        return self.traj.value(t_plan).flatten()


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

    def CalcPositionCommand(self, q_iiwa, v_iiwa, tau_iiwa, t_plan, control_period):
        if self.traj is None:
            self.UpdateTrajectory(q_start=q_iiwa)
        return self.traj.value(t_plan).flatten()


class JacobianBasedPlan(PlanBase):
    def __init__(self, plan_type, trajectory, Q_WL7_ref, p_L7Q):
        self.plant_iiwa = station.get_controller_plant()
        self.tree_iiwa =self.plant_iiwa.tree()
        self.context_iiwa = self.plant_iiwa.CreateDefaultContext()
        self.l7_frame = self.plant_iiwa.GetFrameByName('iiwa_link_7')

        # Store EE rotation reference as a quaternion.
        self.Q_WL7_ref = Q_WL7_ref
        self.p_L7Q = p_L7Q

        PlanBase.__init__(self,
                          type=plan_type,
                          trajectory=trajectory)

        # data members updated by CalcKinematics
        self.X_WL7 = None
        self.p_WQ = None
        self.Q_WL7=None
        self.Jv_WL7q = None

    def CalcKinematics(self, q_iiwa, v_iiwa):
        """
        @param q_iiwa: robot configuration.
        @param v_iiwa: robot velocity.
        Updates the following data members:
        - Jv_WL7q: geometric jacboain of point Q in frame L7.
        - p_WQ: position of point Q in world frame.
        - Q_WL7: orientation of frame L7 in world frame as a quaternion.
        - X_WL7: pose of frame L7 relative to world frame.
        """
        x_iiwa_mutable = \
            self.tree_iiwa.GetMutablePositionsAndVelocities(self.context_iiwa)
        x_iiwa_mutable[:7] = q_iiwa
        x_iiwa_mutable[7:] = v_iiwa

        # Pose of frame L7 in world frame
        self.X_WL7 = self.tree_iiwa.CalcRelativeTransform(
            self.context_iiwa, frame_A=self.plant_iiwa.world_frame(),
            frame_B=self.l7_frame)

        # Position of Q in world frame
        self.p_WQ = self.X_WL7.multiply(self.p_L7Q)

        # Orientation of Q in world frame
        self.Q_WL7 = self.X_WL7.quaternion()

        # calculate Geometric jacobian (6 by 7 matrix) of point Q in frame L7.
        self.Jv_WL7q = self.tree_iiwa.CalcFrameGeometricJacobianExpressedInWorld(
            context=self.context_iiwa, frame_B=self.l7_frame,
            p_BoFo_B=self.p_L7Q)

    def CalcPositionError(self, t_plan):
        pass

    def CalcOrientationError(self, t_plan):
        # must be called after calling CalcKinematics
        Q_L7L7r = self.Q_WL7.inverse().multiply(self.Q_WL7_ref)
        return Q_L7L7r

class IiwaTaskSpacePlan(JacobianBasedPlan):
    def __init__(self,
                 xyz_traj,
                 Q_WL7_ref,
                 p_L7Q=np.zeros(3)):
        """
        @param xyz_traj (3-dimensional PiecewisePolynomial): desired
            trajectory ]of point Q in world frame,
            RELATIVE TO ITS POSITION AT THE BEGINNING OF THE PLAN.
        @param Q_WL7_ref (Quaternion): fixed orientation of the end effector while it its
            origin moves along the given trajectory
        @param p_L7Q: the point in frame L7 that tracks xyz_traj. Its default value is the
            origin of L7.
        """
        assert xyz_traj.rows() == 3
        self.xyz_offset = None
        self.q_iiwa_previous = np.zeros(7)

        JacobianBasedPlan.__init__(
            self,
            plan_type=PlanTypes["IiwaTaskSpacePlan"],
            trajectory=xyz_traj,
            Q_WL7_ref=Q_WL7_ref,
            p_L7Q=p_L7Q)

    def UpdateXyzOffset(self, xyz_offset):
        assert len(xyz_offset) == 3
        self.xyz_offset = np.copy(xyz_offset)

    def CalcPositionError(self, t_plan):
        # must be called after calling CalcKinematics
        return self.traj.value(t_plan).flatten() + self.xyz_offset - self.p_WQ

    def CalcPositionCommand(self, q_iiwa, v_iiwa, tau_iiwa, t_plan, control_period):
        self.CalcKinematics(q_iiwa, v_iiwa)

        if self.xyz_offset is None:
            self.UpdateXyzOffset(self.p_WQ)

        if t_plan < self.duration:
            # position and orientation errors.
            err_xyz = self.CalcPositionError(t_plan)
            Q_L7L7r = self.CalcOrientationError(t_plan)

            # first 3: angular velocity, last 3: translational velocity
            v_ee_desired = np.zeros(6)

            # Translation
            kp_translation = np.array([100., 100., 200])
            v_ee_desired[3:6] = kp_translation * err_xyz

            # Rotation
            kp_rotation = np.array([20., 20, 20])
            v_ee_desired[0:3] = self.Q_WL7.multiply(kp_rotation * Q_L7L7r.xyz())

            result = np.linalg.lstsq(self.Jv_WL7q, v_ee_desired, rcond=None)
            qdot_desired = np.clip(result[0], -1, 1)

            self.q_iiwa_previous[:] = q_iiwa
            return q_iiwa + qdot_desired * control_period
        else:
            return self.q_iiwa_previous


#------------------------------------------------------------------------------
#TODO: The following plan types are not supported yet.
'''
trajectory is a 3-dimensional PiecewisePolynomial.
trajectory[0] is the x-position of the gripper in world frame.
trajectory[1] is the y-position of the gripper in world frame.
trajectory[2] is the angle between the ee frame and the world frame. 
'''
class PlanarTaskSpacePlan(PlanBase):
    def __init__(self,
                 trajectory=None):
        PlanBase.__init__(self,
                          type=PlanTypes[2],
                          trajectory=trajectory)


'''
The end effector of a planar robot has three degrees of freedom:
    translation along y, z and rotation about x. 
x_ee_traj is a k-dimensional (k<=3) PiecewisePolynomial that describes the desired trajectory
    of the position-controlled DOFs.
f_ee_traj is a (3-k)-dimensional PiecewisePolynomial that describes the desired trajecotry
    of the force-controlled DOFs (applied by the robot on the world).
The "Task frame" or "constrained frame" (as in Raibert and Craig, 1981) is aligned 
    with the world frame, the origin of the task frame is coincident with the body 
    frame of the end effector.
selector is a (3,) numpy array that whether the i-th DOF is force-controlled (selector[i] == 1)
    or position-controlled(selector[i] == 0).
'''
class PlanarHybridPositionForcePlan(PlanBase):
    def __init__(self,
                 x_ee_traj=None,
                 f_ee_traj=None,
                 selector=None):
        PlanBase.__init__(self,
                          type=PlanTypes[3],
                          trajectory=x_ee_traj)
        assert np.isclose(x_ee_traj.end_time(), f_ee_traj.end_time())
        assert x_ee_traj.rows() + f_ee_traj.rows() == 3
        assert selector.shape == (3,)
        self.f_ee_traj = f_ee_traj
        self.selector = selector

        # using notations from De Luca's notes on hybrid force/motion control
        k = x_ee_traj.rows()
        self.T = np.zeros((3, k))
        j = 0
        for i in range(3):
            if selector[i] == 0:
                self.T[i,j] = 1
                j += 1

        j = 0
        self.Y = np.zeros((3, 3-k))
        for i in range(3):
            if selector[i] == 1:
                self.Y[i,j] = 1
                j+= 1
