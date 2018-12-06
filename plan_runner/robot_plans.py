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


class JointSpacePlan(PlanBase):
    def __init__(self,
                 trajectory=None):
        PlanBase.__init__(self,
                          type=PlanTypes["JointSpacePlan"],
                          trajectory=trajectory)

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


X_EEa = GetEndEffectorWorldAlignedFrame()
X_L7E = GetL7EeTransform()

'''
trajectory is a 3-dimensional PiecewisePolynomial. It describes the trajectory
    of a point fixed to the ee frame in world frame, 
    RELATIVE TO WHERE THE POINT IS AT THE BEGINNING OF THE PLAN.
R_WE_ref is the fixed pose (RotationMatrix) of the end effector while it its origin moves along
    the given trajectory. 
the value of duration will be overwritten by trajectory.end_time() if trajectory
    is a valid PiecewisePolynomial.
The axes of L7 and Ea (End Effector World Frane Aligned) are actually aligned. 
    So R_EaL7 is identity. 
p_EQ: the point in frame L7 that tracks trajectory. Its default value is the origin of L7.
'''
class IiwaTaskSpacePlan(PlanBase):
    def __init__(self,
                 duration=None,
                 trajectory=None,
                 R_WEa_ref=None,
                 p_EQ=np.zeros(3)):
        self.xyz_offset = None

        self.plant_iiwa = station.get_controller_plant()
        self.tree_iiwa =self.plant_iiwa.tree()
        self.context_iiwa = self.plant_iiwa.CreateDefaultContext()
        self.l7_frame = self.plant_iiwa.GetFrameByName('iiwa_link_7')

        PlanBase.__init__(self,
                          type=PlanTypes["IiwaTaskSpacePlan"],
                          trajectory=trajectory)
        self.duration = duration

        self.q_iiwa_previous = np.zeros(7)
        if trajectory is not None and R_WEa_ref is not None:
            assert trajectory.rows() == 3
            assert np.allclose(self.duration, trajectory.end_time())
            self.traj = trajectory
            self.p_L7Q = X_L7E.multiply(p_EQ)

            # L7 and Ea are already aligned.
            # This is the identity matrix.
            self.R_EaL7 = X_EEa.inverse().rotation().dot(X_L7E.inverse().rotation())

            # Store EE rotation reference as a quaternion.
            self.Q_WL7_ref = R_WEa_ref.ToQuaternion()
            self.p_EQ = p_EQ

    def UpdateXyzOffset(self, xyz_offset):
        assert len(xyz_offset) == 3
        self.xyz_offset = np.copy(xyz_offset)

    def CalcXyzReference(self, t_plan):
        return self.traj.value(t_plan).flatten() + self.xyz_offset

    def CalcPositionCommand(self, t_plan, q_iiwa, control_period):
        if t_plan < self.duration * 2:
            # Update context
            x_iiwa_mutable = \
                self.tree_iiwa.GetMutablePositionsAndVelocities(self.context_iiwa)
            x_iiwa_mutable[:7] = q_iiwa

            # calculate Geometric jacobian (6 by 7 matrix) of point Q in frame L7.
            Jv_WL7q = self.tree_iiwa.CalcFrameGeometricJacobianExpressedInWorld(
                context=self.context_iiwa, frame_B=self.l7_frame,
                p_BoFo_B=self.p_L7Q)

            # Pose of frame L7 in world frame
            X_WL7 = self.tree_iiwa.CalcRelativeTransform(
                self.context_iiwa, frame_A=self.plant_iiwa.world_frame(),
                frame_B=self.l7_frame)

            # position and orientation errors.
            p_WQ = X_WL7.multiply(self.p_L7Q)
            err_xyz = self.CalcXyzReference(t_plan) - p_WQ
            Q_WL7 = X_WL7.quaternion()
            Q_L7L7r = Q_WL7.inverse().multiply(self.Q_WL7_ref)

            # first 3: angular velocity, last 3: translational velocity
            v_ee_desired = np.zeros(6)

            # Translation
            kp_translation = np.array([100., 100., 200])
            v_ee_desired[3:6] = kp_translation * err_xyz

            # Rotation
            kp_rotation = np.array([20., 20, 20])
            v_ee_desired[0:3] = Q_WL7.multiply(kp_rotation * Q_L7L7r.xyz())

            result = np.linalg.lstsq(Jv_WL7q, v_ee_desired, rcond=None)
            qdot_desired = np.clip(result[0], -2, 2)

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
