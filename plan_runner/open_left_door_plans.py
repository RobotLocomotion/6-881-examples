from robot_plans import *

#--------------------------------------------------------------------------------------------------
#------------------------------------ open left door related constants       ----------------
# Define positions and transforms used in various tasks, e.g. opening left door.
# home position of point Q in world frame.
p_WQ_home = np.array([0.5, 0, 0.41])

# L: frame the cupboard left door, whose origin is at the center of the door body.
p_WL = np.array([0.7477, 0.1445, 0.4148]) #+ [-0.1, 0, 0]
# center of the left hinge of the door in frame L and W
p_LC_left_hinge = np.array([0.008, 0.1395, 0])
p_WC_left_hinge = p_WL + p_LC_left_hinge

# center of handle in frame L and W
p_LC_handle = np.array([-0.033, -0.1245, 0])
p_WC_handle = p_WL + p_LC_handle

# distance between the hinge center and the handle center
p_handle_2_hinge = p_LC_handle - p_LC_left_hinge
r_handle = np.linalg.norm(p_handle_2_hinge)

# angle between the world y axis and the line connecting the hinge cneter to the
# handle center when the left door is fully closed (left_hinge_angle = 0).
theta0_hinge = np.arctan2(np.abs(p_handle_2_hinge[0]),
                          np.abs(p_handle_2_hinge[1]))

# position of point Q in EE frame.  Point Q is fixed in the EE frame.
p_EQ = GetEndEffectorWorldAlignedFrame().multiply(np.array([0., 0., 0.090]))

# orientation of end effector aligned frame
R_WEa_ref = RollPitchYaw(0, np.pi / 180 * 135, 0).ToRotationMatrix()

q_home = np.array([0, -0.2136, 0, -2.094, 0, 0.463, 0])
q_pre_swing = np.array([2.44, 16.72, -17.43, -89.56, 47.30, 63.53, -83.77])*np.pi/180
q_post_swing = np.array([20.0, 16.72, -17.43, -89.56, 47.30, 63.53, -83.77])*np.pi/180



class OpenLeftDoorPlan(PlanBase):
    def __init__(self, angle_start, angle_end=np.pi/4, duration=10.0, type=None):
        angle_traj = ConnectPointsWithCubicPolynomial(
            [angle_start], [angle_end], duration)
        # Axes of Ea and L7 are aligned.
        self.Q_WL7_ref = R_WEa_ref.ToQuaternion()

        PlanBase.__init__(self,
                          type=type,
                          trajectory=angle_traj)

    def CalcKinematics(self, l7_frame, world_frame, tree_iiwa, context_iiwa, t_plan):
        """
        @param X_L7E: transformation from frame E (end effector) to frame L7.
        @param l7_frame: A BodyFrame object of frame L7.
        @param world_frame: A BodyFrame object of the world frame.
        @param tree_iiwa: A MultibodyTree object of the robot.
        @param context_iiwa: A Context object that describes the current state of the robot.
        @param t_plan: time passed since the beginning of this Plan, expressed in seconds.
        @return: Jv_WL7q: geometric jacboain of point Q in frame L7.
                p_HrQ: position of point Q relative to frame Hr.
        """
        # calculate Geometric jacobian (6 by 7 matrix) of point Q in frame L7.
        p_L7Q = X_L7E.multiply(p_EQ)
        Jv_WL7q = tree_iiwa.CalcFrameGeometricJacobianExpressedInWorld(
            context=context_iiwa, frame_B=l7_frame,
            p_BoFo_B=p_L7Q)

        # Translation
        # Hr: handle reference frame
        # p_HrQ: position of point Q relative to frame Hr.
        # X_WHr: transformation from Hr to world frame W.
        X_WHr = Isometry3()
        handle_angle_ref = self.traj.value(t_plan).flatten()
        X_WHr.set_rotation(
            RollPitchYaw(0, 0, -handle_angle_ref).ToRotationMatrix().matrix())
        X_WHr.set_translation(
            p_WC_left_hinge +
            [-r_handle * np.sin(handle_angle_ref),
             -r_handle * np.cos(handle_angle_ref), 0])
        X_HrW = X_WHr.inverse()

        X_WL7 = tree_iiwa.CalcRelativeTransform(
            context_iiwa, frame_A=world_frame,
            frame_B=l7_frame)

        p_WQ = X_WL7.multiply(p_L7Q)
        p_HrQ = X_HrW.multiply(p_WQ)

        Q_WL7 = X_WL7.quaternion()
        Q_L7L7r = Q_WL7.inverse().multiply(self.Q_WL7_ref)

        return Jv_WL7q, p_HrQ, Q_L7L7r, Q_WL7


class OpenLeftDoorPositionPlan(OpenLeftDoorPlan):
    def __init__(self, angle_start, angle_end=np.pi/4, duration=10.0):
        OpenLeftDoorPlan.__init__(
            self,
            angle_start=angle_start,
            angle_end=angle_end,
            duration=duration,
            type=PlanTypes["OpenLeftDoorPositionPlan"])
        self.q_iiwa_previous = np.zeros(7)

    def CalcPositionCommand(self, t_plan, q_iiwa, Jv_WL7q, p_HrQ, Q_L7L7r, Q_WL7, control_period):
        """
        @param t_plan: t_plan: time passed since the beginning of this Plan, expressed in seconds.
        @param q_iiwa: current configuration of the robot.
        @param Jv_WL7q: geometric jacboain of point Q in frame L7.
        @param p_HrQ: position of point Q relative to frame Hr.
        @param control_period: the amount of time between consecutive command updates.
        @return: position command to the robot.
        """
        if t_plan < self.duration:
            # first 3: angular velocity, last 3: translational velocity
            v_ee_desired = np.zeros(6)

            # Translation
            kp_x = 100*np.clip(t_plan/(0.2*self.duration), 0, 1)
            kp_translation = np.array([kp_x, 2., 100])/4
            v_ee_desired[3:6] = -kp_translation * p_HrQ

            # Rotation
            kp_rotation = np.array([2.5, 10, 10])*4
            v_ee_desired[0:3] = Q_WL7.multiply(kp_rotation * Q_L7L7r.xyz())

            result = np.linalg.lstsq(Jv_WL7q, v_ee_desired, rcond=None)
            qdot_desired = np.clip(result[0], -1, 1)

            self.q_iiwa_previous[:] = q_iiwa
            return q_iiwa + qdot_desired * control_period
        else:
            return self.q_iiwa_previous

    def CalcTorqueCommand(self):
        return np.zeros(7)


class OpenLeftDoorImpedancePlan(OpenLeftDoorPlan):
    def __init__(self, angle_start, angle_end=np.pi/4, duration=10.0):
        OpenLeftDoorPlan.__init__(
            self,
            angle_start=angle_start,
            angle_end=angle_end,
            duration=duration,
            type=PlanTypes["OpenLeftDoorImpedancePlan"])
        self.q_iiwa_previous = np.zeros(7)

    def CalcPositionCommand(self, t_plan, q_iiwa):
        if t_plan < self.duration:
            self.q_iiwa_previous[:] = q_iiwa
            return q_iiwa
        else:
            return self.q_iiwa_previous

    def CalcTorqueCommand(self, t_plan, Jv_WL7q, p_HrQ, Q_L7L7r, Q_WL7):
        if t_plan < self.duration:
            # first 3: angular velocity, last 3: translational velocity
            f_ee_desired = np.zeros(6)

            # translation
            kp_translation = np.array([100., 1., 100])#*15
            f_ee_desired[3:6] = -kp_translation * p_HrQ

            # rotation
            kp_rotation = np.array([10, 40, 40])*5
            f_ee_desired[0:3] = Q_WL7.multiply(kp_rotation * Q_L7L7r.xyz())

            return np.clip(Jv_WL7q.T.dot(f_ee_desired), -20, 20)
        else:
            return np.zeros(7)
