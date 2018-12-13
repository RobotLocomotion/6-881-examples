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

# position of point Q in L7 frame.  Point Q is fixed w.r.t frame L7.
# When the robot is upright (all joint angles = 0), axes of L7 are aligned with axes of world.
# The origin of end effector frame (plant.GetFrameByName('body')) is located at [0, 0, 0.114] in frame L7.
p_L7Q = np.array([0., 0., 0.090]) + np.array([0, 0, 0.114])

# orientation of end effector aligned frame
R_WL7_ref = RollPitchYaw(0, np.pi / 180 * 135, 0).ToRotationMatrix()

q_home = np.array([0, -0.2136, 0, -2.094, 0, 0.463, 0])
q_pre_swing = np.array([2.44, 16.72, -17.43, -89.56, 47.30, 63.53, -83.77])*np.pi/180
q_post_swing = np.array([20.0, 16.72, -17.43, -89.56, 47.30, 63.53, -83.77])*np.pi/180



class OpenLeftDoorPlan(JacobianBasedPlan):
    def __init__(self, angle_start, angle_end, duration, Q_WL7_ref, type):
        # angle is a function of time, which is stored as a trajectory in self.traj
        angle_traj = ConnectPointsWithCubicPolynomial(
            [angle_start], [angle_end], duration)

        JacobianBasedPlan.__init__(self,
                                   plan_type=type,
                                   trajectory=angle_traj,
                                   Q_WL7_ref=Q_WL7_ref,
                                   p_L7Q=p_L7Q)

    def CalcPositionError(self, t_plan):
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

        p_HrQ = X_HrW.multiply(self.p_WQ)
        return -p_HrQ


class OpenLeftDoorPositionPlan(OpenLeftDoorPlan):
    def __init__(self, **kwargs):
        OpenLeftDoorPlan.__init__(
            self,
            angle_start=kwargs['angle_start'],
            angle_end=kwargs['angle_end'],
            duration=kwargs['duration'],
            Q_WL7_ref=kwargs['Q_WL7_ref'],
            type=PlanTypes["OpenLeftDoorPositionPlan"])
        self.q_iiwa_previous = np.zeros(7)

    def CalcPositionCommand(self, q_iiwa, v_iiwa, tau_iiwa, t_plan, control_period):
        self.CalcKinematics(q_iiwa, v_iiwa)

        if t_plan < self.duration:
            err_position = self.CalcPositionError(t_plan)
            Q_L7L7r = self.CalcOrientationError(t_plan)

            # first 3: angular velocity, last 3: translational velocity
            v_ee_desired = np.zeros(6)

            # Translation
            kp_x = 100*np.clip(t_plan/(0.2*self.duration), 0, 1)
            kp_translation = np.array([kp_x, 2., 100])/4
            v_ee_desired[3:6] = kp_translation * err_position

            # Rotation
            kp_rotation = np.array([2.5, 10, 10])*4
            v_ee_desired[0:3] = self.Q_WL7.multiply(kp_rotation * Q_L7L7r.xyz())

            result = np.linalg.lstsq(self.Jv_WL7q, v_ee_desired, rcond=None)
            qdot_desired = np.clip(result[0], -1, 1)

            self.q_iiwa_previous[:] = q_iiwa
            return q_iiwa + qdot_desired * control_period
        else:
            return self.q_iiwa_previous


class OpenLeftDoorImpedancePlan(OpenLeftDoorPlan):
    def __init__(self, **kwargs):
        OpenLeftDoorPlan.__init__(
            self,
            angle_start=kwargs['angle_start'],
            angle_end=kwargs['angle_end'],
            duration=kwargs['duration'],
            Q_WL7_ref=kwargs['Q_WL7_ref'],
            type=PlanTypes["OpenLeftDoorImpedancePlan"])
        self.q_iiwa_previous = np.zeros(7)

    def CalcPositionCommand(self, q_iiwa, v_iiwa, tau_iiwa, t_plan, control_period):
        if t_plan < self.duration:
            self.q_iiwa_previous[:] = q_iiwa
            return q_iiwa
        else:
            return self.q_iiwa_previous

    def CalcTorqueCommand(self, q_iiwa, v_iiwa, tau_iiwa, t_plan, control_period):
        self.CalcKinematics(q_iiwa, v_iiwa)

        if t_plan < self.duration:
            err_position = self.CalcPositionError(t_plan)
            Q_L7L7r = self.CalcOrientationError(t_plan)

            # first 3: angular velocity, last 3: translational velocity
            f_ee_desired = np.zeros(6)

            # translation
            kp_translation = np.array([100., 1., 100])#*15
            f_ee_desired[3:6] = kp_translation * err_position

            # rotation
            kp_rotation = np.array([10, 40, 40])*5
            f_ee_desired[0:3] = self.Q_WL7.multiply(kp_rotation * Q_L7L7r.xyz())

            return np.clip(self.Jv_WL7q.T.dot(f_ee_desired), -20, 20)
        else:
            return np.zeros(7)
