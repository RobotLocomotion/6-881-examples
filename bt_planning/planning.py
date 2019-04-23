from pydrake.multibody import inverse_kinematics
from pydrake.math import RotationMatrix

from plan_runner.manipulation_station_plan_runner import *

# TODO(kmuhlrad): move these all into a class/function

# Define global variables used for IK.
station = ManipulationStation()
station.SetupDefaultStation()
station.Finalize()
plant = station.get_mutable_multibody_plant()

iiwa_model = plant.GetModelInstanceByName("iiwa")
gripper_model = plant.GetModelInstanceByName("gripper")

world_frame = plant.world_frame()
gripper_frame = plant.GetFrameByName("body", gripper_model)


def GetEndEffectorWorldAlignedFrame():
    X_EEa = Isometry3.Identity()
    X_EEa.set_rotation(np.array([[0., 1., 0,],
                                 [0, 0, 1],
                                 [1, 0, 0]]))
    return X_EEa


X_EEa = GetEndEffectorWorldAlignedFrame()
R_EEa = RotationMatrix(X_EEa.rotation())

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

# angle between the world y axis and the line connecting the hinge center to the
# handle center when the left door is fully closed (left_hinge_angle = 0).
theta0_hinge = np.arctan2(np.abs(p_handle_2_hinge[0]),
                          np.abs(p_handle_2_hinge[1]))

# position of point Q in EE frame.  Point Q is fixed in the EE frame.
p_EQ = GetEndEffectorWorldAlignedFrame().multiply(np.array([0., 0., 0.090]))

# orientation of end effector aligned frame
R_WEa_ref = RollPitchYaw(0, np.pi / 180 * 135, 0).ToRotationMatrix()


def GetKukaQKnots(q_knots):
    """
    q returned by IK consists of the configuration of all bodies in a
    MultibodyTree. In this case, it includes the iiwa arm, the cupboard doors,
    the gripper and the manipulands, but the trajectory sent to iiwa only needs
    the configuration of iiwa itself.

    This function takes in an array of shape (n, plant.num_positions()),
    and returns an array of shape (n, 7), which only has the configuration of
    the iiwa arm.
    """
    if len(q_knots.shape) == 1:
        q_knots.resize(1, q_knots.size)
    n = q_knots.shape[0]
    q_knots_kuka = np.zeros((n, 7))
    for i, q_knot in enumerate(q_knots):
        q_knots_kuka[i] = plant.GetPositionsFromArray(iiwa_model, q_knot)

    return q_knots_kuka


def InverseKinPointwise(p_WQ_start, p_WQ_end,
                        angle_start, angle_end,
                        duration, num_knot_points,
                        q_initial_guess,
                        InterpolatePosition=None,
                        InterpolateOrientation=None,
                        position_tolerance=0.005,
                        theta_bound=0.005 * np.pi, # 0.9 degrees
                        is_printing=True):
    """
    Calculates a joint space trajectory for iiwa by repeatedly calling IK. The
    first IK is initialized (seeded) with q_initial_guess. Subsequent IKs are
    seeded with the solution from the previous IK.

    Positions for point Q (p_EQ) and orientations for the end effector,
    generated respectively by InterpolatePosition and InterpolateOrientation,
    are added to the IKs as constraints.

    @param p_WQ_start: The first argument of function InterpolatePosition
        (defined below).
    @param p_WQ_end: The second argument of function InterpolatePosition
        (defined below).
    @param angle_start: The first argument of function InterpolateOrientation
        (defined below).
    @param angle_end: The second argument of function InterpolateOrientation
        (defined below).
    @param duration: The duration of the trajectory returned by this function
        in seconds.
    @param num_knot_points: number of knot points in the trajectory.
    @param q_initial_guess: initial guess for the first IK.
    @param InterpolatePosition: A function with signature
        (start, end, num_knot_points, i). It returns p_WQ, a (3,) numpy array
        which describes the desired position of Q at knot point i in world
        frame.
    @param InterpolateOrientation: A function with signature
        (start, end, num_knot_points, i). It returns R_WEa, a RotationMatrix
        which describes the desired orientation of the end effector at knot
        point i in world frame.
    @param position_tolerance: tolerance for IK position constraints in meters.
    @param theta_bound: tolerance for IK orientation constraints in radians.
    @param is_printing: whether the solution results of IKs are printed.
    @return: qtraj: a 7-dimensional cubic polynomial that describes a
        trajectory for the iiwa arm.
    @return: q_knots: a (n, num_knot_points) numpy array (where
        n = plant.num_positions()) that stores solutions returned by all IKs.
        It can be used to initialize IKs for the next trajectory.
    """
    q_knots = np.zeros((num_knot_points + 1, plant.num_positions()))
    q_knots[0] = q_initial_guess

    for i in range(num_knot_points):
        ik = inverse_kinematics.InverseKinematics(plant)
        q_variables = ik.q()

        # Orientation constraint
        R_WEa_ref = InterpolateOrientation(
            angle_start, angle_end, num_knot_points, i)
        ik.AddOrientationConstraint(
            frameAbar=world_frame, R_AbarA=R_WEa_ref,
            frameBbar=gripper_frame, R_BbarB=R_EEa,
            theta_bound=theta_bound)

        # Position constraint
        p_WQ = InterpolatePosition(p_WQ_start, p_WQ_end, num_knot_points, i)
        ik.AddPositionConstraint(
            frameB=gripper_frame, p_BQ=p_EQ,
            frameA=world_frame,
            p_AQ_lower=p_WQ - position_tolerance,
            p_AQ_upper=p_WQ + position_tolerance)

        prog = ik.prog()
        # use the robot posture at the previous knot point as
        # an initial guess.
        prog.SetInitialGuess(q_variables, q_knots[i])
        result = prog.Solve()
        if is_printing:
            print i, ": ", result
        q_knots[i + 1] = prog.GetSolution(q_variables)

    t_knots = np.linspace(0, duration, num_knot_points + 1)
    q_knots_kuka = GetKukaQKnots(q_knots)
    qtraj = PiecewisePolynomial.Cubic(
        t_knots, q_knots_kuka.T, np.zeros(7), np.zeros(7))

    return qtraj, q_knots


def GetHomeConfiguration(is_printing=True):
    """
    Returns a configuration of the MultibodyPlant in which point Q (defined by
    global variable p_EQ) in robot EE frame is at p_WQ_home, and orientation of
    frame Ea is R_WEa_ref.
    """
    # get "home" pose
    ik_scene = inverse_kinematics.InverseKinematics(plant)

    theta_bound = 0.005 * np.pi # 0.9 degrees
    X_EEa = GetEndEffectorWorldAlignedFrame()
    R_EEa = RotationMatrix(X_EEa.rotation())

    ik_scene.AddOrientationConstraint(
        frameAbar=world_frame, R_AbarA=R_WEa_ref,
        frameBbar=gripper_frame, R_BbarB=R_EEa,
        theta_bound=theta_bound)

    p_WQ0 = p_WQ_home
    p_WQ_lower = p_WQ0 - 0.005
    p_WQ_upper = p_WQ0 + 0.005
    ik_scene.AddPositionConstraint(
        frameB=gripper_frame, p_BQ=p_EQ,
        frameA=world_frame,
        p_AQ_lower=p_WQ_lower, p_AQ_upper=p_WQ_upper)

    prog = ik_scene.prog()
    prog.SetInitialGuess(ik_scene.q(), np.zeros(plant.num_positions()))
    result = prog.Solve()
    if is_printing:
        print result
    return prog.GetSolution(ik_scene.q())


def InterpolateStraightLine(p_WQ_start, p_WQ_end, num_knot_points, i):
    return (p_WQ_end - p_WQ_start) / num_knot_points * (i + 1) + p_WQ_start


def InterpolateArc(angle_start, angle_end, num_knot_points, i):
    radius = r_handle
    theta = angle_start + (angle_end - angle_start) * (i + 1) / num_knot_points
    return p_WC_left_hinge + [-radius * np.sin(theta), -radius * np.cos(theta), 0]


def ReturnConstantOrientation(angle_start, angle_end, num_knot_points, i):
    assert i <= num_knot_points
    return RollPitchYaw(0, angle_start, 0).ToRotationMatrix()


def InterpolatePitchAngle(pitch_start, pitch_end, num_knot_points, i):
    assert i <= num_knot_points
    # pitch_start = np.pi / 180 * 135
    # pitch_end = np.pi / 180 * 90
    pitch_angle = pitch_start + (pitch_end - pitch_start) / num_knot_points * i
    return RollPitchYaw(0, pitch_angle, 0).ToRotationMatrix()


# angle_start = theta0_hinge
# angle_end = handle_angle_end
def InterpolateYawAngle(yaw_start, yaw_end, num_knot_points, i):
    assert i <= num_knot_points
    yaw_angle = 0. + (yaw_end - yaw_start) / num_knot_points * i
    return RollPitchYaw(0, np.pi / 2, -yaw_angle).ToRotationMatrix()


def GenerateApproachHandlePlans(InterpolateOrientation, is_printing=True):
    """
    Returns a list of Plans that move the end effector from its home position
    to the left door handle. Also returns the corresponding gripper setpoints
    and IK solutions.

    @param InterpolateOrientation: a function passed to InverseKinPointwise,
        which returns the desired end effector orientation along the trajectory.
    """
    q_home_full = GetHomeConfiguration(is_printing)

    # Generating trajectories
    num_knot_points = 10

    # move to grasp left door handle
    p_WQ_start = p_WQ_home
    p_WQ_end = p_WC_handle
    qtraj_move_to_handle, q_knots_full = InverseKinPointwise(
        p_WQ_start, p_WQ_end, duration=5.0,
        num_knot_points=num_knot_points, q_initial_guess=q_home_full,
        InterpolatePosition=InterpolateStraightLine,
        InterpolateOrientation=InterpolateOrientation,
        position_tolerance=0.001,
        is_printing=is_printing)

    # close gripper
    q_knots = np.zeros((2, 7))
    q_knots[0] = qtraj_move_to_handle.value(qtraj_move_to_handle.end_time()).squeeze()
    qtraj_close_gripper = PiecewisePolynomial.ZeroOrderHold([0, 1], q_knots.T)

    q_traj_list = [qtraj_move_to_handle,
                   qtraj_close_gripper,]

    plan_list = []
    for q_traj in q_traj_list:
        plan_list.append(JointSpacePlan(q_traj))

    gripper_setpoint_list = [0.02, 0.005] # robot

    # initial guess for the next IK
    q_final_full = q_knots_full[-1]
    return plan_list, gripper_setpoint_list, q_final_full


# TODO(kmuhlrad): make some more general functions