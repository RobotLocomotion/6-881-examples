from pydrake.multibody import inverse_kinematics
from pydrake.common.eigen_geometry import Isometry3
from pydrake.math import RollPitchYaw, RotationMatrix

from plan_runner.manipulation_station_plan_runner import *

# Define global variables used for IK.
plant = station.get_mutable_multibody_plant()

iiwa_model = plant.GetModelInstanceByName("iiwa")
world_frame = plant.world_frame()
l7_frame = plant.GetFrameByName("iiwa_link_7", iiwa_model)


def GetKukaQKnots(q_knots):
    """
    q returned by IK consists of the configuration of all bodies in a MultibodyTree.
    In this case, it includes the iiwa arm, the cupboard doors, the gripper and the manipulands, but
    the trajectory sent to iiwa only needs the configuration of iiwa itself.

    This function takes in an array of shape (n, plant.num_positions()),
    and returns an array of shape (n, 7), which only has the configuration of the iiwa arm.
    """
    if len(q_knots.shape) == 1:
        q_knots.resize(1, q_knots.size)
    n = q_knots.shape[0]
    q_knots_kuka = np.zeros((n, 7))
    for i, q_knot in enumerate(q_knots):
        q_knots_kuka[i] = plant.GetPositionsFromArray(iiwa_model, q_knot)

    return q_knots_kuka


def InverseKinPointwise(p_WQ_start, p_WQ_end, duration,
                        num_knot_points,
                        q_initial_guess,
                        InterpolatePosition=None,
                        InterpolateOrientation=None,
                        position_tolerance=0.005,
                        theta_bound=0.005 * np.pi, # 0.9 degrees
                        is_printing=True):
    """
    Calculates a joint space trajectory for iiwa by repeatedly calling IK. The first IK is initialized (seeded)
    with q_initial_guess. Subsequent IKs are seeded with the solution from the previous IK.

    Positions for point Q (p_EQ) and orientations for the end effector, generated respectively by
    InterpolatePosition and InterpolateOrientation, are added to the IKs as constraints.

    @param p_WQ_start: The first argument of function InterpolatePosition (defined below).
    @param p_WQ_end: The second argument of function InterpolatePosition (defined below).
    @param duration: The duration of the trajectory returned by this function in seconds.
    @param num_knot_points: number of knot points in the trajectory.
    @param q_initial_guess: initial guess for the first IK.
    @param InterpolatePosition: A function with signature (start, end, num_knot_points, i). It returns
       p_WQ, a (3,) numpy array which describes the desired position of Q at knot point i in world frame.
    @param InterpolateOrientation: A function with signature
    @param position_tolerance: tolerance for IK position constraints in meters.
    @param theta_bound: tolerance for IK orientation constraints in radians.
    @param is_printing: whether the solution results of IKs are printed.
    @return: qtraj: a 7-dimensional cubic polynomial that describes a trajectory for the iiwa arm.
    @return: q_knots: a (n, num_knot_points) numpy array (where n = plant.num_positions()) that stores solutions
        returned by all IKs. It can be used to initialize IKs for the next trajectory.
    """
    q_knots = np.zeros((num_knot_points + 1, plant.num_positions()))
    q_knots[0] = q_initial_guess

    for i in range(num_knot_points):
        ik = inverse_kinematics.InverseKinematics(plant)
        q_variables = ik.q()

        # Orientation constraint
        R_WL7_ref = InterpolateOrientation(i, num_knot_points)
        ik.AddOrientationConstraint(
            frameAbar=world_frame, R_AbarA=R_WL7_ref,
            frameBbar=l7_frame, R_BbarB=RotationMatrix.Identity(),
            theta_bound=theta_bound)

        # Position constraint
        p_WQ = InterpolatePosition(p_WQ_start, p_WQ_end, num_knot_points, i)
        ik.AddPositionConstraint(
            frameB=l7_frame, p_BQ=p_L7Q,
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
    Returns a configuration of the MultibodyPlant in which point Q (defined by global variable p_EQ)
    in robot EE frame is at p_WQ_home, and orientation of frame Ea is R_WEa_ref.
    """
    # get "home" pose
    ik_scene = inverse_kinematics.InverseKinematics(plant)

    theta_bound = 0.005 * np.pi # 0.9 degrees
    X_EEa = GetEndEffectorWorldAlignedFrame()
    R_EEa = RotationMatrix(X_EEa.rotation())

    ik_scene.AddOrientationConstraint(
        frameAbar=world_frame, R_AbarA=R_WL7_ref,
        frameBbar=l7_frame, R_BbarB=RotationMatrix.Identity(),
        theta_bound=theta_bound)

    p_WQ0 = p_WQ_home
    p_WQ_lower = p_WQ0 - 0.005
    p_WQ_upper = p_WQ0 + 0.005
    ik_scene.AddPositionConstraint(
        frameB=l7_frame, p_BQ=p_L7Q,
        frameA=world_frame,
        p_AQ_lower=p_WQ_lower, p_AQ_upper=p_WQ_upper)

    prog = ik_scene.prog()
    prog.SetInitialGuess(ik_scene.q(), np.zeros(plant.num_positions()))
    result = prog.Solve()
    if is_printing:
        print result
    return prog.GetSolution(ik_scene.q())


def GenerateApproachHandlePlans(InterpolateOrientation, p_WQ_end, is_printing=True):
    """
    Returns a list of Plans that move the end effector from its home position to
    the left door handle. Also returns the corresponding gripper setpoints and IK solutions.

    @param InterpolateOrientation: a function passed to InverseKinPointwise, which returns the desired
        end effector orienttion along the trajectory. 
    """
    q_home_full = GetHomeConfiguration(is_printing)

    def InterpolateStraightLine(p_WQ_start, p_WQ_end, num_knot_points, i):
        return (p_WQ_end - p_WQ_start)/num_knot_points*(i+1) + p_WQ_start

    # Generating trajectories
    num_knot_points = 10

    # move to grasp left door handle
    p_WQ_start = p_WQ_home
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


def GenerateOpenLeftDoorTrajectory(q_initial_guess, handle_angle_end,
                                   duration, is_printing=True):
    """
    Creates an iiwa trajectory that opens the left door by pulling the handle.
    The left door should be fully closed at the beginning of the trajectory.
    @param q_initial_guess: initial guess for the first IK.
    @param handle_angle_end: the desired left door angle at the end of the trajectory.
    @param duration: duration of the open-door trajectory.
    @param is_printing: whether IK solution results are printed to the screen.
    """
    # pull handle along an arc
    def InterpolateArc(angle_start, angle_end, num_knot_points, i):
        radius = r_handle
        theta = angle_start + (angle_end - angle_start)*(i+1)/num_knot_points
        return p_WC_left_hinge + [-radius * np.sin(theta), -radius * np.cos(theta), 0]

    angle_start = theta0_hinge
    angle_end = handle_angle_end
    def InterpolateYawAngle(i, num_knot_points):
        assert i <= num_knot_points
        yaw_angle = 0. + (angle_end - angle_start)/num_knot_points*i
        return RollPitchYaw(0, np.pi/2, -yaw_angle).ToRotationMatrix()

    qtraj_pull_handle, q_knots_full = InverseKinPointwise(
        angle_start, angle_end, duration=duration, num_knot_points=20,
        q_initial_guess=q_initial_guess,
        InterpolatePosition=InterpolateArc,
        InterpolateOrientation=InterpolateYawAngle,
        position_tolerance=0.002,
        theta_bound=np.pi/180*5,
        is_printing=is_printing)

    return qtraj_pull_handle

def AddOpenDoorFullyPlans(plan_list, gripper_setpoint_list):
    """
    Appends to plan_list and gripper_setpoint_list hand-crafted trajectories that push
    the left door fully open (~90 degrees) after pulling.
    """
    # Add zero order old to open gripper
    plan_open_gripper = JointSpacePlanRelative(
        duration=2.0, delta_q=np.zeros(7))
    plan_list.append(plan_open_gripper)
    gripper_setpoint_list.append(0.03)

    # Add plans to push the door open after pulling the handle
    xyz_durations = [5., 10., 10.]
    xyz_gripper_setpoint = [0.03, 0.05, 0.02, ]
    delta_xyz = np.zeros((3, 3))
    delta_xyz[0] = [-0.03, 0, 0]
    delta_xyz[1] = [0, -0.15, 0]
    delta_xyz[2] = [0.13, 0, 0]

    for i in range(3):
        xyz_traj = ConnectPointsWithCubicPolynomial(
            np.zeros(3), delta_xyz[i], xyz_durations[i])
        plan_list.append(IiwaTaskSpacePlan(
            xyz_traj=xyz_traj,
            Q_WL7_ref=R_WL7_ref.ToQuaternion(),
            p_L7Q=p_L7Q))
        gripper_setpoint_list.append(xyz_gripper_setpoint[i])

    # plan from current position to pre-swing
    plan_pre_swing = JointSpacePlanGoToTarget(
        duration=4.0,
        q_target=q_pre_swing)
    plan_list.append(plan_pre_swing)
    gripper_setpoint_list.append(0.10)

    # swing to open the door
    plan_swing = JointSpacePlan(
        ConnectPointsWithCubicPolynomial(q_pre_swing, q_post_swing, duration=6.0))
    plan_list.append(plan_swing)
    gripper_setpoint_list.append(0.10)

    # return to home pose
    plan_return_home = JointSpacePlan(
        ConnectPointsWithCubicPolynomial(q_post_swing, q_home, duration=6.0))
    plan_list.append(plan_return_home)
    gripper_setpoint_list.append(0.10)


# global variables used for all door-opening plans.
handle_angle_end = np.pi/180*50
open_door_duration = 10.

def GenerateOpenLeftDoorPlansByTrajectory(is_printing=True):
    """
    Creates iiwa plans and gripper set points that
    - starts at a home configuration,
    - approaches the left door,
    - opens the left door by pulling the handle.
    The pulling actions are a result of following a joint space trajectory generated by solving IKs.
    """
    def InterpolatePitchAngle(i, num_knot_points):
        assert i <= num_knot_points
        pitch_start = np.pi / 180 * 135
        pitch_end = np.pi / 180 * 90
        pitch_angle = pitch_start + (pitch_end - pitch_start) / num_knot_points * i
        return RollPitchYaw(0, pitch_angle, 0).ToRotationMatrix()

    plan_list, gripper_setpoint_list, q_final_full = \
        GenerateApproachHandlePlans(InterpolatePitchAngle,
                                    p_WQ_end=p_WC_handle,
                                    is_printing=is_printing)

    plan_list.append(JointSpacePlan(
        GenerateOpenLeftDoorTrajectory(
            q_initial_guess=q_final_full,
            handle_angle_end=handle_angle_end,
            duration=open_door_duration)))
    gripper_setpoint_list.append(gripper_setpoint_list[-1])

    return plan_list, gripper_setpoint_list

def GenerateOpenLeftDoorPlansByImpedanceOrPosition(
        open_door_method="Impedance", is_open_fully=False, is_printing=True):
    """
    Creates iiwa plans and gripper set points that
    - starts at a home configuration,
    - approaches the left door,
    - opens the left door by pulling the handle.
    The pulling actions are a result of following commands generated by a Position or Impedance controller. Details
    of the controllers are defined in the lab 2 handout. There are no explict trajectories to follow.
    If is_open_fully is True, the robot executes a hand-crafted maneuver to push the door fully open.
    """
    def ReturnConstantOrientation(i, num_knot_points):
        return R_WL7_ref

    # Move end effector towards the left door handle.
    plan_list, gripper_setpoint_list, q_final_full = \
        GenerateApproachHandlePlans(ReturnConstantOrientation,
                                    p_WQ_end=p_WC_handle,
                                    is_printing=is_printing)

    if open_door_method == "Impedance":
        # Add the position/impedance plan that opens the left door.
        plan_list.append(OpenLeftDoorImpedancePlan(
            angle_start=theta0_hinge,
            angle_end=handle_angle_end,
            duration=open_door_duration,
            Q_WL7_ref=R_WL7_ref.ToQuaternion()))
    elif open_door_method == "Position":
        plan_list.append(OpenLeftDoorPositionPlan(
            angle_start=theta0_hinge,
            angle_end=handle_angle_end,
            duration=open_door_duration,
            Q_WL7_ref=R_WL7_ref.ToQuaternion()))
    gripper_setpoint_list.append(gripper_setpoint_list[-1])

    if is_open_fully:
        AddOpenDoorFullyPlans(plan_list, gripper_setpoint_list)

    return plan_list, gripper_setpoint_list

def GenerateExampleJointAndTaskSpacePlans():
    """
    Creates iiwa plans and gripper set points that
    - start at a home configuration
    - approach a point 10 cm behind the door handle (JointSpacePlan)
    - move the end effector along a few starght line segments (IiwaTaskSpacePlan)
    """
    def ReturnConstantOrientation(i, num_knot_points):
        return R_WL7_ref

    # Move end effector towards the left door handle.
    plan_list, gripper_setpoint_list, q_final_full = \
        GenerateApproachHandlePlans(ReturnConstantOrientation,
                                    p_WQ_end=p_WC_handle - np.array([0.11, 0, 0]),
                                    is_printing=True)

    # Add task space plans
    xyz_durations = [6., 6., 6.]
    xyz_gripper_setpoint = [0.01, 0.01, 0.01, ]
    delta_xyz = np.zeros((3, 3))
    delta_xyz[0] = [-0.03, 0, 0]
    delta_xyz[1] = [0, -0.15, 0]
    delta_xyz[2] = [0.13, 0, 0]

    for i in range(3):
        xyz_traj = ConnectPointsWithCubicPolynomial(
            np.zeros(3), delta_xyz[i], xyz_durations[i])
        plan_list.append(IiwaTaskSpacePlan(
            xyz_traj=xyz_traj,
            Q_WL7_ref=R_WL7_ref.ToQuaternion(),
            p_L7Q=p_L7Q))
        gripper_setpoint_list.append(xyz_gripper_setpoint[i])

    return plan_list, gripper_setpoint_list

