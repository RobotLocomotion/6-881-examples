import numpy as np

from plan_runner.manipulation_station_simulator import ManipulationStationSimulator
from plan_runner.manipulation_station_plan_runner import *
from plan_runner.open_left_door import InverseKinPointwise, GetKukaQKnots
from contact_aware_control import JointSpacePlanContact

if __name__ == '__main__':
    manip_station_sim = ManipulationStationSimulator(time_step=2e-3)

    q0 = [0, 0, 0, -1.75, 0, 1.0, 0]

    # create a plan
    R_WL7_ref = RollPitchYaw(0, np.pi, 0).ToRotationMatrix()
    def ReturnConstantOrientation(i, num_knot_points):
        return R_WL7_ref
    def InterpolateStraightLine(p_WQ_start, p_WQ_end, num_knot_points, i):
        return (p_WQ_end - p_WQ_start)/num_knot_points*(i+1) + p_WQ_start

    duration = 5.
    num_knot_points = 10

    qtraj, q_knots = InverseKinPointwise(
        p_WQ_start=np.array([0.5, 0, 0.41]),
        p_WQ_end=np.array([0.5, 0, -0.15]),
        duration=duration,
        num_knot_points=num_knot_points,
        q_initial_guess=np.zeros(18),
        InterpolatePosition=InterpolateStraightLine,
        InterpolateOrientation=ReturnConstantOrientation,
        p_BQ=np.zeros(3))

    t_knots = np.linspace(0, duration, num_knot_points)
    q_knots_kuka = GetKukaQKnots(q_knots[1:])
    qtraj = PiecewisePolynomial.Cubic(
        t_knots, q_knots_kuka.T, np.zeros(7), np.zeros(7))

    plan_list = [JointSpacePlanContact(qtraj)]
    gripper_setpoint_list = [0.1]

    # Run simulation
    iiwa_position_command_log, iiwa_position_measured_log, iiwa_external_torque_log, \
    state_log, t_plan, contact_detector = manip_station_sim.RunSimulation(
        plan_list,
        gripper_setpoint_list,
        extra_time=2.0,
        real_time_rate=0.0,
        q0_kuka=q0)
    PlotExternalTorqueLog(iiwa_external_torque_log)
    PlotIiwaPositionLog(iiwa_position_command_log, iiwa_position_measured_log)