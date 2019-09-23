import argparse
import numpy as np
import cProfile
from plan_runner.manipulation_station_simulator import ManipulationStationSimulator
from plan_runner.manipulation_station_joint_trajectory_runner import *
from plan_runner.open_left_door import GenerateExampleJointAndTaskSpacePlans

if __name__ == '__main__':
    # define command line arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hardware", action='store_true',
        help="Use the ManipulationStationHardwareInterface instead of an "
             "in-process simulation.")
    parser.add_argument(
        "--no_visualization", action="store_true", default=False,
        help="Turns off visualization")
    parser.add_argument(
        "--profiling", action="store_true", default=False,
        help="profile RunSimulation or RunRealRobot with cProfile")
    parser.add_argument(
        "--diagram_plan_runner", action="store_true", default=False,
        help="Use the diagram version of plan_runner")
    args = parser.parse_args()
    is_hardware = args.hardware

    # Construct simulator system.
    manip_station_sim = ManipulationStationSimulator(time_step=2e-3)

    # Generate plans.
    plan_list, gripper_setpoint_list = GenerateExampleJointAndTaskSpacePlans()

    # Run simulator (simulation or hardware).
    if is_hardware:
        if args.profiling:
            cProfile.run("manip_station_sim.RunRealRobot(\
                    plan_list, gripper_setpoint_list,\
                    is_plan_runner_diagram=args.diagram_plan_runner)", "stats_hardware")
        else:
            iiwa_position_command_log, iiwa_position_measured_log, iiwa_velocity_estimated_log, \
                iiwa_external_torque_log, t_plan = \
                manip_station_sim.RunRealRobot(
                    plan_list, gripper_setpoint_list,
                    is_plan_runner_diagram=args.diagram_plan_runner)
            PlotExternalTorqueLog(iiwa_external_torque_log)
            PlotIiwaPositionLog(iiwa_position_command_log, iiwa_position_measured_log)
            PlotEeOrientationError(iiwa_position_measured_log, R_WL7_ref.ToQuaternion(), t_plan)
    else:
        q0 = [0, 0, 0, -1.75, 0, 1.0, 0]
        if args.profiling:
            cProfile.run("manip_station_sim.RunSimulation(\
                plan_list, gripper_setpoint_list, extra_time=2.0, real_time_rate=0.0, q0_kuka=q0,\
                is_visualizing=not args.no_visualization,\
                is_plan_runner_diagram=args.diagram_plan_runner)", "stats_simulation")
        else:
            iiwa_position_command_log, iiwa_position_measured_log, iiwa_external_torque_log, \
                state_log, t_plan = manip_station_sim.RunSimulation(
                    plan_list, gripper_setpoint_list, extra_time=2.0, real_time_rate=1.0, q0_kuka=q0,
                    is_visualizing=not args.no_visualization,
                    is_plan_runner_diagram=args.diagram_plan_runner)
            PlotExternalTorqueLog(iiwa_external_torque_log)
            PlotIiwaPositionLog(iiwa_position_command_log, iiwa_position_measured_log)
            PlotEeOrientationError(iiwa_position_measured_log, R_WL7_ref.ToQuaternion(), t_plan)