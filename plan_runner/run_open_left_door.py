import argparse
import numpy as np

from pydrake.common import FindResourceOrThrow

from plan_runner.manipulation_station_simulator import ManipulationStationSimulator
from plan_runner.manipulation_station_plan_runner import *
from plan_runner.open_left_door import (GenerateOpenLeftDoorPlansByTrajectory,
                                        GenerateOpenLeftDoorPlansByImpedanceOrPosition,)

if __name__ == '__main__':
    # define command line arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hardware", action='store_true',
        help="Use the ManipulationStationHardwareInterface instead of an "
             "in-process simulation.")
    parser.add_argument(
        "--open_fully", action='store_true',
        help="Add additional plans to fully open the door after impedance/position plans.")
    parser.add_argument(
        "-c", "--controller", type=str, default="Trajectory",
        choices=["Trajectory", "Impedance", "Position"],
        help="Specify the controller used to open the door. Its value should be: "
             "'Trajectory' (default), 'Impedance' or 'Position.")
    parser.add_argument(
        "--no_visualization", action="store_true", default=False,
        help="Turns off visualization")
    parser.add_argument(
        "--diagram_plan_runner", action="store_true", default=False,
        help="Use the diagram version of plan_runner")
    args = parser.parse_args()
    is_hardware = args.hardware

    # Construct simulator system.
    object_file_path = FindResourceOrThrow(
        "drake/examples/manipulation_station/models/061_foam_brick.sdf")

    manip_station_sim = ManipulationStationSimulator(
        time_step=2e-3,
        object_file_path=object_file_path,
        object_base_link_name="base_link",)

    # Generate plans.
    plan_list = None
    gripper_setpoint_list = None
    if args.controller == "Trajectory":
        plan_list, gripper_setpoint_list = GenerateOpenLeftDoorPlansByTrajectory()
    elif args.controller == "Impedance" or args.controller == "Position":
        plan_list, gripper_setpoint_list = GenerateOpenLeftDoorPlansByImpedanceOrPosition(
            open_door_method=args.controller, is_open_fully=args.open_fully)

    # Run simulator (simulation or hardware).
    if is_hardware:
        iiwa_position_command_log, iiwa_position_measured_log, iiwa_external_torque_log = \
            manip_station_sim.RunRealRobot(
                plan_list, gripper_setpoint_list,
                is_plan_runner_diagram=args.diagram_plan_runner)
        PlotExternalTorqueLog(iiwa_external_torque_log)
        PlotIiwaPositionLog(iiwa_position_command_log, iiwa_position_measured_log)
    else:
        q0 = [0, 0, 0, -1.75, 0, 1.0, 0]
        iiwa_position_command_log, iiwa_position_measured_log, iiwa_external_torque_log, \
            state_log = manip_station_sim.RunSimulation(
                plan_list, gripper_setpoint_list, extra_time=2.0, real_time_rate=0.0, q0_kuka=q0,
                is_visualizing=not args.no_visualization,
                is_plan_runner_diagram=args.diagram_plan_runner)
        PlotExternalTorqueLog(iiwa_external_torque_log)
        PlotIiwaPositionLog(iiwa_position_command_log, iiwa_position_measured_log)
