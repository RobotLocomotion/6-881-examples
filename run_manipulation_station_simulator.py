import argparse
from plan_runner.manipulation_station_simulator import ManipulationStationSimulator

import numpy as np
from pydrake.multibody import inverse_kinematics
from pydrake.trajectories import (
    PiecewisePolynomial
)
from pydrake.util.eigen_geometry import Isometry3
from pydrake.math import RollPitchYaw, RotationMatrix
from plan_runner.robot_plans import *

from pydrake.common import FindResourceOrThrow
import matplotlib.pyplot as plt
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
        help="Add additional plans to fully open the door after imepdance/position plans.")
    args = parser.parse_args()
    is_hardware = args.hardware

    object_file_path = FindResourceOrThrow(
        "drake/examples/manipulation_station/models/061_foam_brick.sdf")
    # pose_dict_file_name = "./dope_poses.txt"
    # pose_dict = read_poses_from_file(pose_dict_file_name)

    manip_station_sim = ManipulationStationSimulator(
        time_step=2e-3,
        object_file_path=object_file_path,
        object_base_link_name="base_link",)
        # X_WObject=pose_dict["soup"])

    # Generate plans.
    q0 = [0, 0, 0, -1.75, 0, 1.0, 0]

    # plan_list, gripper_setpoint_list = GenerateOpenLeftDoorPlansByTrajectory()
    plan_list, gripper_setpoint_list = GenerateOpenLeftDoorPlansByImpedanceOrPosition(
        open_door_method="Position", is_open_fully=True)


    if is_hardware:
        iiwa_position_command_log, iiwa_position_measured_log, iiwa_external_torque_log = \
            manip_station_sim.RunRealRobot(plan_list, gripper_setpoint_list)
        PlotExternalTorqueLog(iiwa_external_torque_log)
        PlotIiwaPositionLog(iiwa_position_command_log, iiwa_position_measured_log)
    else:
        iiwa_position_command_log, iiwa_position_measured_log, iiwa_external_torque_log, \
                state_log = \
            manip_station_sim.RunSimulation(plan_list, gripper_setpoint_list,
                                        extra_time=2.0, real_time_rate=1.0, q0_kuka=q0)
        PlotExternalTorqueLog(iiwa_external_torque_log)
        PlotIiwaPositionLog(iiwa_position_command_log, iiwa_position_measured_log)

