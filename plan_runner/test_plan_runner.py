import unittest
import numpy as np

from plan_runner.manipulation_station_simulator import ManipulationStationSimulator
from plan_runner.open_left_door import (GenerateOpenLeftDoorPlansByTrajectory,
                                        GenerateOpenLeftDoorPlansByImpedanceOrPosition,)


class TestOpenDoor(unittest.TestCase):
    def setUp(self):
        self.q0 = [0, 0, 0, -1.75, 0, 1.0, 0]

    def InspectLog(self, state_log, plant):
        tree = plant.tree()
        data = state_log.data()

        # create a context of final state.
        x_final = data[:, -1]
        context = plant.CreateDefaultContext()
        x_mutalbe = tree.GetMutablePositionsAndVelocities(context)
        x_mutalbe[:] = x_final

        # cupboard must be open.
        hinge_joint = plant.GetJointByName("left_door_hinge")
        joint_angle = hinge_joint.get_angle(context)
        self.assertTrue(np.abs(joint_angle) > np.pi/6,
                        "Cupboard door is not fully open.")

        # velocity must be small throughout the simulation.
        for x in data.T:
            v = x[plant.num_positions():]
            self.assertTrue((np.abs(v) < 3.).all(), "velocity is too large.")

    def HasReturnedToQtarget(self, q_iiwa_target, state_log, plant):
        tree = plant.tree()
        data = state_log.data()

        q_final = data[:, -1][:plant.num_positions()]
        iiwa_model = plant.GetModelInstanceByName("iiwa")
        q_iiwa_final = tree.GetPositionsFromArray(iiwa_model, q_final)

        return (np.abs(q_iiwa_target - q_iiwa_final) < 0.01).all()

    def test_open_door_by_trajectory(self):
        is_plan_runner_diagram_list = [True, False]

        for is_plan_runner_diagram in is_plan_runner_diagram_list:
            plan_list, gripper_setpoint_list = \
                GenerateOpenLeftDoorPlansByTrajectory()
            # Create simulator
            manip_station_sim = ManipulationStationSimulator(time_step=2e-3)
            iiwa_position_command_log, iiwa_position_measured_log, iiwa_external_torque_log, \
                plant_state_log = manip_station_sim.RunSimulation(
                    plan_list, gripper_setpoint_list,
                    extra_time=2.0, real_time_rate=0.0, q0_kuka=self.q0, is_visualizing=False,
                    is_plan_runner_diagram=is_plan_runner_diagram)

            # Run tests
            self.InspectLog(plant_state_log, manip_station_sim.plant)

    def test_open_door_by_impedance_and_position(self):
        modes = ("Impedance", "Position")
        is_plan_runner_diagram_list = [False, True]

        for is_plan_runner_diagram in is_plan_runner_diagram_list:
            for mode in modes:
                # Create simulator
                manip_station_sim = ManipulationStationSimulator(time_step=2e-3)
                plan_list, gripper_setpoint_list = \
                    GenerateOpenLeftDoorPlansByImpedanceOrPosition(
                        open_door_method=mode, is_open_fully=True)

                # Make a copy of the initial position of the plans passed to PlanRunner.
                q_iiwa_beginning = plan_list[0].traj.value(0).flatten()
                iiwa_position_command_log, iiwa_position_measured_log, iiwa_external_torque_log, \
                    plant_state_log = manip_station_sim.RunSimulation(
                        plan_list, gripper_setpoint_list,
                        extra_time=2.0, real_time_rate=0.0, q0_kuka=self.q0, is_visualizing=False,
                        is_plan_runner_diagram=is_plan_runner_diagram)

                # Run tests
                self.InspectLog(plant_state_log, manip_station_sim.plant)
                self.assertTrue(
                    self.HasReturnedToQtarget(q_iiwa_beginning, plant_state_log, manip_station_sim.plant))

