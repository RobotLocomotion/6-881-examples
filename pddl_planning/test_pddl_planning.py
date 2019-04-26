import unittest
import numpy as np
import os
import meshcat

from pddl_planning.problems import load_dope
from pddl_planning.simulation import compute_duration, ForceControl
from plan_runner.manipulation_station_simulator import ManipulationStationSimulator
from plan_runner.open_left_door import GenerateOpenLeftDoorPlansByImpedanceOrPosition

class TestPDDLPlanning(unittest.TestCase):
    def setUp(self):
        self.q0 = [0, 0, 0, -1.75, 0, 1.0, 0]
        self.time_step = 2e-3

        self.prevdir = os.getcwd()
        os.chdir(os.path.expanduser("pddl_planning"))

        task, diagram, state_machine = load_dope(time_step=self.time_step,
                                                 dope_path="poses.txt",
                                                 goal_name="soup",
                                                 is_visualizing=False)
        plant = task.mbp

        task.publish()
        context = diagram.GetMutableSubsystemContext(plant, task.diagram_context)

        world_frame = plant.world_frame()
        X_WSoup = plant.CalcRelativeTransform(
            context, frame_A=world_frame, frame_B=plant.GetFrameByName("base_link_soup"))

        self.manip_station_sim = ManipulationStationSimulator(
            time_step=self.time_step,
            object_file_path="./models/ycb_objects/soup_can.sdf",
            object_base_link_name="base_link_soup",
            X_WObject=X_WSoup)

    def tearDown(self):
        os.chdir(self.prevdir)

    def InspectLog(self, state_log, plant):
        tree = plant.tree()
        data = state_log.data()

        # create a context of final state.
        x_final = data[:, -1]
        context = plant.CreateDefaultContext()
        x_mutable = tree.GetMutablePositionsAndVelocities(context)
        x_mutable[:] = x_final

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

        return (np.abs(q_iiwa_target - q_iiwa_final) < 0.03).all()

    def test_pddl(self):
        splines = np.load("test_data/splines.npy")
        setpoints = np.load("test_data/gripper_setpoints.npy")

        plan_list = []
        gripper_setpoints = []
        for control, setpoint in zip(splines, setpoints):
            plan_list.append(control.plan())
            gripper_setpoints.append(setpoint)

        sim_duration = compute_duration(plan_list)

        q_iiwa_beginning = plan_list[0].traj.value(0).flatten()

        iiwa_position_command_log, iiwa_position_measured_log, iiwa_external_torque_log, \
            plant_state_log, t_plan = \
            self.manip_station_sim.RunSimulation(plan_list, gripper_setpoints,
                                            extra_time=2.0, real_time_rate=0.0,
                                            q0_kuka=self.q0, is_visualizing=False)

        # Run Tests
        self.InspectLog(plant_state_log, self.manip_station_sim.plant)
        self.assertTrue(
                self.HasReturnedToQtarget(q_iiwa_beginning, plant_state_log, self.manip_station_sim.plant))

    def test_pddl_force_control(self):
        splines = np.load("test_data/splines_force_control.npy")
        setpoints = np.load("test_data/gripper_setpoints_force_control.npy")

        plan_list = []
        gripper_setpoints = []
        for control, setpoint in zip(splines, setpoints):
            if isinstance(control, ForceControl):
                new_plans, new_setpoints = \
                    GenerateOpenLeftDoorPlansByImpedanceOrPosition("Impedance", is_open_fully=True)
                plan_list.extend(new_plans)
                gripper_setpoints.extend(new_setpoints)
            else:
                plan_list.append(control.plan())
                gripper_setpoints.append(setpoint)

        sim_duration = compute_duration(plan_list)

        q_iiwa_beginning = plan_list[0].traj.value(0).flatten()

        iiwa_position_command_log, iiwa_position_measured_log, iiwa_external_torque_log, \
            plant_state_log, t_plan = \
            self.manip_station_sim.RunSimulation(plan_list, gripper_setpoints,
                                            extra_time=2.0, real_time_rate=0.0,
                                            q0_kuka=self.q0, is_visualizing=False)

        # Run Tests
        self.InspectLog(plant_state_log, self.manip_station_sim.plant)
        self.assertTrue(
                self.HasReturnedToQtarget(q_iiwa_beginning, plant_state_log, self.manip_station_sim.plant))
