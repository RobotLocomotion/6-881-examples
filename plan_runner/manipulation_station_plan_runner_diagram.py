import numpy as np
import sys

from pydrake.all import (
    BasicVector,
    PortDataType,
)
from pydrake.systems.framework import Diagram, AbstractValue, LeafSystem, DiagramBuilder
from plan_runner.robot_plans import *
from plan_runner.open_left_door_plans import *


class PlanScheduler(LeafSystem):
    def __init__(self, kuka_plans, gripper_setpoint_list, update_period=0.05):
        LeafSystem.__init__(self)
        self.set_name("Plan Scheduler")

        assert len(kuka_plans) == len(gripper_setpoint_list)

        # Add a zero order hold to hold the current position of the robot
        kuka_plans.insert(0, JointSpacePlanRelative(
            duration=3.0, delta_q=np.zeros(7)))
        gripper_setpoint_list.insert(0, 0.055)

        if len(kuka_plans) > 1:
            # Insert to the beginning of plan_list a plan that moves the robot from its
            # current position to plan_list[0].traj.value(0)
            kuka_plans.insert(1, JointSpacePlanGoToTarget(
                duration=6.0, q_target=kuka_plans[1].traj.value(0).flatten()))
            gripper_setpoint_list.insert(0, 0.055)

        self.gripper_setpoint_list = gripper_setpoint_list
        self.kuka_plans_list = kuka_plans

        self.current_plan = None
        self.current_gripper_setpoint = None
        self.current_plan_idx = 0

        # Output ports for plans and gripper setpoints
        self.iiwa_plan_output_port = self._DeclareAbstractOutputPort(
            "iiwa_plan",
            lambda: AbstractValue.Make(PlanBase()),
            self._GetCurrentPlan)

        self.hand_setpoint_output_port = \
            self._DeclareVectorOutputPort(
                "gripper_setpoint", BasicVector(1), self._CalcHandSetpointOutput)
        self.gripper_force_limit_output_port = \
            self._DeclareVectorOutputPort(
                "force_limit", BasicVector(1), self._CalcForceLimitOutput)

        self._DeclarePeriodicPublish(update_period)
        self.kPlanDurationMultiplier = 1.1

    def _GetCurrentPlan(self, context, y_data):
        t = context.get_time()
        if self.current_plan is None:
            # This is true only after the constructor is called and at the first control tick after the
            # simulator starts.
            self.current_plan = self.kuka_plans_list.pop(0)
            self.current_gripper_setpoint = self.gripper_setpoint_list.pop(0)
            self.current_plan.start_time = 0.
        else:
            if t - self.current_plan.start_time >= self.current_plan.duration * self.kPlanDurationMultiplier:
                if len(self.kuka_plans_list) > 0:
                    self.current_plan = self.kuka_plans_list.pop(0)
                    self.current_gripper_setpoint = self.gripper_setpoint_list.pop(0)
                else:
                    # There are no more available plans. Hold current position.
                    self.current_plan = JointSpacePlanRelative(
                        duration=3600., delta_q=np.zeros(7))
                    print 'No more plans to run, holding current position...\n'

                self.current_plan.start_time = t
                self.current_plan_idx += 1
                print 'Running plan %d' % self.current_plan_idx + " (type: " + self.current_plan.type + \
                      "), starting at %f for a duration of %f seconds." % \
                      (t, self.current_plan.duration * self.kPlanDurationMultiplier) + "\n"

        y_data.set_value(self.current_plan)

    def _CalcHandSetpointOutput(self, context, y_data):
        y = y_data.get_mutable_value()
        # Get the ith finger control output
        y[:] = self.current_gripper_setpoint

    def _CalcForceLimitOutput(self, context, output):
        output.SetAtIndex(0, 15.0)


class IiwaController(LeafSystem):
    def __init__(self, station, control_period=0.005, print_period=0.5):
        LeafSystem.__init__(self)
        self.set_name("Iiwa Controller")

        self.current_plan = None

        # Stuff for iiwa control
        self.nu = 7
        self.print_period = print_period
        self.last_print_time = -print_period
        self.control_period = control_period

        # create a multibodyplant containing the robot only, which is used for
        # jacobian calculations.
        self.plant_iiwa = station.get_controller_plant()
        self.tree_iiwa = self.plant_iiwa.tree()
        self.context_iiwa = self.plant_iiwa.CreateDefaultContext()
        self.l7_frame = self.plant_iiwa.GetFrameByName('iiwa_link_7')

        # Declare iiwa_position/torque_command publishing rate
        self._DeclarePeriodicPublish(control_period)

        # iiwa position input port
        self.iiwa_position_input_port = \
            self._DeclareInputPort(
                "iiwa_position", PortDataType.kVectorValued, 7)

        # iiwa velocity input port
        self.iiwa_velocity_input_port = \
            self._DeclareInputPort(
                "iiwa_velocity", PortDataType.kVectorValued, 7)

        # plan abstract input port
        self.plan_input_port = \
            self._DeclareAbstractInputPort(
                "iiwa_plan", AbstractValue.Make(PlanBase()))

        # position and torque command output port
        # first 7 elements are position commands.
        # last 7 elements are torque commands.
        self.iiwa_position_command_output_port = \
            self._DeclareVectorOutputPort("iiwa_position_and_torque_command",
                                          BasicVector(self.nu*2), self._CalcIiwaCommand)

    def _CalcIiwaCommand(self, context, y_data):
        t = context.get_time()

        self.current_plan = self.EvalAbstractInput(
            context, self.plan_input_port.get_index()).get_value()
        q_iiwa = self.EvalVectorInput(
            context, self.iiwa_position_input_port.get_index()).get_value()
        t_plan = t - self.current_plan.start_time
        new_position_command = np.zeros(7)
        new_position_command[:] = q_iiwa
        new_torque_command = np.zeros(7)

        if self.current_plan.type == PlanTypes["JointSpacePlan"]:
            new_position_command[:] = self.current_plan.traj.value(t_plan).flatten()

        elif self.current_plan.type == PlanTypes["JointSpacePlanRelative"] or \
                self.current_plan.type == PlanTypes["JointSpacePlanGoToTarget"]:
            if self.current_plan.traj is None:
                self.current_plan.UpdateTrajectory(q_start=q_iiwa)
            new_position_command[:] = self.current_plan.traj.value(t_plan).flatten()

        elif self.current_plan.type == PlanTypes["IiwaTaskSpacePlan"]:
            if self.current_plan.xyz_offset is None:
                # update self.context_iiwa
                x_iiwa_mutable = \
                    self.tree_iiwa.GetMutablePositionsAndVelocities(self.context_iiwa)
                x_iiwa_mutable[:7] = q_iiwa

                # Pose of frame L7 in world frame
                X_WL7 = self.tree_iiwa.CalcRelativeTransform(
                    self.context_iiwa, frame_A=self.plant_iiwa.world_frame(),
                    frame_B=self.l7_frame)

                # Position of Q in world frame
                p_L7Q = X_L7E.multiply(p_EQ)
                p_WQ = X_WL7.multiply(p_L7Q)

                self.current_plan.UpdateXyzOffset(p_WQ)

            new_position_command[:] = self.current_plan.CalcPositionCommand(
                t_plan, q_iiwa, self.control_period)

        elif self.current_plan.type == PlanTypes["OpenLeftDoorImpedancePlan"] or \
                self.current_plan.type == PlanTypes["OpenLeftDoorPositionPlan"]:
            # update self.context_iiwa
            x_iiwa_mutable = \
                self.tree_iiwa.GetMutablePositionsAndVelocities(self.context_iiwa)
            x_iiwa_mutable[:7] = q_iiwa

            Jv_WL7q, p_HrQ, R_L7L7r, R_WL7 = self.current_plan.CalcKinematics(
                l7_frame=self.l7_frame,
                world_frame=self.plant_iiwa.world_frame(),
                tree_iiwa=self.tree_iiwa, context_iiwa=self.context_iiwa,
                t_plan=t_plan)

            # compute commands
            if self.current_plan.type == PlanTypes["OpenLeftDoorPositionPlan"]:
                new_position_command[:] = self.current_plan.CalcPositionCommand(
                    t_plan, q_iiwa, Jv_WL7q, p_HrQ, R_L7L7r, R_WL7, self.control_period)
                new_torque_command[:] = self.current_plan.CalcTorqueCommand()
            elif self.current_plan.type == PlanTypes["OpenLeftDoorImpedancePlan"]:
                new_position_command[:] = self.current_plan.CalcPositionCommand(t_plan, q_iiwa)
                new_torque_command[:] = self.current_plan.CalcTorqueCommand(
                    t_plan, Jv_WL7q, p_HrQ, R_L7L7r, R_WL7)

        y = y_data.get_mutable_value()
        y[:self.nu] = new_position_command[:]
        y[self.nu:] = new_torque_command[:]

        # print current simulation time
        if (self.print_period and
                t - self.last_print_time >= self.print_period):
            print "t: ", context.get_time()
            self.last_print_time = context.get_time()


def CreateManipStationPlanRunnerDiagram(station, kuka_plans, gripper_setpoint_list, print_period=1.0):
    builder = DiagramBuilder()

    iiwa_controller = IiwaController(station, print_period=print_period)
    builder.AddSystem(iiwa_controller)
    plan_scheduler = PlanScheduler(kuka_plans, gripper_setpoint_list)
    builder.AddSystem(plan_scheduler)

    builder.Connect(plan_scheduler.iiwa_plan_output_port,
                    iiwa_controller.plan_input_port)

    builder.ExportInput(iiwa_controller.iiwa_position_input_port,
                        "iiwa_position")
    builder.ExportInput(iiwa_controller.iiwa_velocity_input_port,
                        "iiwa_velocity")

    builder.ExportOutput(iiwa_controller.iiwa_position_command_output_port,
                         "iiwa_position_and_torque_command")
    builder.ExportOutput(plan_scheduler.hand_setpoint_output_port, "gripper_setpoint")
    builder.ExportOutput(plan_scheduler.gripper_force_limit_output_port, "force_limit")

    plan_runner = builder.Build()
    plan_runner.set_name("Plan Runner")

    return plan_runner, plan_scheduler.kPlanDurationMultiplier



