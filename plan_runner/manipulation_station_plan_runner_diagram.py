import numpy as np
import sys

from pydrake.systems.framework import Diagram, AbstractValue, LeafSystem, DiagramBuilder, BasicVector, PortDataType
from plan_runner.robot_plans import *
from plan_runner.open_left_door_plans import *


class PlanScheduler(LeafSystem):
    def __init__(self, kuka_plans, gripper_setpoint_list):
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
        self.current_gripper_setpoint = gripper_setpoint_list[0]
        self.current_plan_idx = 0

        # Output ports for plans and gripper setpoints
        self.iiwa_plan_output_port = self.DeclareAbstractOutputPort(
            "iiwa_plan",
            lambda: AbstractValue.Make(PlanBase()),
            self._GetCurrentPlan)

        self.gripper_setpoint_output_port = \
            self.DeclareVectorOutputPort(
                "gripper_setpoint", BasicVector(1), self._CalcGripperSetpointOutput)
        self.gripper_force_limit_output_port = \
            self.DeclareVectorOutputPort(
                "force_limit", BasicVector(1), self._CalcForceLimitOutput)

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

    def _CalcGripperSetpointOutput(self, context, y_data):
        y = y_data.get_mutable_value()
        # Get the ith finger control output
        y[:] = self.current_gripper_setpoint

    def _CalcForceLimitOutput(self, context, output):
        output.SetAtIndex(0, 15.0)


class IiwaController(LeafSystem):
    def __init__(self, control_period=0.005, print_period=0.5):
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
        self.context_iiwa = self.plant_iiwa.CreateDefaultContext()
        self.l7_frame = self.plant_iiwa.GetFrameByName('iiwa_link_7')

        # iiwa position input port
        self.iiwa_position_input_port = \
            self.DeclareVectorInputPort(
                "iiwa_position", BasicVector(7))

        # iiwa velocity input port
        self.iiwa_velocity_input_port = \
            self.DeclareVectorInputPort(
                "iiwa_velocity", BasicVector(7))

        # iiwa external torque input port
        self.iiwa_external_torque_input_port = \
            self.DeclareVectorInputPort(
                "iiwa_torque_external", BasicVector(7))

        # plan abstract input port
        self.plan_input_port = \
            self.DeclareAbstractInputPort(
                "iiwa_plan", AbstractValue.Make(PlanBase()))

        # position and torque command output port
        self.iiwa_position_command_output_port = \
            self.DeclareVectorOutputPort("iiwa_position_command",
                                          BasicVector(self.nu), self._CalcIiwaPositionCommand)
        self.iiwa_torque_command_output_port = \
            self.DeclareVectorOutputPort("iiwa_torque_command",
                                          BasicVector(self.nu), self._CalcIiwaTorqueCommand)

        # Declare command publishing rate
        # state[0:7]: position command
        # state[7:14]: torque command
        # state[14]: gripper_setpoint
        self.DeclareDiscreteState(self.nu*2)
        self.DeclarePeriodicDiscreteUpdate(period_sec=self.control_period)

    def DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        # Call base method to ensure we do not get recursion.
        LeafSystem.DoCalcDiscreteVariableUpdates(self, context, events, discrete_state)

        t= context.get_time()
        self.current_plan = self.EvalAbstractInput(
            context, self.plan_input_port.get_index()).get_value()
        q_iiwa = self.EvalVectorInput(
            context, self.iiwa_position_input_port.get_index()).get_value()
        v_iiwa = self.EvalVectorInput(
            context, self.iiwa_velocity_input_port.get_index()).get_value()
        tau_iiwa = self.EvalVectorInput(
            context, self.iiwa_external_torque_input_port.get_index()).get_value()
        t_plan = t - self.current_plan.start_time

        new_control_output = discrete_state.get_mutable_vector().get_mutable_value()

        new_control_output[0:self.nu] = \
            self.current_plan.CalcPositionCommand(q_iiwa, v_iiwa, tau_iiwa, t_plan, self.control_period)
        new_control_output[self.nu:2*self.nu] = \
            self.current_plan.CalcTorqueCommand(q_iiwa, v_iiwa, tau_iiwa, t_plan, self.control_period)

        # print current simulation time
        if (self.print_period and
                t - self.last_print_time >= self.print_period):
            print "t: ", t
            self.last_print_time = t

    def _CalcIiwaPositionCommand(self, context, y_data):
        state = context.get_discrete_state_vector().get_value()
        y = y_data.get_mutable_value()
        # Get the ith finger control output
        y[:] = state[0:self.nu]

    def _CalcIiwaTorqueCommand(self, context, y_data):
        state = context.get_discrete_state_vector().get_value()
        y = y_data.get_mutable_value()
        # Get the ith finger control output
        y[:] = state[self.nu:2*self.nu]


def CreateManipStationPlanRunnerDiagram(kuka_plans, gripper_setpoint_list, print_period=1.0):
    builder = DiagramBuilder()

    iiwa_controller = IiwaController(print_period=print_period)
    builder.AddSystem(iiwa_controller)
    plan_scheduler = PlanScheduler(kuka_plans, gripper_setpoint_list)
    builder.AddSystem(plan_scheduler)

    builder.Connect(plan_scheduler.iiwa_plan_output_port,
                    iiwa_controller.plan_input_port)

    builder.ExportInput(iiwa_controller.iiwa_position_input_port,
                        "iiwa_position")
    builder.ExportInput(iiwa_controller.iiwa_velocity_input_port,
                        "iiwa_velocity")
    builder.ExportInput(iiwa_controller.iiwa_external_torque_input_port,
                        "iiwa_torque_external")

    builder.ExportOutput(iiwa_controller.iiwa_position_command_output_port,
                         "iiwa_position_command")
    builder.ExportOutput(iiwa_controller.iiwa_torque_command_output_port,
                         "iiwa_torque_command")
    builder.ExportOutput(plan_scheduler.gripper_setpoint_output_port, "gripper_setpoint")
    builder.ExportOutput(plan_scheduler.gripper_force_limit_output_port, "force_limit")

    plan_runner = builder.Build()
    plan_runner.set_name("Plan Runner")

    return plan_runner, plan_scheduler.kPlanDurationMultiplier



