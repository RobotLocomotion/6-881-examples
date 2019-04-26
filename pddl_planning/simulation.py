from __future__ import print_function

import time
import numpy as np
from pydrake.trajectories import PiecewisePolynomial

from pddl_planning.utils import get_configuration, user_input


class Control(object):
    def polynomial(self):
        raise NotImplementedError()
    def plan(self):
        from manipulation_station_plan_runner.robot_plans import JointSpacePlan
        return JointSpacePlan(self.polynomial())


class ForceControl(object):
    def __init__(self, open_fully=True):
        self.open_fully = open_fully
    def plan(self):
        raise NotImplementedError()


class HoldControl(Control):
    def __init__(self, q_knot, duration):
        self.q_knot = q_knot
        self.duration = duration
    def polynomial(self):
        q_knots_kuka = np.zeros((2, len(self.q_knot)))
        q_knots_kuka[0] = self.q_knot
        return PiecewisePolynomial.ZeroOrderHold([0, self.duration], q_knots_kuka.T)


class PositionControl(Control):
    def __init__(self, q_knots, t_knots):
        self.q_knots = q_knots
        self.t_knots = t_knots
    def polynomial(self):
        d, n = self.q_knots.shape
        return PiecewisePolynomial.Cubic(
            breaks=self.t_knots,
            knots=self.q_knots,
            knot_dot_start=np.zeros(d),
            knot_dot_end=np.zeros(d))

##################################################

def step_trajectories(diagram, diagram_context, plant_context, trajectories, time_step=0.001, teleport=False):
    diagram.Publish(diagram_context)
    user_input('Step?')
    for traj in trajectories:
        if teleport:
            traj.path = traj.path[::len(traj.path)-1]
        for _ in traj.iterate(plant_context):
            diagram.Publish(diagram_context)
            if time_step is None:
                user_input('Continue?')
            else:
                time.sleep(time_step)
    user_input('Finish?')

##################################################


def get_hold_spline(mbp, context, robot, duration=1.0):
    return HoldControl(get_configuration(mbp, context, robot), duration)


def get_gripper_setpoint(mbp, context, gripper):
    lower, upper = get_configuration(mbp, context, gripper)
    return upper - lower


def convert_controls(mbp, robot, gripper, context, trajectories):
    splines = [
        get_hold_spline(mbp, context, robot),
    ]
    gripper_setpoints = [
        get_gripper_setpoint(mbp, context, gripper),
    ]
    for traj in trajectories:
        traj.path[-1].assign(context)
        gripper_setpoints.append(get_gripper_setpoint(mbp, context, gripper))
        if traj.force_control:
            splines.append(ForceControl())
            continue
        if len(traj.joints) == 8:
            traj.joints.pop()

        if len(traj.joints) == 2:
            splines.append(get_hold_spline(mbp, context, robot))
        elif len(traj.joints) == 7:
            splines.append(PositionControl(*traj.retime()))
        else:
            raise ValueError('Invalid number of joints: {}'.format(len(traj.joints)))
    return splines, gripper_setpoints


def dump_plans(plan_list, gripper_setpoints):
    print()
    assert len(plan_list) == len(gripper_setpoints)
    for i, (plan, setpoint) in enumerate(zip(plan_list, gripper_setpoints)):
        if plan.traj is not None:
            d = plan.traj.rows()
            n = plan.traj.get_number_of_segments()
        else:
            d, n = None, None
        print('{}) d={}, n={}, setpoint={}, duration={:.3f}'.format(
            i, d, n, setpoint, plan.get_duration()))


def compute_duration(plan_list, multiplier=1.1, extra_time=5.0):
    sim_duration = 0.0
    for plan in plan_list:
        sim_duration += plan.get_duration() * multiplier
    sim_duration += extra_time
    return sim_duration
