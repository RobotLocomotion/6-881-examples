from pydrake.trajectories import PiecewisePolynomial
import numpy as np
from pydrake.util.eigen_geometry import Isometry3, Quaternion
from pydrake.examples.manipulation_station import ManipulationStation
import matplotlib.pyplot as plt


# Create a cubic polynomial that connects x_start and x_end.
# x_start and x_end should be list or np arrays.
def ConnectPointsWithCubicPolynomial(x_start, x_end, duration):
    t_knots = [0, duration / 2, duration]
    n = len(x_start)
    assert n == len(x_end)
    x_knots = np.zeros((3, n))
    x_knots[0] = x_start
    x_knots[2] = x_end
    x_knots[1] = (x_knots[0] + x_knots[2]) / 2
    return  PiecewisePolynomial.Cubic(
        t_knots, x_knots.T, np.zeros(n), np.zeros(n))

'''
Create an instance of ManipulationStation for kinematic and dynamic calculations. 
'''
station = ManipulationStation()
station.Finalize()


'''
Ea, or End_Effector_world_aligned is a frame fixed w.r.t the gripper.
Ea has the same origin as the end effector's body frame, but
its axes are aligned with those of the world frame when the system
has zero state, i.e. the robot is upright with all joint angles
equal to zero.
This frame is defined so that it is convenient to define end effector orientation
relative to the world frame using RollPitchYaw.
'''
def GetEndEffectorWorldAlignedFrame():
    X_EEa = Isometry3.Identity()
    X_EEa.set_rotation(np.array([[0., 1., 0,],
                                 [0, 0, 1],
                                 [1, 0, 0]]))
    return X_EEa

X_EEa = GetEndEffectorWorldAlignedFrame()

'''
get relative transforms between EE frame (wsg gripper) and iiwa_link_7
'''
def GetL7EeTransform():
    plant = station.get_mutable_multibody_plant()
    tree = plant.tree()

    context_plant = plant.CreateDefaultContext()
    X_L7E = tree.CalcRelativeTransform(
        context_plant,
        frame_A=plant.GetFrameByName("iiwa_link_7"),
        frame_B=plant.GetFrameByName("body"))

    return X_L7E

'''
Plots iiwa_external_torque from its signal logger system. 
'''
def PlotExternalTorqueLog(iiwa_external_torque_log):
    fig_external_torque = plt.figure(figsize=(8, 18), dpi=150)
    t = iiwa_external_torque_log.sample_times()
    for i, torque in enumerate(iiwa_external_torque_log.data()):
        ax = fig_external_torque.add_subplot(711 + i)
        ax.plot(t, torque, label='torque_external@joint_%d' % (i + 1))
        ax.set_xlabel("t(s)")
        ax.set_ylabel("N/m")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

'''
Plots iiwa_position from signal logger systems. 
'''
def PlotIiwaPositionLog(iiwa_position_command_log, iiwa_position_measured_log):
    fig = plt.figure(figsize=(8, 18), dpi=150)
    t = iiwa_position_command_log.sample_times()
    for i in range(len(iiwa_position_command_log.data())):
        ax = fig.add_subplot(711 + i)
        q_commanded = iiwa_position_command_log.data()[i]
        q_measured = iiwa_position_measured_log.data()[i]
        ax.plot(t, q_commanded/np.pi*180, label='q_commanded@joint_%d' % (i + 1))
        ax.plot(t, q_measured/np.pi*180, label='q_measrued@joint_%d' % (i + 1))
        ax.set_xlabel("t(s)")
        ax.set_ylabel("degrees")
        # ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def GetPlanStartingTimes(kuka_plans):
    """
    :param kuka_plans: a list of Plans.
    :return: t_plan is a list of length (len(kuka_plans) + 1). t_plan[i] is the starting time of kuka_plans[i];
        t_plan[-1] is the time at which the last plan ends.
    """
    num_plans = len(kuka_plans)
    t_plan = np.zeros(num_plans + 1)
    for i in range(0, num_plans):
        t_plan[i + 1] = \
            t_plan[i] + kuka_plans[i].get_duration()
    print "Plan starting times(s)\n", t_plan
    return t_plan

def RenderSystemWithGraphviz(system, output_file="system_view.gz"):
    """ Renders the Drake system (presumably a diagram,
    otherwise this graph will be fairly trivial) using
    graphviz to a specified file. """
    from graphviz import Source
    string = system.GetGraphvizString()
    src = Source(string)
    src.render(output_file, view=False)