from plan_runner.planar_arm_2link_simulator import *
from pydrake.trajectories import PiecewisePolynomial
import matplotlib.pyplot as plt


def PlotQLog(q_cmd_log, q_log, q_traj_ref):
    n = len(q_cmd_log.data())
    fig = plt.figure(figsize=(8, 2.5*n), dpi=150)
    t = q_cmd_log.sample_times()
    q_ref = np.zeros((n, len(t)))
    for i, ti in enumerate(t):
        q_ref[:, i] = q_traj_ref.value(ti).ravel()

    for i in range(n):
        ax = fig.add_subplot(100*n + 11 + i)
        q_cmd_i = q_cmd_log.data()[i]
        q_i = q_log.data()[i]
        ax.plot(t, q_cmd_i/np.pi*180, label='q_cmd%d' % (i + 1))
        ax.plot(t, q_i/np.pi*180, label='q%d' % (i + 1))
        ax.plot(t, q_ref[i]/np.pi*180, label='q_ref%d' % (i+1))
        ax.set_xlabel("t(s)")
        ax.set_ylabel("degrees")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sim = PlanarArmSimulator()
    t_knots = np.array([0,2])
    q_knots = np.array([[1, 0], [np.pi/4, -np.pi/3 - np.pi/4]])
    traj = PiecewisePolynomial.Cubic(t_knots, q_knots.T, np.zeros(2), np.zeros(2))

    sim.RunSimulation(traj)
    PlotQLog(sim.q_cmd_log, sim.q_log, traj)

