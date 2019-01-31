from plan_runner.planar_arm_2link_simulator import *
from pydrake.trajectories import PiecewisePolynomial

if __name__ == "__main__":
    sim = PlanarArmSimulator()
    t_knots = np.array([0,2])
    q_knots = np.array([[0., 0], [1, 0]])
    traj = PiecewisePolynomial.Cubic(t_knots, q_knots.T, np.zeros(2), np.zeros(2))

    sim.RunSimulation(traj)
