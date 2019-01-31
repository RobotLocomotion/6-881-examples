import numpy as np
import os

from pydrake.multibody.parsing import Parser
from pydrake.multibody.multibody_tree import (UniformGravityFieldElement,
                                              MultibodyForces)
from pydrake.multibody.multibody_tree.multibody_plant import MultibodyPlant
from pydrake.systems.analysis import Simulator
from pydrake.common.eigen_geometry import Isometry3
from pydrake.math import RollPitchYaw, RotationMatrix
from pydrake.systems.framework import (
    AbstractValue, BasicVector, LeafSystem, PortDataType, DiagramBuilder
)
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph,
)
# from plan_runner.plan_utils import RenderSystemWithGraphviz
from pydrake.systems.primitives import LogOutput
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer

# length of robot limbs
l1 = 1.0
l2 = 1.0

def ForwardKinematics2Link(q):
    from numpy import sin, cos
    x = l1 * cos(q[0]) + l2 * cos(q[0] + q[1])
    y = l1 * sin(q[0]) + l2 * sin(q[0] + q[1])
    theta = q.sum()
    while theta <= -np.pi:
        theta += np.pi
    while theta >= np.pi:
        theta -= np.pi
    return np.array([x, y, theta])

def Jacobian2Link(q):
    from numpy import sin, cos
    J = np.ones((3, 2))
    J[0, 0] = -l1 * sin(q[0]) - l2 * sin(q[0] + q[1])
    J[0, 1] = - l2 * sin(q[0] + q[1])
    J[1, 0] = l1 * cos(q[0]) + l2 * cos(q[0] + q[1])
    J[1, 1] = l2 * cos(q[0] + q[1])

    return J


def ForwardKinematicsMbp(x, frame, mbp):
    assert len(x) == mbp.num_positions() + mbp.num_velocities()
    context = mbp.CreateDefaultContext()
    tree = mbp.tree()
    x_mutalbe = tree.get_mutable_multibody_state_vector(context)
    x_mutalbe[:] = x
    world_frame = mbp.world_frame()
    X_WE = tree.CalcRelativeTransform(
        context, frame_A=world_frame, frame_B=frame)
    return X_WE


class RobotInverseDynamicsController(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self.set_name('robot_inverse_dynamics_controller')
        self.plant = plant
        self.context = plant.CreateDefaultContext()

        self.nq = plant.num_positions()
        self.nv = plant.num_velocities()
        self.robot_state_input_port = \
            self._DeclareInputPort(
                "robot_state", PortDataType.kVectorValued, self.nq + self.nv)
        self.joint_angle_commanded_input_port = \
            self._DeclareInputPort(
                "q_robot_commanded", PortDataType.kVectorValued, self.nq)
        self.joint_torque_output_port = \
            self._DeclareVectorOutputPort(
                "joint_torques", BasicVector(self.nv), self._CalcJointTorques)

        # control rate
        self.control_period = 0.002
        self._DeclareDiscreteState(self.nv)
        self._DeclarePeriodicDiscreteUpdate(period_sec=self.control_period)
        self.q_cmd_prev = None

    def _DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        LeafSystem._DoCalcDiscreteVariableUpdates(self, context, events, discrete_state)
        # read input ports
        x = self.EvalVectorInput(
            context, self.robot_state_input_port.get_index()).get_value()
        q_cmd = self.EvalVectorInput(
            context,
            self.joint_angle_commanded_input_port.get_index()).get_value()
        q = x[:self.nq]
        v = x[self.nq:]

        # estimate velocity
        if self.q_cmd_prev is None:
            v_cmd = np.zeros(self.nv)
            self.q_cmd_prev = q_cmd
        else:
            v_cmd = (q_cmd - self.q_cmd_prev)/self.control_period

        # compute desired acceleration
        Kp = 100.
        Kv = 10.
        vDt_desired = Kp * (q_cmd - q) + Kv * (v_cmd - v)

        # inverse dynamics calculation
        x_mutable = \
            self.plant.GetMutablePositionsAndVelocities(self.context)
        x_mutable[:] = x
        tau = self.plant.CalcInverseDynamics(
            context=self.context,
            known_vdot=vDt_desired,
            external_forces=MultibodyForces(self.plant))

        output = discrete_state.get_mutable_vector().get_mutable_value()
        output[:] = tau

    def _CalcJointTorques(self, context, y_data):
        state = context.get_discrete_state_vector().get_value()
        y = y_data.get_mutable_value()
        y[:] = state


class RobotTrajRunner(LeafSystem):
    def __init__(self, plant, trajectory=None):
        LeafSystem.__init__(self)
        self.set_name('robot_trajectory_runner')

        self.plant = plant
        self.traj = trajectory

        self.nq = plant.num_positions()
        self.nv = plant.num_velocities()

        self.q_cmd_output_port = \
            self._DeclareVectorOutputPort(
                "q_robot_commanded", BasicVector(self.nv), self._CalcQcmd)

        # control rate
        self.control_period = 0.005
        self._DeclareDiscreteState(self.nv)
        self._DeclarePeriodicDiscreteUpdate(period_sec=self.control_period)
        self.q_cmd_prev = None

    def _DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        t = context.get_time()
        ooutput = discrete_state.get_mutable_vector().get_mutable_value()
        ooutput[:] = self.traj.value(t).flatten()

    def _CalcQcmd(self, context, y_data):
        state = context.get_discrete_state_vector().get_value()
        y = y_data.get_mutable_value()
        y[:] = state


class PlanarArmSimulator:
    def __init__(self):
        builder = DiagramBuilder()

        # MultibodyPlant
        plant = MultibodyPlant(2e-3)
        self.plant = plant
        _, scene_graph = AddMultibodyPlantSceneGraph(builder, plant=plant)
        parser = Parser(plant=plant, scene_graph=scene_graph)

        robot_sdf_path = os.path.join(
            os.getcwd(), "models", "two_link_arm.sdf")
        robot_model = parser.AddModelFromFile(robot_sdf_path)
        plant.AddForceElement(UniformGravityFieldElement())
        plant.Finalize()

        # InverseDynamicsController
        controller = RobotInverseDynamicsController(plant=plant)
        builder.AddSystem(controller)
        builder.Connect(controller.GetOutputPort("joint_torques"),
                        plant.get_actuation_input_port())
        builder.Connect(plant.get_continuous_state_output_port(),
                        controller.robot_state_input_port)

        # TrajRunner
        traj_runner = builder.AddSystem(RobotTrajRunner(plant=plant))
        builder.Connect(traj_runner.GetOutputPort('q_robot_commanded'),
                        controller.joint_angle_commanded_input_port)
        # builder.Connect(plant.get_contact_results_output_port(),
        #                 controller.contact_results_input_port)
        self.traj_runner = traj_runner

        # MeshcatVisualizer
        viz = MeshcatVisualizer(
            scene_graph,
            draw_contact_force=True,
            plant=self.plant)
        builder.AddSystem(viz)
        builder.Connect(scene_graph.get_pose_bundle_output_port(),
                        viz.GetInputPort("lcm_visualization"))
        builder.Connect(plant.GetOutputPort("contact_results"),
                        viz.GetInputPort("contact_results"))

        diagram = builder.Build()
        # RenderSystemWithGraphviz(diagram)

        # Loggers
        self.state_log = LogOutput(plant.get_continuous_state_output_port(), builder)
        self.state_log._DeclarePeriodicPublish(0.02)

        # Create simulation diagram context
        diagram_context = diagram.CreateDefaultContext()
        self.mbp_context = diagram.GetMutableSubsystemContext(
            plant, diagram_context)

        self.simulator = Simulator(diagram, diagram_context)
        self.simulator.set_publish_every_time_step(False)
        self.simulator.set_target_realtime_rate(1.0)

    def RunSimulation(self, traj):
        q0 = traj.value(0)
        for i, joint_angle in enumerate(q0):
            joint = self.plant.GetJointByName("joint_%d" % (i + 1))
            joint.set_angle(context=self.mbp_context, angle=joint_angle)

        self.traj_runner.traj = traj
        self.simulator.Initialize()
        self.simulator.StepTo(traj.end_time() * 1.1)

        return self.state_log





