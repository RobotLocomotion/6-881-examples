from pydrake.systems.framework import (
    AbstractValue, BasicVector, LeafSystem, PortDataType
)
from pydrake.multibody.multibody_tree.multibody_plant import ContactResults
from pydrake.solvers import mathematicalprogram as mp
from robot_plans import *


plant = station.get_multibody_plant()
iiwa_model = plant.GetModelInstanceByName("iiwa")
gripper_model = plant.GetModelInstanceByName("gripper")
robot_body_indices = plant.GetBodyIndices(iiwa_model)

class RobotContactDetector(LeafSystem):
    """
    A LeafSystem whose output is the same as contact particle filter (CPF).
    Input: contact results
    Outputs:
        list [contact_info_1, contact_info_2, ...]
        where contact_info_i :=
            (robot_body_index, // int
             contact_location, // (3,) numpy array
             contact_force, // (3,) numpy array
             contact_normal), (3,) numpy array
        in which contact_force is acting on the body indexed by robot_body_index.
    """
    def __init__(self, log=False):
        LeafSystem.__init__(self)
        self.set_name('robot_contact_detector')
        self._DeclarePeriodicPublish(0.05, 0.0)  # assuming CPF runs at 20Hz.

        # A set of body indices that belong to the robot

        self.robot_body_indices = set()
        for idx in robot_body_indices:
            self.robot_body_indices.add(int(idx))
        self.gripper_body_idx = int(plant.GetBodyByName('body', gripper_model).index())
        self.link7_idx = int(plant.GetBodyByName('iiwa_link_7').index())
        self.robot_body_indices.add(self.gripper_body_idx)

        # Contact results input port from MultibodyPlant
        self.contact_results_input_port = \
            self._DeclareAbstractInputPort(
                "contact_results", AbstractValue.Make(ContactResults()))

        # contact "info" output port
        self.contact_info_output_port = \
            self._DeclareAbstractOutputPort(
                "contact_info",
                lambda: AbstractValue.Make(list()),
                self._GetContactInfo)

        # system "state"
        self.contact_info = list()

        # logging
        self.log = log
        self.data = list()
        self.sample_times = list()

    def _DoPublish(self, context, event):
        LeafSystem._DoPublish(self, context, event)

        self.contact_info = list()
        contact_results = self.EvalAbstractInput(context, 0).get_value()
        for i_contact in range(contact_results.num_contacts()):
            contact_info_i = contact_results.contact_info(i_contact)

            bodyA_idx = int(contact_info_i.bodyA_index())
            bodyB_idx = int(contact_info_i.bodyB_index())

            if bodyA_idx in self.robot_body_indices:
                robot_body_idx = bodyA_idx
                a = 1
            elif bodyB_idx in self.robot_body_indices:
                robot_body_idx = bodyB_idx
                a = -1
            else:
                continue
            if robot_body_idx == self.gripper_body_idx:
                robot_body_idx = self.link7_idx
            self.contact_info.append(
                (robot_body_idx,
                 contact_info_i.contact_point(),
                 contact_info_i.contact_force() * a,
                 contact_info_i.point_pair().nhat_BA_W * a)
            )

        if self.log:
            self.data.append(self.contact_info)
            self.sample_times.append(context.get_time())

    def _GetContactInfo(self, context, y_data):
        y_data.set_value(self.contact_info)


class JointSpacePlanContact(JointSpacePlan):
    def __init__(self, trajectory):
        JointSpacePlan.__init__(self, trajectory)
        self.plant_iiwa = station.get_controller_plant()
        self.context_iiwa = self.plant_iiwa.CreateDefaultContext()
        self.type = PlanTypes["JointSpacePlanContact"]

        # maps integer body indices in the full ManipulationStation plant to BodyIndex
        # in the controller plant.
        self.robot_frame_map = dict()
        for i, idx in enumerate(robot_body_indices):
            frame_idx = self.plant_iiwa.GetFrameByName("iiwa_link_%i"%i).index()
            self.robot_frame_map[int(idx)] = frame_idx

    def CalcPositionCommand(
            self, q_iiwa, v_iiwa, tau_iiwa, t_plan, control_period, **kwargs):
        contact_info_list = kwargs['contact_info']

        x_iiwa_mutable = \
            self.plant_iiwa.GetMutablePositionsAndVelocities(self.context_iiwa)
        x_iiwa_mutable[:7] = q_iiwa
        x_iiwa_mutable[7:] = v_iiwa

        nc = len(contact_info_list)  # number of contacts
        Jn = np.zeros((nc, 1, 7))  # normal j\acobians
        Jv = np.zeros((nc, 4, 7))  # friction cone jacobians
        Jc = np.zeros((nc, 3, 7)) # "contact" jacobian
        for i, contact_info in enumerate(contact_info_list):
            # unpack contact_info, C denotes the contact point
            robot_body_idx, p_WC, f_C_W, normal = contact_info

            frame_B_idx = self.robot_frame_map[robot_body_idx]
            frame_B = self.plant_iiwa.get_frame(frame_B_idx)

            # pose of frame_B, body frame of the robot link on which the current
            # contact force acts.
            X_WB = self.plant_iiwa.CalcRelativeTransform(
                self.context_iiwa,
                frame_A=self.plant_iiwa.world_frame(),
                frame_B=frame_B)
            p_BC = X_WB.inverse().multiply(p_WC)

            # generating vectors of the tangent plane
            dC = np.zeros((4, 3))
            if np.abs(normal[0]) < 1e-6:
                dC[0] = [0, normal[2], -normal[1]]
            else:
                dC[0] = [normal[1], -normal[0], 0]
            dC[0] /= np.linalg.norm(dC[0])
            dC[1] = np.cross(normal, dC[0])
            dC[2:] = - dC[:2]

            # generating unit vectors of friction cone
            # Let's use a coefficient of friction is 0.3 for all contacts
            vC = 0.3*dC + normal
            for j in range(len(vC)):
                vC[j] /= np.linalg.norm(vC[j])

            # geometric jacobian
            Jg = self.plant_iiwa.CalcFrameGeometricJacobianExpressedInWorld(
                context=self.context_iiwa,
                frame_B=frame_B,
                p_BoFo_B=p_BC)

            Jn[i] = normal.dot(Jg[3:])
            Jv[i] = vC.dot(Jg[3:])
            Jc[i] = Jg[3:]

        prog = mp.MathematicalProgram()
        dq = prog.NewContinuousVariables(7, "dq")   # actual change in joint angles
        dq_d = prog.NewContinuousVariables(7, "dq_d")  # desired change in joint angles
        f = prog.NewContinuousVariables(4*nc, "f")  # contact forces
        f.resize((nc, 4))

        # no penetration
        for i in range(nc):
            prog.AddLinearConstraint(
                Jn[i]/control_period, np.zeros(1), np.full(1, np.inf), dq)

        # joint torque due to joint stiffness
        Kq = 100*np.ones(7)
        tau_k = - Kq * (dq - dq_d)

        # joint torque due to external force
        tau_ext = np.zeros(7, dtype=object)
        for i in range(nc):
            tau_ext += Jv[i].T.dot(f[i])

        # force balance
        for i in range(7):
            prog.AddLinearConstraint(tau_k[i] + tau_ext[i] == 0)

        # objectives
        q_iiwa_next = self.traj.value(t_plan).flatten()
        prog.AddQuadraticCost(((q_iiwa + dq_d - q_iiwa_next)**2).sum())
        f.resize(nc*4)
        prog.AddQuadraticCost(100*(f**2).sum())

        result = prog.Solve()
        assert result == mp.SolutionResult.kSolutionFound

        return q_iiwa + prog.GetSolution(dq_d)






