from pydrake.systems.framework import (
    AbstractValue, BasicVector, LeafSystem, PortDataType
)
from pydrake.multibody.multibody_tree.multibody_plant import ContactResults
from pydrake.solvers import mathematicalprogram as mp
from robot_plans import *

def GetIiwaBodyAndEeIndex():
    """
    :param station: an instance of ManipulationStation.
    :return:
    robot_body_indices: a set of BodyIndex cast as integers, include both robot links
        and the end effector.
    end_effector_indices: a set of BodyIndex cast as integers, include bodies of the
        end effector.
    last_link_index: the integer body index of the body to which the end effector is
        rigidly attahced. For the iiwas it's link7.
    """
    robot_body_indices = set()
    end_effector_indices = set()

    plant = station.get_multibody_plant()
    iiwa_model = plant.GetModelInstanceByName("iiwa")
    gripper_model = plant.GetModelInstanceByName("gripper")
    robot_body_indices_list = plant.GetBodyIndices(iiwa_model)

    # link indices
    for idx in robot_body_indices_list:
        robot_body_indices.add(int(idx))

    # end effector indices
    end_effector_indices.add(int(plant.GetBodyByName('body', gripper_model).index()))
    robot_body_indices |= end_effector_indices

    # last link index
    last_link_index = int(plant.GetBodyByName('iiwa_link_7').index())

    return (robot_body_indices,
            end_effector_indices,
            last_link_index)

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
    def __init__(self, GetBodyIndexSets, log=False):
        LeafSystem.__init__(self)
        self.set_name('robot_contact_detector')
        self._DeclarePeriodicPublish(0.05, 0.0)  # assuming CPF runs at 20Hz.

        # A set of body indices that belong to the robot
        self.robot_body_indices, self.end_effector_indices, self.last_link_index = \
            GetBodyIndexSets()

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
            if robot_body_idx in self.end_effector_indices:
                robot_body_idx = self.last_link_index
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
        plant = station.get_multibody_plant()
        iiwa_model = plant.GetModelInstanceByName("iiwa")
        robot_body_indices_list = plant.GetBodyIndices(iiwa_model)
        self.robot_frame_map = dict()
        for i, idx in enumerate(robot_body_indices_list):
            frame_idx = self.plant_iiwa.GetFrameByName("iiwa_link_%i"%i).index()
            self.robot_frame_map[int(idx)] = frame_idx

        self.nq = self.plant_iiwa.num_positions()
        self.nv = self.plant_iiwa.num_velocities()

    def CalcPositionCommand(
            self, q_iiwa, v_iiwa, tau_iiwa, t_plan, control_period, **kwargs):
        contact_info_list = kwargs['contact_info']

        x_iiwa_mutable = \
            self.plant_iiwa.GetMutablePositionsAndVelocities(self.context_iiwa)
        x_iiwa_mutable[:7] = q_iiwa
        x_iiwa_mutable[7:] = v_iiwa

        q_robot_ref_next = self.traj.value(t_plan).flatten()

        q_robot_cmd = self.CalcQcommanded(
            contact_info_list=contact_info_list,
            nq=self.nq,
            nd=4,
            robot_frame_map=self.robot_frame_map,
            plant_robot=self.plant_iiwa,
            context_robot=self.context_iiwa,
            control_period=control_period,
            q_robot=q_iiwa,
            q_robot_ref_next=q_robot_ref_next)

        return q_robot_cmd

    @staticmethod
    def CalcQcommanded(contact_info_list, nq, nd,
                       robot_frame_map, plant_robot, context_robot, control_period,
                       q_robot, q_robot_ref_next):
        """

        :param contact_info_list:
        :param nq: number of joints of the robot.
        :param nd: number of vectors spanning tangent contact planes, can be 2 or 4.
        :return:
        """
        nc = len(contact_info_list)  # number of contacts
        Jn = np.zeros((nc, 1, nq))  # normal jacobians
        Jv = np.zeros((nc, nd, nq))  # friction cone jacobians
        Jc = np.zeros((nc, 3, nq)) # "contact" jacobian
        for i, contact_info in enumerate(contact_info_list):
            # unpack contact_info, C denotes the contact point
            robot_body_idx, p_WC, f_C_W, normal = contact_info

            frame_B_idx = robot_frame_map[robot_body_idx]
            frame_B = plant_robot.get_frame(frame_B_idx)

            # pose of frame_B, body frame of the robot link on which the current
            # contact force acts.
            X_WB = plant_robot.CalcRelativeTransform(
                context_robot,
                frame_A=plant_robot.world_frame(),
                frame_B=frame_B)
            p_BC = X_WB.inverse().multiply(p_WC)

            # generating vectors of the tangent plane
            dC = np.zeros((nd, 3))
            if np.linalg.norm(normal[:2]) < 1e-6:
                dC[0] = [0, normal[2], -normal[1]]
            else:
                dC[0] = [normal[1], -normal[0], 0]
            dC[0] /= np.linalg.norm(dC[0])

            if nd == 4:
                dC[1] = np.cross(normal, dC[0])
                dC[2:] = - dC[:2]
            elif nd == 2:
                dC[1] = - dC[0]
            else:
                raise NotImplementedError

            # generating unit vectors of friction cone
            # Let's use a coefficient of friction is 0.3 for all contacts
            vC = 0.3*dC + normal
            for j in range(len(vC)):
                vC[j] /= np.linalg.norm(vC[j])

            # geometric jacobian
            Jg = plant_robot.CalcFrameGeometricJacobianExpressedInWorld(
                context=context_robot,
                frame_B=frame_B,
                p_BoFo_B=p_BC)

            Jn[i] = normal.dot(Jg[3:])
            Jv[i] = vC.dot(Jg[3:])
            Jc[i] = Jg[3:]

        prog = mp.MathematicalProgram()
        dq = prog.NewContinuousVariables(nq, "dq")   # actual change in joint angles
        dq_d = prog.NewContinuousVariables(nq, "dq_d")  # desired change in joint angles
        f = prog.NewContinuousVariables(nd*nc, "f")  # contact forces
        f.resize((nc, nd))

        # no penetration
        for i in range(nc):
            prog.AddLinearConstraint(
                Jn[i]/control_period, np.zeros(1), np.full(1, np.inf), dq)

        # joint torque due to joint stiffness
        Kq = 100*np.ones(nq)
        tau_k = - Kq * (dq - dq_d)

        # joint torque due to external force
        tau_ext = np.zeros(nq, dtype=object)
        for i in range(nc):
            tau_ext += Jv[i].T.dot(f[i])

        # force balance
        for i in range(nq):
            prog.AddLinearConstraint(tau_k[i] + tau_ext[i] == 0)

        # objectives
        prog.AddQuadraticCost(((q_robot + dq_d - q_robot_ref_next)**2).sum())
        f.resize(nc*4)
        prog.AddQuadraticCost(10*(f**2).sum())

        result = prog.Solve()
        assert result == mp.SolutionResult.kSolutionFound

        return q_robot + prog.GetSolution(dq_d)





