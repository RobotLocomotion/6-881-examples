from pydrake.systems.framework import (
    AbstractValue, BasicVector, LeafSystem, PortDataType
)
from pydrake.multibody.multibody_tree.multibody_plant import ContactResults
from plan_utils import *

plant = station.get_multibody_plant()
iiwa_model = plant.GetModelInstanceByName("iiwa")


class RobotContactDetector(LeafSystem):
    """
    A LeafSystem whose output is the same as contact particle filter (CPF).
    Input: contact results
    Outputs:
        list [contact_info_1, contact_info_2, ...]
        where contact_info_i := (robot_body_index, contact_location, contact_force),
        in which contact_force is acting on the body indexed by robot_body_index.
    """
    def __init__(self):
        LeafSystem.__init__(self)
        self.set_name('robot_contact_detector')
        self._DeclarePeriodicPublish(0.05, 0.0)  # assuming CPF runs at 20Hz.

        # A set of body indices that belong to the robot
        robot_body_indices = plant.GetBodyIndices(iiwa_model)
        self.robot_body_indices = set()
        for body_index in robot_body_indices:
            self.robot_body_indices.add(int(body_index))

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

    def _DoPublish(self, context, event):
        LeafSystem._DoPublish(self, context, event)

        self.contact_info = list()
        contact_results = self.EvalAbstractInput(context, 0).get_value()
        for i_contact in range(contact_results.num_contacts()):
            contact_info_i = contact_results.contact_info(i_contact)

            bodyA_idx = int(contact_info_i.bodyA_index())
            bodyB_idx = int(contact_info_i.bodyB_index())

            if bodyA_idx in self.robot_body_indices:
                robot_body_idx = contact_info_i.bodyA_index()
                a = -1
            elif bodyB_idx in self.robot_body_indices:
                robot_body_idx = contact_info_i.bodyB_index()
                a = 1
            else:
                continue
            self.contact_info.append(
                (robot_body_idx,
                 contact_info_i.contact_point(),
                 contact_info_i.contact_force() * a))

    def _GetContactInfo(self, context, y_data):
        y_data.set_value(self.contact_info)
