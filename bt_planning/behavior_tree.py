import argparse
import numpy as np

from pydrake.examples.manipulation_station import ManipulationStation, _xyz_rpy
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import AbstractValue, DiagramBuilder, LeafSystem, BasicVector
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer, MeshcatPointCloudVisualizer
from pydrake.systems.sensors import PixelType
import pydrake.perception as mut
from pydrake.systems.rendering import PoseBundle

from perception_tools.file_utils import LoadCameraConfigFile

class BehaviorTree(LeafSystem):

    def __init__(self, root):
        """
        A system that takes in N point clouds and N RigidTransforms that
        put each point cloud in world frame. The system returns one point cloud
        combining all of the transformed point clouds. Each point cloud must
        have XYZs. RGBs are optional. If absent, those points will be white.

        @param transform_dict dict. A map from point cloud IDs to RigidTransforms
            to put the point cloud of that ID in world frame.
        @param default_rgb list. A list containing the RGB values to use in the
            absence of PointCloud.rgbs. Values should be between 0. and 255.
            The default is white.

        @system{
          @input_port{point_cloud_id0}
          .
          .
          .
          @input_port{point_cloud_P_idN}
          @output_port{combined_point_cloud_W}
        }
        """
        LeafSystem.__init__(self)

        self.root = root
        self.root.setup(timeout=15)

        # TODO(kmuhlrad): figure out size of pose_bundle
        self.pose_bundle_port = self.DeclareAbstractInputPort(
            "pose_bundle_W", AbstractValue.Make(PoseBundle()))

        # iiwa position input port
        self.iiwa_position_input_port = \
            self.DeclareVectorInputPort(
                "iiwa_position", BasicVector(7))

        # iiwa velocity input port
        self.iiwa_velocity_input_port = \
            self.DeclareVectorInputPort(
                "iiwa_velocity", BasicVector(7))

        self.gripper_position_input_port = \
            self.DeclareVectorInputPort(
                "gripper_position", BasicVector(1))

        self.gripper_force_input_port = \
            self.DeclareVectorInputPort(
                "gripper_force", BasicVector(1))

        # TODO(kmuhlrad): add output ports

    def UpdateBlackboard(self, context):
        """
        Evaluate all the input ports and update the blackboard accordingly.
        This is so conditions can be checked only by reading the blackboard
        and don't need to know about ports at all. This method should always
        be called before ticking the tree.

        :param context:
        :return:
        """

    def Tick(self, context, output):
        self.UpdateBlackboard(context)

        # Also use the blackboard to update the output port? So for example in
        # a Place behavior, it would write the controller type and data to
        # the blackboard, and then that would be read and set to this output
        # port?

        self.root.tick_once()

        # output.get_mutable_value()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera_config_file",
        required=True,
        help="The path to a camera configuration .yml file")
    MeshcatVisualizer.add_argparse_argument(parser)
    args = parser.parse_args()

    builder = DiagramBuilder()

    # Create the ManipulationStation.
    station = builder.AddSystem(ManipulationStation())
    station.SetupDefaultStation()
    ycb_objects = CreateYcbObjectClutter()
    for model_file, X_WObject in ycb_objects:
        station.AddManipulandFromFile(model_file, X_WObject)
    station.Finalize()

    id_list = station.get_camera_names()

    camera_configs = LoadCameraConfigFile(args.camera_config_file)

    transform_dict = {}
    for id in id_list:
        transform_dict[id] = camera_configs[id]["camera_pose_world"].multiply(
            camera_configs[id]["camera_pose_internal"])

    # Create the PointCloudSynthesis system.
    pc_synth = builder.AddSystem(PointCloudSynthesis(transform_dict))

    # Create the duts.
    # use scale factor of 1/1000 to convert mm to m
    duts = {}
    for id in id_list:
        duts[id] = builder.AddSystem(mut.DepthImageToPointCloud(
            camera_configs[id]["camera_info"], PixelType.kDepth16U, 1e-3,
            fields=mut.BaseField.kXYZs | mut.BaseField.kRGBs))

        builder.Connect(
            station.GetOutputPort("camera_{}_rgb_image".format(id)),
            duts[id].color_image_input_port())
        builder.Connect(
            station.GetOutputPort("camera_{}_depth_image".format(id)),
            duts[id].depth_image_input_port())

        builder.Connect(duts[id].point_cloud_output_port(),
                        pc_synth.GetInputPort("point_cloud_P_{}".format(id)))

    meshcat = builder.AddSystem(MeshcatVisualizer(
        station.get_scene_graph(), zmq_url=args.meshcat,
        open_browser=args.open_browser))
    builder.Connect(station.GetOutputPort("pose_bundle"),
                    meshcat.get_input_port(0))

    scene_pc_vis = builder.AddSystem(MeshcatPointCloudVisualizer(
        meshcat, name="scene_point_cloud"))
    builder.Connect(pc_synth.GetOutputPort("combined_point_cloud_W"),
                    scene_pc_vis.GetInputPort("point_cloud_P"))

    # build diagram
    diagram = builder.Build()

    # construct simulator
    simulator = Simulator(diagram)

    context = diagram.GetMutableSubsystemContext(
        pc_synth, simulator.get_mutable_context())

    station_context = diagram.GetMutableSubsystemContext(
        station, simulator.get_mutable_context())

    pc = pc_synth.GetOutputPort("combined_point_cloud_W").Eval(context)

    q0 = station.GetIiwaPosition(station_context)
    station_context.FixInputPort(station.GetInputPort(
        "iiwa_position").get_index(), q0)
    station_context.FixInputPort(station.GetInputPort(
        "iiwa_feedforward_torque").get_index(), np.zeros(7))
    station_context.FixInputPort(station.GetInputPort(
        "wsg_position").get_index(), np.array([0.1]))
    station_context.FixInputPort(station.GetInputPort(
        "wsg_force_limit").get_index(), np.array([40.0]))

    simulator.set_publish_every_time_step(False)
    simulator.set_target_realtime_rate(1.0)
    simulator.StepTo(0.1)
