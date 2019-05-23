import argparse
import numpy as np

from pydrake.examples.manipulation_station import ManipulationStation, _xyz_rpy
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import AbstractValue, DiagramBuilder, LeafSystem
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer, MeshcatPointCloudVisualizer
from pydrake.systems.sensors import PixelType
import pydrake.perception as mut

from perception_tools.file_utils import LoadCameraConfigFile

class PointCloudSynthesis(LeafSystem):

    def __init__(self, transform_dict, default_rgb=[255., 255., 255.]):
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

        self.point_cloud_ports = {}
        self.transform_ports = {}

        self.transform_dict = transform_dict
        self.id_list = self.transform_dict.keys()

        self._default_rgb = np.array(default_rgb)

        output_fields = mut.Fields(mut.BaseField.kXYZs | mut.BaseField.kRGBs)

        for id in self.id_list:
            self.point_cloud_ports[id] = self.DeclareAbstractInputPort(
                "point_cloud_P_{}".format(id),
                AbstractValue.Make(mut.PointCloud(fields=output_fields)))


        self.DeclareAbstractOutputPort("combined_point_cloud_W",
                                        lambda: AbstractValue.Make(
                                            mut.PointCloud(
                                                fields=output_fields)),
                                        self.DoCalcOutput)

    def _AlignPointClouds(self, context):
        points = {}
        colors = {}

        for id in self.id_list:
            point_cloud = self.EvalAbstractInput(
                context, self.point_cloud_ports[id].get_index()).get_value()

            # Make a homogenous version of the points.
            points_h_P = np.vstack((point_cloud.xyzs(),
                                   np.ones((1, point_cloud.xyzs().shape[1]))))

            X_WP = self.transform_dict[id].matrix()
            points[id] = X_WP.dot(points_h_P)[:3, :]

            if point_cloud.has_rgbs():
                colors[id] = point_cloud.rgbs()
            else:
                # Need manual broadcasting.
                colors[id] = np.tile(np.array([self._default_rgb]).T,
                                     (1, points[id].shape[1]))

        # Combine all the points and colors into two arrays.
        scene_points = None
        scene_colors = None

        for id in points:
            if scene_points is None:
                scene_points = points[id]
            else:
                scene_points = np.hstack((points[id], scene_points))

            if scene_colors is None:
                scene_colors = colors[id]
            else:
                scene_colors = np.hstack((colors[id], scene_colors))

        valid_indices = np.logical_not(np.isnan(scene_points))

        scene_points = scene_points[:, valid_indices[0, :]]
        scene_colors = scene_colors[:, valid_indices[0, :]]

        return scene_points, scene_colors

    def DoCalcOutput(self, context, output):
        scene_points, scene_colors = self._AlignPointClouds(context)

        output.get_mutable_value().resize(scene_points.shape[1])
        output.get_mutable_value().mutable_xyzs()[:] = scene_points
        output.get_mutable_value().mutable_rgbs()[:] = scene_colors


def CreateYcbObjectClutter():
    ycb_object_pairs = []

    X_WCracker = _xyz_rpy([0.35, 0.14, 0.09], [0, -1.57, 4])
    ycb_object_pairs.append(
        ("drake/manipulation/models/ycb/sdf/003_cracker_box.sdf", X_WCracker))

    # The sugar box pose.
    X_WSugar = _xyz_rpy([0.28, -0.17, 0.03], [0, 1.57, 3.14])
    ycb_object_pairs.append(
        ("drake/manipulation/models/ycb/sdf/004_sugar_box.sdf", X_WSugar))

    X_WSoup = _xyz_rpy([0.40, -0.07, 0.03], [-1.57, 0, 3.14])
    ycb_object_pairs.append(
        ("drake/manipulation/models/ycb/sdf/005_tomato_soup_can.sdf", X_WSoup))

    # The mustard bottle pose.
    X_WMustard = _xyz_rpy([0.44, -0.16, 0.09], [-1.57, 0, 3.3])
    ycb_object_pairs.append(
        ("drake/manipulation/models/ycb/sdf/006_mustard_bottle.sdf",
         X_WMustard))

    # The potted meat can pose.
    X_WMeat = _xyz_rpy([0.35, -0.32, 0.03], [-1.57, 0, 2.5])
    ycb_object_pairs.append(
        ("drake/manipulation/models/ycb/sdf/010_potted_meat_can.sdf", X_WMeat))

    return ycb_object_pairs


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

    meshcat = builder.AddSystem(MeshcatVisualizer(
        station.get_scene_graph(), zmq_url=args.meshcat,
        open_browser=args.open_browser))
    builder.Connect(station.GetOutputPort("pose_bundle"),
                    meshcat.get_input_port(0))

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

        scene_pc_vis = builder.AddSystem(MeshcatPointCloudVisualizer(
            meshcat, name="point_cloud_{}".format(id)))
        builder.Connect(duts[id].point_cloud_output_port(),
                        scene_pc_vis.GetInputPort("point_cloud_P"))


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

    import cv2
    image_num = 2
    image = station.GetOutputPort("camera_{}_depth_image".format(image_num)).Eval(station_context).data

    # img8 = (image.astype(np.float32)/256.).astype('uint8')
    # rescaled = np.clip(img8*255./6, 0, 255).astype('uint8')

    img8 = (image.astype(np.float32)*127/256.).astype('uint8')
    # rescaled = np.clip(img8*255./6, 0, 255).astype('uint8')

    counts = np.zeros(256)
    for i in range(img8.shape[0]):
        for j in range(img8.shape[1]):
            counts[img8[i, j]] += 1

    print img8


    # ind = np.argpartition(img8.reshape((407040, 1)).flatten(), -4)[-4:]
    # print img8.reshape((407040, 1)).flatten()[ind]

    cv2.imshow("image_{}".format(image_num), img8)
    cv2.waitKey(0)

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
