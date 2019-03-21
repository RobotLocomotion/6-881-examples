import numpy as np

from pydrake.examples.manipulation_station import ManipulationStation, _xyz_rpy
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import AbstractValue, DiagramBuilder, LeafSystem
from pydrake.systems.sensors import PixelType

import pydrake.perception as mut

from perception_tools.load_config_file import LoadConfigFile

class PointCloudSynthesis(LeafSystem):

    def __init__(self, transform_dict, viz=False):
        """
        # TODO(kmuhlrad): make having RGBs optional.
        # TODO(kmuhlrad): remove viz with MeshcatPointCloudVisualizer
        A system that takes in N point clouds and N Isometry3 transforms that
        put each point cloud in world frame. The system returns one point cloud
        combining all of the transformed point clouds. Each point cloud must
        have XYZs and RGBs.

        @param transform_dict dict. A map from point cloud IDs to transforms to
            put the point cloud of that ID in world frame.
        @param viz bool. If True, save the combined point clouds
            as serialized numpy arrays.

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

        self.viz = viz

        self.point_cloud_ports = {}
        self.transform_ports = {}

        self.transform_dict = transform_dict
        self.id_list = self.transform_dict.keys()

        for id in self.id_list:
            self.point_cloud_ports[id] = self._DeclareAbstractInputPort(
                "point_cloud_P_{}".format(id),
                AbstractValue.Make(mut.PointCloud()))

        output_fields = mut.Fields(mut.BaseField.kXYZs | mut.BaseField.kRGBs)
        self._DeclareAbstractOutputPort("combined_point_cloud_W",
                                        lambda: AbstractValue.Make(
                                            mut.PointCloud(
                                                fields=output_fields)),
                                        self._DoCalcOutput)

    def _AlignPointClouds(self, context):
        points = {}
        colors = {}

        for id in self.id_list:
            point_cloud = self.EvalAbstractInput(
                context, self.point_cloud_ports[id].get_index()).get_value()

            colors[id] = point_cloud.rgbs()

            # Make a homogenous version of the points.
            points_h_P = np.vstack((point_cloud.xyzs(),
                                   np.ones((1, point_cloud.xyzs().shape[1]))))

            X_WP = self.transform_dict[id]
            points[id] = X_WP.dot(points_h_P)[:3, :]

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

    def _DoCalcOutput(self, context, output):
        scene_points, scene_colors = self._AlignPointClouds(context)

        if self.viz:
            np.save("scene_points", scene_points)
            np.save("scene_colors", scene_colors)

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
    builder = DiagramBuilder()

    # Create the ManipulationStation.
    station = builder.AddSystem(ManipulationStation())
    station.SetupDefaultStation()
    ycb_objects = CreateYcbObjectClutter()
    for model_file, X_WObject in ycb_objects:
        station.AddManipulandFromFile(model_file, X_WObject)
    station.Finalize()

    id_list = station.get_camera_names()

    camera_configs = LoadConfigFile(
        "/home/amazon/6-881-examples/perception/config/sim.yml")

    transform_dict = {}
    for id in id_list:
        transform_dict[id] = camera_configs[id]["camera_pose_world"].dot(
            camera_configs[id]["camera_pose_internal"])

    # Create the PointCloudSynthesis system.
    pc_synth = builder.AddSystem(PointCloudSynthesis(transform_dict, True))

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

    # build diagram
    diagram = builder.Build()

    # construct simulator
    simulator = Simulator(diagram)

    context = diagram.GetMutableSubsystemContext(
        pc_synth, simulator.get_mutable_context())

    pc = pc_synth.GetOutputPort("combined_point_cloud_W").Eval(context)
    print pc.size()