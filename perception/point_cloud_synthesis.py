import numpy as np
import yaml

from pydrake.systems.framework import AbstractValue, LeafSystem
from pydrake.common.eigen_geometry import Isometry3, AngleAxis
from pydrake.systems.sensors import CameraInfo, ImageRgba8U
import pydrake.perception as mut

import meshcat.transformations as tf

class PointCloudSynthesis(LeafSystem):

    def __init__(self, config_file, viz=False):
        """
        # TODO(kmuhlrad): make having RGBs optional.
        A system that takes in N point clouds and N Isometry3 transforms that
        put each point cloud in world frame. The system returns one point cloud
        combining all of the transformed point clouds. Each point cloud must
        have XYZs and RGBs.

        @param config_file str. A path to a .yml configuration file for the
            cameras.
        @param viz bool. If True, save the combined point clouds
            as serialized numpy arrays.

        @system{
          @input_port{point_cloud_id0}
          @input_port{X_WPC_id0}
          .
          .
          .
          @input_port{point_cloud_idN}
          @input_port{X_WPC_idN}
          @output_port{combined_point_cloud}
        }
        """
        LeafSystem.__init__(self)

        self._LoadConfigFile(config_file)
        self.viz = viz

        self.input_ports = {}

        for id in self.camera_configs:
            self.input_ports[id] = self._DeclareAbstractInputPort(
                "point_cloud_" + id, AbstractValue.Make(mut.PointCloud()))

        output_fields = mut.Fields(mut.BaseField.kXYZs | mut.BaseField.kRGBs)
        self._DeclareAbstractOutputPort("combined_point_cloud",
                                        lambda: AbstractValue.Make(
                                            mut.PointCloud(
                                                fields=output_fields)),
                                        self._DoCalcOutput)

    def _LoadConfigFile(self, config_file):
        with open(config_file, 'r') as stream:
            try:
                config = yaml.load(stream)
                self.camera_configs = {}
                for camera in config:
                    serial_no, X_WCamera, X_CameraDepth, camera_info = \
                        self._ParseCameraConfig(config[camera])
                    id = str(serial_no)
                    self.camera_configs[id] = {}
                    self.camera_configs[id]["camera_pose_world"] = X_WCamera
                    self.camera_configs[id]["camera_pose_internal"] = \
                        X_CameraDepth
                    self.camera_configs[id]["camera_info"] = camera_info
            except yaml.YAMLError as exc:
                print "could not parse config file"
                print exc

    def _ParseCameraConfig(self, camera_config):
        # extract serial number
        serial_no = camera_config["serial_no"]

        # construct the transformation matrix
        world_transform = camera_config["world_transform"]
        X_WCamera = tf.euler_matrix(world_transform["roll"],
                                    world_transform["pitch"],
                                    world_transform["yaw"])
        X_WCamera[:3, 3] = \
            [world_transform["x"], world_transform["y"], world_transform["z"]]

        # construct the transformation matrix
        internal_transform = camera_config["internal_transform"]
        X_CameraDepth = tf.euler_matrix(internal_transform["roll"],
                                        internal_transform["pitch"],
                                        internal_transform["yaw"])
        X_CameraDepth[:3, 3] = ([internal_transform["x"],
                                 internal_transform["y"],
                                 internal_transform["z"]])

        # construct the camera info
        camera_info_data = camera_config["camera_info"]
        if "fov_y" in camera_info_data:
            camera_info = CameraInfo(camera_info_data["width"],
                                     camera_info_data["height"],
                                     camera_info_data["fov_y"])
        else:
            camera_info = CameraInfo(
                camera_info_data["width"], camera_info_data["height"],
                camera_info_data["focal_x"], camera_info_data["focal_y"],
                camera_info_data["center_x"], camera_info_data["center_y"])

        return serial_no, X_WCamera, X_CameraDepth, camera_info

    def _AlignPointClouds(self, context):
        points = {}
        colors = {}

        for id in self.camera_configs:
            point_cloud = self.EvalAbstractInput(
                context, self.input_ports[id].get_index()).get_value()

            colors[id] = point_cloud.rgbs()

            # Make a homogenous version of the points.
            points_h = np.vstack((point_cloud.xyzs(),
                                  np.ones((1, point_cloud.xyzs().shape[1]))))

            X_WDepth = self.camera_configs[id]["camera_pose_world"].dot(
                self.camera_configs[id]["camera_pose_internal"])
            points_transformed = X_WDepth.dot(points_h)

            points[id] = points_transformed[:3, :]

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