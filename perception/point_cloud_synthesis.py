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
        # TODO(kmuhlrad): update this if it becomes a thing
        A system that takes in 3 Drake PointClouds and ImageRgba8U from
        RGBDCameras and determines the pose of an object in them. The user must
        supply a segmentation function and pose alignment to determine the pose.
        If these functions aren't supplied, the returned pose will always be the
        4x4 identity matrix.

        @param config_file str. A path to a .yml configuration file for the
            cameras.
        @param viz bool. If True, save the combined point clouds
            as serialized numpy arrays.

        @system{
          @input_port{left_point_cloud},
          @input_port{middle_point_cloud},
          @input_port{right_point_cloud},
          @output_port{combined_point_cloud}
        }
        """
        LeafSystem.__init__(self)

        # fields=mut.BaseFields.kXYZs | mut.BaseFields.kRGBs
        self.left_depth = self._DeclareAbstractInputPort(
            "left_point_cloud", AbstractValue.Make(mut.PointCloud()))
        self.middle_depth = self._DeclareAbstractInputPort(
            "middle_point_cloud", AbstractValue.Make(mut.PointCloud()))
        self.right_depth = self._DeclareAbstractInputPort(
            "right_point_cloud", AbstractValue.Make(mut.PointCloud()))

        self._DeclareAbstractOutputPort("combined_point_cloud",
                                        lambda: AbstractValue.Make(
                                            mut.PointCloud(
                                                fields=mut.Fields(
                                                    mut.BaseField.kXYZs | mut.BaseField.kRGBs))),
                                        self._DoCalcOutput)

        self._LoadConfigFile(config_file)

        self.viz = viz

    def _LoadConfigFile(self, config_file):
        with open(config_file, 'r') as stream:
            try:
                config = yaml.load(stream)
                self.camera_configs = {}
                for camera in config:
                    serial_no, X_WCamera, X_CameraDepth, camera_info = \
                        self._ParseCameraConfig(config[camera])
                    self.camera_configs[camera + "_serial"] = str(serial_no)
                    self.camera_configs[camera + "_pose_world"] = X_WCamera
                    self.camera_configs[camera + "_pose_internal"] = \
                        X_CameraDepth
                    self.camera_configs[camera + "_info"] = camera_info
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


    def _ExtractPointCloud(self, context):
        left_point_cloud = self.EvalAbstractInput(
            context, self.left_depth.get_index()).get_value()
        middle_point_cloud = self.EvalAbstractInput(
            context, self.middle_depth.get_index()).get_value()
        right_point_cloud = self.EvalAbstractInput(
            context, self.right_depth.get_index()).get_value()

        left_points = np.array(left_point_cloud.xyzs())
        left_colors = np.array(left_point_cloud.rgbs())

        middle_points = np.array(middle_point_cloud.xyzs())
        middle_colors = np.array(middle_point_cloud.rgbs())

        right_points = np.array(right_point_cloud.xyzs())
        right_colors = np.array(right_point_cloud.rgbs())

        if self.viz:
            np.save("left_points", left_points.T)
            np.save("left_colors", left_colors.T)
            np.save("middle_points", middle_points.T)
            np.save("middle_colors", middle_colors.T)
            np.save("right_points", right_points.T)
            np.save("right_colors", right_colors.T)

        return self._AlignPointClouds(left_points,
                                      left_colors,
                                      middle_points,
                                      middle_colors,
                                      right_points,
                                      right_colors)

    def _AlignPointClouds(self, left_points, left_colors,
                          middle_points, middle_colors,
                          right_points, right_colors):
        lh = np.ones((4, left_points.shape[1]))
        lh[:3, :] = np.copy(left_points)

        X_WDepth = self.camera_configs["left_camera_pose_world"].dot(
            self.camera_configs["left_camera_pose_internal"])
        left_points_transformed = X_WDepth.dot(lh)

        mh = np.ones((4, middle_points.shape[1]))
        mh[:3, :] = np.copy(middle_points)

        X_WDepth = self.camera_configs["middle_camera_pose_world"].dot(
            self.camera_configs["middle_camera_pose_internal"])
        middle_points_transformed = X_WDepth.dot(mh)

        rh = np.ones((4, right_points.shape[1]))
        rh[:3, :] = np.copy(right_points)

        X_WDepth = self.camera_configs["right_camera_pose_world"].dot(
            self.camera_configs["right_camera_pose_internal"])
        right_points_transformed = X_WDepth.dot(rh)

        scene_points = np.hstack((left_points_transformed[:3, :],
                                  middle_points_transformed[:3, :],
                                  right_points_transformed[:3, :]))
        scene_colors = np.hstack((left_colors, middle_colors, right_colors))

        nan_indices = np.logical_not(np.isnan(scene_points))

        scene_points = scene_points[:, nan_indices[0, :]]
        scene_colors = scene_colors[:, nan_indices[0, :]]

        return scene_points.T, scene_colors.T


    def _DoCalcOutput(self, context, output):
        scene_points, scene_colors = self._ExtractPointCloud(context)

        if self.viz:
            np.save("scene_points", scene_points)
            np.save("scene_colors", scene_colors)

        output.get_mutable_value().resize(scene_points.shape[0])
        output.get_mutable_value().mutable_xyzs()[:] = scene_points.T
        output.get_mutable_value().mutable_rgbs()[:] = scene_colors.T