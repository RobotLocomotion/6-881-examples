import numpy as np
import yaml

from pydrake.systems.framework import AbstractValue, LeafSystem
from pydrake.common.eigen_geometry import Isometry3, AngleAxis
from pydrake.systems.sensors import CameraInfo, ImageRgba8U
import pydrake.perception as mut

import meshcat.transformations as tf

class PointCloudToPoseSystem(LeafSystem):

    def __init__(self, config_file, viz=False, segment_scene_function=None,
                 get_pose_function=None):
        """
        A system that takes in 3 Drake PointClouds and ImageRgba8U from
        RGBDCameras and determines the pose of an object in them. The user must
        supply a segmentation function and pose alignment to determine the pose.
        If these functions aren't supplied, the returned pose will always be the
        4x4 identity matrix.

        @param config_file str. A path to a .yml configuration file for the
            cameras.
        @param viz bool. If True, save the aligned and segmented point clouds
            as serialized numpy arrays.
        @param segment_scene_function A Python function that returns a subset of
            the scene point cloud. See self.SegmentScene for more details.
        @param get_pose_function A Python function that calculates a pose from a
            segmented point cloud. See self.GetPose for more details.

        @system{
          @input_port{camera_left_rgb},
          @input_port{camera_middle_rgb},
          @input_port{camera_right_rgb},
          @input_port{left_point_cloud},
          @input_port{middle_point_cloud},
          @input_port{right_point_cloud},
          @output_port{X_WObject}
        }
        """
        LeafSystem.__init__(self)

        # TODO(kmuhlrad): Remove once Drake PointCloud object supports RGB
        # fields.
        self.left_rgb = self._DeclareAbstractInputPort(
            "camera_left_rgb", AbstractValue.Make(ImageRgba8U(848, 480)))
        self.middle_rgb = self._DeclareAbstractInputPort(
            "camera_middle_rgb", AbstractValue.Make(ImageRgba8U(848, 480)))
        self.right_rgb = self._DeclareAbstractInputPort(
            "camera_right_rgb", AbstractValue.Make(ImageRgba8U(848, 480)))

        self.left_depth = self._DeclareAbstractInputPort(
            "left_point_cloud", AbstractValue.Make(mut.PointCloud()))
        self.middle_depth = self._DeclareAbstractInputPort(
            "middle_point_cloud", AbstractValue.Make(mut.PointCloud()))
        self.right_depth = self._DeclareAbstractInputPort(
            "right_point_cloud", AbstractValue.Make(mut.PointCloud()))

        self._DeclareAbstractOutputPort("X_WObject",
                                        lambda: AbstractValue.Make(
                                            Isometry3.Identity()),
                                        self._DoCalcOutput)

        self.segment_scene_function = segment_scene_function
        self.get_pose_function = get_pose_function

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
        left_rgb_image = self.EvalAbstractInput(
            context, self.left_rgb.get_index()).get_value()
        middle_rgb_image = self.EvalAbstractInput(
            context, self.middle_rgb.get_index()).get_value()
        right_rgb_image = self.EvalAbstractInput(
            context, self.right_rgb.get_index()).get_value()

        if (not left_rgb_image.height() or 
            not middle_rgb_image.height() or 
            not right_rgb_image.height()):
            
            print "not enough data points"
            return None
        else:
            left_point_cloud = self.EvalAbstractInput(
                context, self.left_depth.get_index()).get_value()
            middle_point_cloud = self.EvalAbstractInput(
                context, self.middle_depth.get_index()).get_value()
            right_point_cloud = self.EvalAbstractInput(
                context, self.right_depth.get_index()).get_value()

            left_points = np.array(left_point_cloud.xyzs())
            left_colors = self._ConstructPointCloudColors(left_rgb_image)

            middle_points = np.array(middle_point_cloud.xyzs())
            middle_colors = self._ConstructPointCloudColors(middle_rgb_image)

            right_points = np.array(right_point_cloud.xyzs())
            right_colors = self._ConstructPointCloudColors(right_rgb_image)

            if self.viz:
                np.save("saved_point_clouds/left_points", left_points.T)
                np.save("saved_point_clouds/left_colors", left_colors.T)
                np.save("saved_point_clouds/middle_points", middle_points.T)
                np.save("saved_point_clouds/middle_colors", middle_colors.T)
                np.save("saved_point_clouds/right_points", right_points.T)
                np.save("saved_point_clouds/right_colors", right_colors.T)

            return self._AlignPointClouds(left_points,
                                          left_colors,
                                          middle_points,
                                          middle_colors,
                                          right_points,
                                          right_colors)

    def _ConstructPointCloudColors(self, rgb_image):
        colors = np.zeros((3, rgb_image.height() * rgb_image.width()))
        cnt = 0
        for i in range(rgb_image.height()):
            for j in range(rgb_image.width()):
                colors[:3, cnt] = rgb_image.at(j, i)[:3]
                cnt += 1
        return colors / 255.

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

    def SegmentScene(self, scene_points, scene_colors):
        """
        Returns a subset of the scene point cloud representing the segmentation
        of the object of interest.

        @param scene_points An Nx3 numpy array representing a scene.
        @param scene_colors An Nx3 numpy array of rgb values corresponding to
            the points in scene_points.

        @return segmented_points An Mx3 numpy array of segmented object points.
        @return segmented_colors An Mx3 numpy array of corresponding segmented
            object colors.
        """
        if self.segment_scene_function:
            return self.segment_scene_function(scene_points, scene_colors)
        return scene_points, scene_colors

    def GetPose(self, segmented_scene_points, segmented_scene_colors):
        """Returns the pose of the object of interest.

        Args:
        @param segmented_scene_points An Nx3 numpy array of the segmented object
            points.
        @param segmented_scene_colors An Nx3 numpy array of the segmented object
            colors.

        Returns:
        @return A 4x4 numpy array representing the pose of the object. The
            default is the identity matrix if a get_pose_function is not
            supplied.
        """
        if self.get_pose_function:
            return self.get_pose_function(
                segmented_scene_points, segmented_scene_colors)
        return np.eye(4)

    def _DoCalcOutput(self, context, output):
        scene_points, scene_colors = self._ExtractPointCloud(context)
        segmented_scene_points, segmented_scene_colors = \
            self.SegmentScene(scene_points, scene_colors)

        if self.viz:
            np.save("saved_point_clouds/aligned_scene_points", scene_points)
            np.save("saved_point_clouds/aligned_scene_colors", scene_colors)
            np.save("saved_point_clouds/segmented_scene_points",
                    segmented_scene_points)
            np.save("saved_point_clouds/segmented_scene_colors",
                    segmented_scene_colors)

        X_WObject = self.GetPose(segmented_scene_points, segmented_scene_colors)

        output.get_mutable_value().set_matrix(X_WObject)