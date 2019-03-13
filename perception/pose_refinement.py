import argparse
import numpy as np
import os
import yaml

from PIL import Image
from perception_tools.iterative_closest_point import RunICP

import pydrake.perception as mut

from pydrake.common.eigen_geometry import Isometry3
from pydrake.examples.manipulation_station import (
    ManipulationStation, ManipulationStationHardwareInterface,
    CreateDefaultYcbObjectList)
from pydrake.math import RollPitchYaw
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import AbstractValue, DiagramBuilder, LeafSystem
from pydrake.systems.rendering import PoseBundle
from pydrake.systems.sensors import CameraInfo, PixelType


class PoseRefinement(LeafSystem):

    # TODO(kmuhlrad): update documentation
    def __init__(self, config_file, model_points_file, model_image_file,
                 object_name, segment_scene_function=None, alignment_function=None,
                 viz=False, viz_save_location=""):
        """
        A system that takes in a point cloud, an initial pose guess, and an
        object model and calculates a refined pose of the object. The user can
        optionally supply a custom segmentation function and pose alignment
        function used to determine the pose. If these functions aren't supplied,
        the default functions in this class will be used. The input point_cloud
        is assumed to be in camera frame, and will be transformed according to
        the supplied camera configuration file into world frame. The input
        X_WObject_guess is the initial guess of the pose of the object with
        respect to world frame. The points in model_points_file are expected to
        be in world frame. The output X_WObject_refined is also in world frame.

        @param config_file str. A path to a .yml configuration file for the
            camera. Note that only the "right camera" will be used.
        # TODO(kmuhlrad): use trimesh with an obj instead?
        @param model_points_file str. A path to an .npy file containing the
            object mesh.
        @param model_image_file str. A path to an image file containing the
            object texture.
        @param segment_scene_function A Python function that returns a subset of
            the scene point cloud. See self.SegmentScene for more details.
        @param alignment_function A Python function that calculates a pose from
            a segmented point cloud. See self.AlignPose for more details.
        @param viz bool. If True, save the transformed and segmented point
            clouds as serialized numpy arrays in viz_save_location.
        @param viz_save_location str. If viz is True, the directory to save
            the transformed and segmented point clouds. The default is saving
            all point clouds to the current directory.

        @system{
          @input_port{point_cloud},
          # @input_port{X_WObject_guess},
          @input_port{pose_bundle},
          @output_port{X_WObject_refined}
        }
        """
        LeafSystem.__init__(self)

        self.point_cloud_port = self._DeclareAbstractInputPort(
            "point_cloud", AbstractValue.Make(mut.PointCloud()))
        # self.init_pose_port = self._DeclareAbstractInputPort(
        #     "X_WObject_guess", AbstractValue.Make(Isometry3))


        # TODO(kmuhlrad): don't hardcode this
        self.pose_bundle_port = self._DeclareAbstractInputPort(
            "pose_bundle", AbstractValue.Make(PoseBundle(num_poses=6)))

        self._DeclareAbstractOutputPort("X_WObject_refined",
                                        lambda: AbstractValue.Make(
                                            Isometry3.Identity()),
                                        self._DoCalcOutput)

        self.segment_scene_function = segment_scene_function
        self.alignment_function = alignment_function
        self.model = np.load(model_points_file)
        self.model_image = Image.open(model_image_file)
        self.object_name = object_name

        self._LoadConfigFile(config_file)

        self.viz = viz
        self.viz_save_location = viz_save_location

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
        X_WCamera = Isometry3.Identity()
        X_WCamera.set_rotation(
            RollPitchYaw(world_transform["roll"],
                         world_transform["pitch"],
                         world_transform["yaw"]).ToRotationMatrix().matrix())
        X_WCamera.set_translation(
            [world_transform["x"], world_transform["y"], world_transform["z"]])

        # construct the transformation matrix
        internal_transform = camera_config["internal_transform"]

        X_CameraDepth = Isometry3.Identity()
        X_CameraDepth.set_rotation(
            RollPitchYaw(internal_transform["roll"],
                         internal_transform["pitch"],
                         internal_transform["yaw"]).ToRotationMatrix().matrix())
        X_CameraDepth.set_translation([internal_transform["x"],
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

    def _TransformPointCloud(self, point_cloud, colors):
        # transform the point cloud according to the config file
        pc_h = np.ones((4, point_cloud.shape[1]))
        pc_h[:3, :] = np.copy(point_cloud)

        X_WDepth = self.camera_configs["right_camera_pose_world"].multiply(
            self.camera_configs["right_camera_pose_internal"])
        point_cloud_transformed = X_WDepth.matrix().dot(pc_h)

        # Filter the final point cloud for NaNs and infs
        nan_indices = np.logical_not(np.isnan(point_cloud_transformed))

        point_cloud_transformed = point_cloud_transformed[:, nan_indices[0, :]]
        filtered_colors = colors[:, nan_indices[0, :]]

        inf_indices = np.logical_not(np.isinf(point_cloud_transformed))

        point_cloud_transformed = point_cloud_transformed[:, inf_indices[0, :]]
        filtered_colors = filtered_colors[:, inf_indices[0, :]]

        return point_cloud_transformed[:3, :].T, filtered_colors.T

    def ThresholdArray(self, arr, min_val, max_val):
        """
        Finds where the values of arr are between min_val and max_val inclusive.

        @param arr An (N, ) numpy array containing number values.
        @param min_val number. The minimum value threshold.
        @param max_val number. The maximum value threshold.

        @return An (M, ) numpy array of the integer indices in arr with values
            that are between min_val and max_val.
        """
        return np.where(
            abs(arr - (max_val + min_val) / 2.) < (max_val - min_val) / 2.)[0]

    def SegmentScene(
            self, scene_points, scene_colors, model, model_image, init_pose):
        """
        Returns a subset of the scene point cloud representing the segmentation
        of the object of interest.

        The default segmentation function has two parts:
            1. An area filter based on the object model and initial pose. Points
            are only included if they are within the size of the largest model
            dimension of either side of the initial pose. For example, if the
            largest dimension of the model was 2 and init_pose was located at
            (0, 3, 4), all points included in the segmentation mask have
            x-values between [-2, 2], y-values between [1, 5], and z-values
            between [2, 6].

            2. A color filter based on the average (r, g, b) value of the
            model texture, excluding black. Points are only included if their
            (r, g, b) value is within a threshold of the average value. For
            example, if the average (r, g, b) value was (0, 127, 255) and the
            threshold was 10, all points included in the segmentation mask have
            r-values between [0, 10], g-values between [117, 137], and b-values
            between [245, 255]. The default threshold is 51.

        The final segmentation mask is the intersection between the area filter
        and the color filter.

        If a custom scene segmentation function is supplied, it must have this
        method signature.

        @param scene_points An Nx3 numpy array representing a scene.
        @param scene_colors An Nx3 numpy array of rgb values corresponding to
            the points in scene_points.
        @param model A Px3 numpy array representing the object model.
        @param model_image A PIL.Image containing the object texture.
        @param init_pose An Isometry3 representing the initial guess of the
            pose of the object.

        @return segmented_points An Mx3 numpy array of segmented object points.
        @return segmented_colors An Mx3 numpy array of corresponding segmented
            object colors.
        """
        if self.segment_scene_function:
            return self.segment_scene_function(
                scene_points, scene_colors, model, model_image, init_pose)

        # Filter by area around initial pose guess
        max_delta_x = np.abs(np.max(model[:, 0]) - np.min(model[:, 0]))
        max_delta_y = np.abs(np.max(model[:, 1]) - np.min(model[:, 1]))
        max_delta_z = np.abs(np.max(model[:, 2]) - np.min(model[:, 2]))

        max_delta = np.max([max_delta_x, max_delta_y, max_delta_z])

        init_x = init_pose.matrix()[0, 3]
        init_y = init_pose.matrix()[1, 3]
        init_z = init_pose.matrix()[2, 3]

        x_min = init_x - max_delta
        x_max = init_x + max_delta

        y_min = init_y - max_delta
        y_max = init_y + max_delta

        z_min = init_z - max_delta
        z_max = init_z + max_delta

        x_indices = self.ThresholdArray(scene_points[:, 0], x_min, x_max)
        y_indices = self.ThresholdArray(scene_points[:, 1], y_min, y_max)
        z_indices = self.ThresholdArray(scene_points[:, 2], z_min, z_max)

        indices = reduce(np.intersect1d, (x_indices, y_indices, z_indices))

        return scene_points[indices, :], scene_colors[indices, :]

    def AlignPose(self, segmented_scene_points, segmented_scene_colors,
                model, model_image, init_pose):
        """Returns the pose of the object of interest.

        The default pose alignment function runs ICP on the segmented scene
        with a maximum of 100 iterations and stopping threshold of 1e-8.

        If a custom pose alignment function is supplied, it must have this
        method signature.

        Args:
        @param segmented_scene_points An Nx3 numpy array of the segmented object
            points.
        @param segmented_scene_colors An Nx3 numpy array of the segmented object
            colors.
        @param model A Px3 numpy array representing the object model.
        @param model_image A PIL.Image containing the object texture.
        @param init_pose An Isometry3 representing the initial guess of the
            pose of the object.

        Returns:
        @return A 4x4 numpy array representing the pose of the object.
        """

        if self.alignment_function:
            return self.alignment_function(
                segmented_scene_points, segmented_scene_colors, model,
                model_image, init_pose)

        X_MS, _, _ = RunICP(
            model, segmented_scene_points, init_guess=init_pose.matrix(),
            max_iterations=100, tolerance=1e-8)

        return X_MS

    def _DoCalcOutput(self, context, output):
        # init_pose = self.EvalAbstractInput(
        #     context, self.init_pose_port.get_index()).get_value()
        pose_bundle = self.EvalAbstractInput(
            context, self.pose_bundle_port.get_index()).get_value()
        init_pose = None
        for i in range(pose_bundle.get_num_poses()):
            if pose_bundle.get_name(i) == self.object_name:
                init_pose = pose_bundle.get_pose(i)
                break
        X_WDepth = self.camera_configs["right_camera_pose_world"].multiply(
            self.camera_configs["right_camera_pose_internal"])
        init_pose = X_WDepth.multiply(init_pose)
        point_cloud = self.EvalAbstractInput(
            context, self.point_cloud_port.get_index()).get_value()

        # scene_points, scene_colors = self._TransformPointCloud(
        #     point_cloud.xyzs(), point_cloud.rgbs())
        scene_points = np.copy(point_cloud.xyzs()).T
        scene_colors = np.copy(point_cloud.rgbs()).T

        segmented_scene_points, segmented_scene_colors = self.SegmentScene(
            scene_points, scene_colors, self.model, self.model_image, init_pose)

        if self.viz:
            np.save(os.path.join(
                        self.viz_save_location, "transformed_scene_points"),
                    scene_points)
            np.save(os.path.join(
                        self.viz_save_location, "transformed_scene_colors"),
                    scene_colors)
            np.save(os.path.join(
                        self.viz_save_location, "segmented_scene_points"),
                    segmented_scene_points)
            np.save(os.path.join(
                        self.viz_save_location, "segmented_scene_colors"),
                    segmented_scene_colors)

        # init_pose = self._TransformDopePose(self.object_name, init_pose)
        X_WObject_refined = self.AlignPose(
            segmented_scene_points, segmented_scene_colors, self.model,
            self.model_image, init_pose)

        output.get_mutable_value().set_matrix(X_WObject_refined)

# TODO(kmuhlrad): make this more formal, right now this is only the crackers
def TransformDopePose(self, object_name, pose):
    transforms = {}
    X_Cracker = Isometry3.Identity()
    X_Cracker.set_matrix(
        np.array([[0, 0, 1., 0 ],
                  [ -1., 0, 0, 0 ],
                  [ 0, -1., 0, 0 ],
                  [ -.014141999483108521, .10347499847412109, .012884999513626099, 1 ]]).T)
    transforms['cracker'] = X_Cracker

    X_Sugar = Isometry3.Identity()
    X_Sugar.set_matrix(
        np.array([
            [ -3.4877998828887939, 3.4899001121520996, 99.878196716308594, 0 ],
            [ -99.926002502441406, -1.7441999912261963, -3.4284999370574951, 0 ],
            [ 1.6224000453948975, -99.923896789550781, 3.5481998920440674, 0 ],
            [ -1.795199990272522, 8.7579002380371094, 0.38839998841285706, 100 ]]).T / 100.)
    transforms['sugar'] = X_Sugar

    X_Soup = Isometry3.Identity()
    X_Soup.set_matrix(
        np.array([
            [ 99.144500732421875, 0, -13.052599906921387, 0 ],
            [ 13.052599906921387, 0, 99.144500732421875, 0 ],
            [ 0, -100, 0, 0 ],
            [ -0.1793999969959259, 5.1006999015808105, -8.4443998336791992, 100 ]]).T / 100.)
    transforms['soup'] = X_Soup

    X_Mustard = Isometry3.Identity()
    X_Mustard.set_matrix(
        np.array([
            [ 92.050498962402344, 0, 39.073101043701172, 0 ],
            [ -39.073101043701172, 0, 92.050498962402344, 0 ],
            [ 0, -100, 0, 0 ],
            [ 0.49259999394416809, 9.2497997283935547, 2.7135999202728271, 100 ]]).T / 100.)
    transforms['mustard'] = X_Mustard

    X_Gelatin = Isometry3.Identity()
    X_Gelatin.set_matrix(
        np.array([
            [ 22.494199752807617, 97.436996459960938, 0.19629999995231628, 0 ],
            [ -97.433296203613281, 22.495100021362305, -0.85030001401901245, 0 ],
            [ -0.87269997596740723, 0, 99.996200561523438, 0 ],
            [ -0.29069998860359192, 2.3998000621795654, -1.4543999433517456, 100 ]]).T / 100.)
    transforms['gelatin'] = X_Gelatin

    X_Meat = Isometry3.Identity()
    X_Meat.set_matrix(
        np.array([
            [ 99.862998962402344, 0, -5.2336001396179199, 0 ],
            [ 5.2336001396179199, 0, 99.862998962402344, 0 ],
            [ 0, -100, 0, 0 ],
            [ 3.4065999984741211, 3.8582999706268311, 2.4767000675201416, 100 ]]).T / 100.)
    transforms['meat'] = X_Meat

    return pose.multiply(transforms[object_name])

def ReadPosesFromFile(filename):
    pose_dict = {}
    row_num = 0
    object_name = ""
    cur_matrix = np.eye(4)
    with open(filename, "r") as f:
        for line in f:
            line = line.rstrip()
            if not line.lstrip(" ").startswith("["):
                object_name = line
            else:
                row = np.matrix(line)
                cur_matrix[row_num, :] = row
                row_num += 1
                if row_num == 4:
                    pose_dict[object_name] = Isometry3(
                        TransformDopePose(object_name, cur_matrix))
                    cur_matrix = np.eye(4)
                row_num %= 4

    return pose_dict


def CustomAlignmentFunctionDummy(segmented_scene_points, segmented_scene_colors,
                                 model, model_image, init_pose):
    """Returns the identity matrix of the object of interest.

    This is an example of writing a custom pose alignment function. Although
    this particular function is not very useful, it illustrates the required
    method signature and how to use it in a PoseRefinement System.

    Args:
    @param segmented_scene_points An Nx3 numpy array of the segmented object
        points.
    @param segmented_scene_colors An Nx3 numpy array of the segmented object
        colors.
    @param model A Px3 numpy array representing the object model.
    @param model_image A PIL.Image containing the object texture.
    @param init_pose An Isometry3 representing the initial guess of the
        pose of the object.

    Returns:
    @return The 4x4 numpy identity matrix
    """

    return np.eye(4)


def Main(config_file, model_points_file, model_image_file, dope_pose_file,
         object_name, viz=True, save_path="", custom_align=False):
    """Estimates the pose of the given object in a ManipulationStation
    DopeClutterClearing setup.

    @param config_file str. A path to a .yml configuration file for the camera.
    @param model_points_file str. A path to an .npy file containing the object
        mesh.
    @param model_image_file str. A path to an image file containing the object
        texture.
    @param viz bool. If True, save the transformed and segmented point clouds as
        serialized numpy arrays in viz_save_location.
    @param save_path str. If viz is True, the directory to save the transformed
        and segmented point clouds. The default is saving all point clouds to
        the current directory.
    @param custom_align bool. If True, use the example custom pose alignment
        function.

    @return A 4x4 Numpy array representing the pose of the object.
    """

    builder = DiagramBuilder()

    if custom_align:
        pose_refinement = builder.AddSystem(PoseRefinement(
            config_file, model_points_file, model_image_file,
            alignment_function=CustomAlignmentFunctionDummy, viz=viz,
            viz_save_location=save_path))
    else:
        pose_refinement = builder.AddSystem(PoseRefinement(
            config_file, model_points_file, model_image_file, viz=viz,
            viz_save_location=save_path))

    # realsense serial numbers are >> 100
    use_hardware = \
        int(pose_refinement.camera_configs["right_camera_serial"]) > 100

    if use_hardware:
        camera_ids = [pose_refinement.camera_configs["right_camera_serial"]]
        station = builder.AddSystem(
            ManipulationStationHardwareInterface(camera_ids))
        station.Connect()
    else:
        station = builder.AddSystem(ManipulationStation())
        station.SetupClearingStation()
        ycb_objects = CreateDefaultYcbObjectList()
        for model_file, X_WObject in ycb_objects:
            station.AddManipulandFromFile(model_file, X_WObject)
        station.Finalize()

    right_camera_info = pose_refinement.camera_configs["right_camera_info"]
    right_name_prefix = \
        "camera_" + pose_refinement.camera_configs["right_camera_serial"]

    # use scale factor of 1/1000 to convert mm to m
    dut = builder.AddSystem(
        mut.DepthImageToPointCloud(right_camera_info, PixelType.kDepth16U, 1e-3))

    builder.Connect(station.GetOutputPort(right_name_prefix + "_depth_image"),
                    dut.depth_image_input_port())
    builder.Connect(station.GetOutputPort(right_name_prefix + "_rgb_image"),
                    dut.rgb_image_input_port())

    builder.Connect(dut.point_cloud_output_port(),
                    pose_refinement.GetInputPort("point_cloud"))

    diagram = builder.Build()
    simulator = Simulator(diagram)

    dope_poses = ReadPosesFromFile(dope_pose_file)
    dope_pose = dope_poses[object_name]


    context = diagram.GetMutableSubsystemContext(
        pose_refinement, simulator.get_mutable_context())

    context.FixInputPort(pose_refinement.GetInputPort(
        "X_WObject_guess").get_index(), AbstractValue.Make(dope_pose))


    station_context = diagram.GetMutableSubsystemContext(
        station, simulator.get_mutable_context())

    station_context.FixInputPort(station.GetInputPort(
        "iiwa_feedforward_torque").get_index(), np.zeros(7))

    q0 = station.GetIiwaPosition(station_context)
    station_context.FixInputPort(station.GetInputPort(
        "iiwa_position").get_index(), q0)

    station_context.FixInputPort(station.GetInputPort(
        "wsg_position").get_index(), np.array([0.1]))

    station_context.FixInputPort(station.GetInputPort(
        "wsg_force_limit").get_index(), np.array([40.0]))


    simulator.set_publish_every_time_step(False)
    simulator.set_target_realtime_rate(1.0)
    simulator.StepTo(0.1)

    return pose_refinement.GetOutputPort("X_WObject_refined").Eval(context)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config_file",
        required=True,
        help="The path to a .yml camera config file")
    parser.add_argument(
        "--model_file",
        required=True,
        help="The path to a .npy model file")
    parser.add_argument(
        "--model_image_file",
        required=True,
        help="The path to a .png model texture file")
    parser.add_argument(
        "--dope_pose_file",
        required=True,
        help="The path to a .txt file containing poses returned by DOPE")
    parser.add_argument(
        "--object_name",
        required=True,
        help="One of 'cracker', 'sugar', 'gelatin', 'meat', 'soup', 'mustard'")
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Save the aligned and segmented point clouds for visualization")
    parser.add_argument(
        "--viz_save_path",
        required=False,
        default="",
        help="A path to a directory to save point clouds")
    parser.add_argument(
        "--custom_align",
        action="store_true",
        help="A path to a directory to save point clouds")
    args = parser.parse_args()

    print Main(
        args.config_file, args.model_file, args.model_image_file,
        args.dope_pose_file, args.object_name, args.viz, args.viz_save_path,
        args.custom_align)
