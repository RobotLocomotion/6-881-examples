import argparse
import numpy as np

from PIL import Image
from perception_tools.file_utils import ReadPosesFromFile, LoadCameraConfigFile
from perception_tools.iterative_closest_point import RunICP
from perception_tools.visualization_utils import ThresholdArray

import pydrake.perception as mut

from pydrake.examples.manipulation_station import (
    ManipulationStation, ManipulationStationHardwareInterface,
    CreateDefaultYcbObjectList)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import AbstractValue, DiagramBuilder, LeafSystem
from pydrake.systems.rendering import PoseBundle
from pydrake.systems.sensors import PixelType

class ObjectInfo(object):

    def __init__(self, object_name, model_points_file, model_image_file,
                 segment_scene_function=None, alignment_function=None):
        """
        ObjectInfo is a data structure containing the relevant information
        about objects whose pose will be refined. More information about its
        use can be found with the PoseRefinement class documentation.

        @param object_name str. The name of the object.
        @param model_points_file str. The path to a .npy file containing a
            point cloud of the model object in world frame.
        @param model_image_file str. The path to an image file containing the
            colored object texture.
        @param segment_scene_function function. A function that segments the
            object out of the entire scene point cloud. For more details about
            the method signature and expected behavior, see
            PoseRefinement.DefaultSegmentSceneFunction.
        @param alignment_function function. A function that takes the
            segmented point cloud and aligns it with the model point cloud. For
            more details about the method signature and expected behavior, see
            PoseRefinement.DefaultAlignPoseFunction.
        """
        self.object_name = object_name
        self.model_points_file = model_points_file
        self.model_image_file = model_image_file
        self.segment_scene_function = segment_scene_function
        self.alignment_function = alignment_function


class PoseRefinement(LeafSystem):

    def __init__(self, object_info_dict):
        """
        A system that takes in a point cloud, initial pose guesses, and some
        ObjectInfo objects to calculate refined poses for each of the
        objects. Through the ObjectInfo dictionary, the user can optionally
        provide custom segmentation and pose alignment function used to
        determine the final pose. If these functions aren't supplied,
        the default functions in this class will be used. All point clouds and
        poses are assumed to be in world frame. The output is a pose_bundle
        containing the refined poses of every object. If the segmented
        point cloud for a given object is empty, the returned pose will be
        the initial guess. There are also output ports of each of the
        segmented point clouds for visualization purposes.

        @param object_info_dict dict. A dictionary of ObjectInfo objects keyed
            by the ObjectInfo.object_name.

        @system{
          @input_port{point_cloud_W},
          @input_port{pose_bundle_W},
          @output_port{refined_pose_bundle_W}
          @output_port{segmented_point_cloud_W_obj_name_1}
          .
          .
          .
          @output_port{segmented_point_cloud_W_obj_name_n}
        }
        """
        LeafSystem.__init__(self)

        self.object_info_dict = object_info_dict

        self.point_cloud_port = self.DeclareAbstractInputPort(
            "point_cloud_W", AbstractValue.Make(mut.PointCloud()))


        self.pose_bundle_port = self.DeclareAbstractInputPort(
            "pose_bundle_W", AbstractValue.Make(PoseBundle(
                num_poses=len(self.object_info_dict))))

        self.DeclareAbstractOutputPort("refined_pose_bundle_W",
                                        lambda: AbstractValue.Make(
                                            PoseBundle(
                                                num_poses=len(
                                                    self.object_info_dict))),
                                        self.DoCalcOutput)

        output_fields = mut.Fields(mut.BaseField.kXYZs | mut.BaseField.kRGBs)
        for object_name in self.object_info_dict:
            self.DeclareAbstractOutputPort(
                "segmented_point_cloud_W_{}".format(object_name),
                lambda: AbstractValue.Make(
                    mut.PointCloud(
                    fields=output_fields)),
                lambda context, output: self.DoCalcSegmentedPointCloud(
                    context, output, object_name))

    def DefaultSegmentSceneFunction(
            self, scene_points, scene_colors, model, model_image, init_pose):
        """
        Returns a subset of the scene point cloud representing the segmentation
        of the object of interest.

        The default segmentation function is an area filter based on the object
        model and initial pose. Points are only included if they are within the
        size of the largest model dimension of either side of the initial pose.
        For example, if the largest dimension of the model was 2 and init_pose
        was located at (0, 3, 4), all points included in the segmentation mask
        have x-values between [-2, 2], y-values between [1, 5], and z-values
        between [2, 6].

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

        x_indices = ThresholdArray(scene_points[:, 0], x_min, x_max)
        y_indices = ThresholdArray(scene_points[:, 1], y_min, y_max)
        z_indices = ThresholdArray(scene_points[:, 2], z_min, z_max)

        indices = reduce(np.intersect1d, (x_indices, y_indices, z_indices))

        return scene_points[indices, :], scene_colors[indices, :]

    def DefaultAlignPoseFunction(self, segmented_object_points,
                                 segmented_object_colors, model,
                                 model_image, init_pose):
        """Returns the pose of the object of interest.

        The default pose alignment function runs ICP on the segmented scene
        with a maximum of 100 iterations and stopping threshold of 1e-8. If
        the segmented scene is empty, it will return the initial pose.

        If a custom pose alignment function is supplied, it must have this
        method signature.

        Args:
        @param segmented_object_points An Nx3 numpy array of the segmented
            object points.
        @param segmented_object_colors An Nx3 numpy array of the segmented
            object colors.
        @param model A Px3 numpy array representing the object model.
        @param model_image A PIL.Image containing the object texture.
        @param init_pose An Isometry3 representing the initial guess of the
            pose of the object.

        Returns:
        @return A 4x4 numpy array representing the pose of the object.
        """
        if len(segmented_object_points):
            X_MS, _, _ = RunICP(
                model, segmented_object_points, init_guess=init_pose.matrix(),
                max_iterations=100, tolerance=1e-8)
        else:
            X_MS = init_pose.matrix()

        return X_MS

    def _SegmentObject(self, point_cloud, object_info, init_pose):
        model = np.load(object_info.model_points_file)
        model_image = Image.open(object_info.model_image_file)

        scene_points = np.copy(point_cloud.xyzs()).T
        scene_colors = np.copy(point_cloud.rgbs()).T

        if object_info.segment_scene_function:
            return object_info.segment_scene_function(
                    scene_points, scene_colors, model, model_image, init_pose)
        else:
            return self.DefaultSegmentSceneFunction(
                    scene_points, scene_colors, model, model_image, init_pose)

    def _RefineSinglePose(self, point_cloud, object_info, init_pose):
        model = np.load(object_info.model_points_file)
        model_image = Image.open(object_info.model_image_file)

        object_points, object_colors = self._SegmentObject(
            point_cloud, object_info, init_pose)

        if object_info.alignment_function:
            return object_info.alignment_function(
                object_points, object_colors, model, model_image, init_pose)
        else:
            return self.DefaultAlignPoseFunction(
                object_points, object_colors, model, model_image, init_pose)

    def DoCalcOutput(self, context, output):
        pose_bundle = self.EvalAbstractInput(
            context, self.pose_bundle_port.get_index()).get_value()
        point_cloud = self.EvalAbstractInput(
            context, self.point_cloud_port.get_index()).get_value()

        for i in range(pose_bundle.get_num_poses()):
            if pose_bundle.get_name(i):
                object_name = pose_bundle.get_name(i)
                init_pose = pose_bundle.get_pose(i)
                X_WObject_refined = self._RefineSinglePose(
                    point_cloud, self.object_info_dict[object_name], init_pose)
                output.get_mutable_value().set_name(i, object_name)
                output.get_mutable_value().set_pose(i, X_WObject_refined)

    def DoCalcSegmentedPointCloud(self, context, output, object_name):
        pose_bundle = self.EvalAbstractInput(
                context, self.pose_bundle_port.get_index()).get_value()
        point_cloud = self.EvalAbstractInput(
            context, self.point_cloud_port.get_index()).get_value()

        pose_bundle_index = 0
        for i in range(pose_bundle.get_num_poses()):
            if pose_bundle.get_name(i) == object_name:
                pose_bundle_index = i
                break

        init_pose = pose_bundle.get_pose(pose_bundle_index)
        object_points, object_colors = self._SegmentObject(
            point_cloud, self.object_info_dict[object_name], init_pose)

        output.get_mutable_value().resize(object_points.shape[0])
        output.get_mutable_value().mutable_xyzs()[:] = object_points.T
        output.get_mutable_value().mutable_rgbs()[:] = object_colors.T


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


def ConstructDefaultObjectInfoDict(custom_align):
    object_info_dict = {}

    model_file_base_path = "models/"
    model_files = {
        "cracker": model_file_base_path + "003_cracker_box_textured.npy",
        "sugar": model_file_base_path + "004_sugar_box_textured.npy",
        "soup": model_file_base_path + "005_tomato_soup_can_textured.npy",
        "mustard": model_file_base_path + "006_mustard_bottle_textured.npy",
        "gelatin": model_file_base_path + "009_gelatin_box_textured.npy",
        "meat": model_file_base_path + "010_potted_meat_can_textured.npy"
    }

    image_file_base_path = "/home/amazon/drake-build/install/share/drake/" \
                           "manipulation/models/ycb/meshes/"
    image_files = {
        "cracker": image_file_base_path + "003_cracker_box_textured.png",
        "sugar": image_file_base_path + "004_sugar_box_textured.png",
        "soup": image_file_base_path + "005_tomato_soup_can_textured.png",
        "mustard": image_file_base_path + "006_mustard_bottle_textured.png",
        "gelatin": image_file_base_path + "009_gelatin_box_textured.png",
        "meat": image_file_base_path + "010_potted_meat_can_textured.png"
    }

    for object_name in model_files:
        info = ObjectInfo(object_name,
                          model_files[object_name],
                          image_files[object_name],
                          alignment_function=(
                              CustomAlignmentFunctionDummy if custom_align else None))
        object_info_dict[object_name] = info

    return object_info_dict


def Main(camera_config_file, camera_serial, init_pose_file, object_name,
         custom_align=False):
    """Estimates the pose of the given object in a ManipulationStation setup.

    @param camera_config_file str. A path to a file containing the camera
        configuration file.
    @param camera_serial str. The serial number of the camera to use the
        point cloud from.
    @param init_pose_file str. A path to a file containing initial guesses of
        object poses in world frame.
    @param object_name str. The short name of the object to get the pose of.
    @param custom_align bool. If True, use the example custom pose alignment
        function.

    @return A 4x4 Numpy array representing the pose of the object.
    """

    object_info_dict = ConstructDefaultObjectInfoDict(custom_align)

    builder = DiagramBuilder()

    pose_refinement = builder.AddSystem(PoseRefinement(object_info_dict))

    camera_configs = LoadCameraConfigFile(camera_config_file)

    # realsense serial numbers are >> 100
    use_hardware = int(camera_serial) > 100

    if use_hardware:
        camera_ids = [camera_serial]
        station = builder.AddSystem(
            ManipulationStationHardwareInterface(camera_ids))
        station.Connect()
    else:
        station = builder.AddSystem(ManipulationStation())
        station.SetupClutterClearingStation()
        ycb_objects = CreateDefaultYcbObjectList()
        for model_file, X_WObject in ycb_objects:
            station.AddManipulandFromFile(model_file, X_WObject)
        station.Finalize()

    # use scale factor of 1/1000 to convert mm to m
    dut = builder.AddSystem(mut.DepthImageToPointCloud(
        camera_configs[camera_serial]["camera_info"],
        PixelType.kDepth16U,
        1e-3,
        fields=mut.BaseField.kXYZs | mut.BaseField.kRGBs))

    builder.Connect(
        station.GetOutputPort("camera_{}_depth_image".format(camera_serial)),
        dut.depth_image_input_port())
    builder.Connect(
        station.GetOutputPort("camera_{}_rgb_image".format(camera_serial)),
        dut.color_image_input_port())

    builder.Connect(dut.point_cloud_output_port(),
                    pose_refinement.GetInputPort("point_cloud_W"))

    diagram = builder.Build()
    simulator = Simulator(diagram)

    # This is a substitute for running an object pose estimator to test this
    # system. An example of a valid pose file can be seen in the package
    # README.md.
    init_poses = ReadPosesFromFile(init_pose_file)

    context = diagram.GetMutableSubsystemContext(
        pose_refinement, simulator.get_mutable_context())

    context.FixInputPort(pose_refinement.GetInputPort(
        "pose_bundle_W").get_index(), AbstractValue.Make(init_poses))

    station_context = diagram.GetMutableSubsystemContext(
        station, simulator.get_mutable_context())

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

    refined_poses = pose_refinement.GetOutputPort(
        "refined_pose_bundle_W").Eval(context)

    for i in range(refined_poses.get_num_poses()):
        if refined_poses.get_name(i) == object_name:
            return refined_poses.get_pose(i)

    return "couldn't find object"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera_config_file",
        required=True,
        help="The path to a camera configuration .yml file")
    parser.add_argument(
        "--camera_serial_number",
        required=True,
        help="The serial number of the camera to look at")
    parser.add_argument(
        "--init_pose_file",
        required=True,
        help="The path to a .txt file containing initial guesses of poses")
    parser.add_argument(
        "--object_name",
        required=True,
        help="One of 'cracker', 'sugar', 'gelatin', 'meat', 'soup', 'mustard'")
    parser.add_argument(
        "--custom_align",
        action="store_true",
        help="A path to a directory to save point clouds")
    args = parser.parse_args()

    print Main(args.camera_config_file, args.camera_serial_number,
               args.init_pose_file, args.object_name, args.custom_align)
