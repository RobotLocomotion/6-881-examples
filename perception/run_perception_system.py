import argparse

import numpy as np

from pydrake.common import FindResourceOrThrow
from pydrake.examples.manipulation_station import (
    ManipulationStation, ManipulationStationHardwareInterface)
from pydrake.multibody.multibody_tree.parsing import AddModelFromSdfFile
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.common.eigen_geometry import Isometry3
import pydrake.perception as mut

import meshcat.transformations as tf
from perception_tools.optimization_based_point_cloud_registration import (
    AlignSceneToModel)
from perception_tools.visualization_utils import ThresholdArray
from point_cloud_to_pose_system import PointCloudToPoseSystem


def SegmentFoamBrick(scene_points, scene_colors):
    """Removes all points that aren't a part of the foam brick.

    @param scene_points An Nx3 numpy array representing a scene.
    @param scene_colors An Nx3 numpy array of rgb values corresponding to the
        points in scene_points.

    @return brick_points An Mx3 numpy array of points in the brick.
    @return brick_colors An Mx3 numpy array of the colors of the brick points.
    """

    x_min = 0.2
    x_max = 0.71

    y_min = -0.4
    y_max = 0.4

    z_min = 0
    z_max = 0.1

    x_indices = ThresholdArray(scene_points[:, 0], x_min, x_max)
    y_indices = ThresholdArray(scene_points[:, 1], y_min, y_max)
    z_indices = ThresholdArray(scene_points[:, 2], z_min, z_max)

    indices = reduce(np.intersect1d, (x_indices, y_indices, z_indices))

    table_points = scene_points[indices, :]
    table_colors = scene_colors[indices, :]

    r_min = 0
    r_max = 1

    g_min = 0
    g_max = 0.2

    b_min = 0
    b_max = 0.2

    r_indices = ThresholdArray(table_colors[:, 0], r_min, r_max)
    g_indices = ThresholdArray(table_colors[:, 1], g_min, g_max)
    b_indices = ThresholdArray(table_colors[:, 2], b_min, b_max)

    indices = reduce(np.intersect1d, (r_indices, g_indices, b_indices))

    brick_points = table_points[indices, :]
    brick_colors = table_colors[indices, :]

    return brick_points, brick_colors


def GetFoamBrickPose(brick_points, brick_colors):
    """Finds a good 4x4 pose of the brick from the segmented points.

    @param brick_points An Nx3 numpy array of brick points.
    @param brick_colors An Nx3 numpy array of corresponding brick colors.

    @return X_MS A 4x4 numpy array of the best-fit brick pose.
    """
    model_brick = np.load("models/foam_brick_model.npy")

    num_sample_points = min(brick_points.shape[0], 250)
    X_MS, error = AlignSceneToModel(
        brick_points, model_brick, num_sample_points=num_sample_points)

    # if the best fit matrix isn't exactly an Isometry3, fix it
    try:
        Isometry3(X_MS)
    except:
        # make valid Isometry3
        sin_th = X_MS[1, 0]
        cos_th = X_MS[0, 0]

        if sin_th > 0:
            theta = np.arccos(np.clip(cos_th, -1.0, 1.0))
        else:
            theta = -np.arccos(np.clip(cos_th, -1.0, 1.0))

        X_MS[0, 0] = np.cos(theta)
        X_MS[1, 1] = np.cos(theta)
        X_MS[0, 1] = -np.sin(theta)
        X_MS[1, 0] = np.sin(theta)

    return X_MS


def GetBrickPose(config_file, viz=False):
    """Estimates the pose of the foam brick in a ManipulationStation setup.

    @param config_file str. The path to a camera configuration file.
    @param viz bool. If True, save point clouds to numpy arrays.

    @return An Isometry3 representing the pose of the brick.
    """  
    builder = DiagramBuilder()

    # create the PointCloudToPoseSystem
    pc_to_pose = builder.AddSystem(PointCloudToPoseSystem(
        config_file, viz, SegmentFoamBrick, GetFoamBrickPose))

    # realsense serial numbers are >> 100
    use_hardware = int(pc_to_pose.camera_configs["left_camera_serial"]) > 100

    if use_hardware:
        camera_ids = [
            pc_to_pose.camera_configs["left_camera_serial"],
            pc_to_pose.camera_configs["middle_camera_serial"],
            pc_to_pose.camera_configs["right_camera_serial"]]
        station = builder.AddSystem(
            ManipulationStationHardwareInterface(camera_ids))
        station.Connect()
    else:
        station = builder.AddSystem(ManipulationStation())
        station.AddCupboard()
        object_file_path = \
            "drake/examples/manipulation_station/models/061_foam_brick.sdf"
        brick = AddModelFromSdfFile(
                    FindResourceOrThrow(object_file_path),
                    station.get_mutable_multibody_plant(),
                    station.get_mutable_scene_graph() )
        station.Finalize()

    # add systems to convert the depth images from ManipulationStation to 
    # PointClouds
    left_camera_info = pc_to_pose.camera_configs["left_camera_info"]
    middle_camera_info = pc_to_pose.camera_configs["middle_camera_info"]
    right_camera_info = pc_to_pose.camera_configs["right_camera_info"]

    left_dut = builder.AddSystem(
        mut.DepthImageToPointCloud(camera_info=left_camera_info))
    middle_dut = builder.AddSystem(
        mut.DepthImageToPointCloud(camera_info=middle_camera_info))
    right_dut = builder.AddSystem(
        mut.DepthImageToPointCloud(camera_info=right_camera_info))

    left_name_prefix = \
        "camera_" + pc_to_pose.camera_configs["left_camera_serial"]
    middle_name_prefix = \
        "camera_" + pc_to_pose.camera_configs["middle_camera_serial"]
    right_name_prefix = \
        "camera_" + pc_to_pose.camera_configs["right_camera_serial"]

    # connect the depth images to the DepthImageToPointCloud converters
    builder.Connect(station.GetOutputPort(left_name_prefix + "_depth_image"),
                  left_dut.depth_image_input_port())
    builder.Connect(station.GetOutputPort(middle_name_prefix + "_depth_image"),
                  middle_dut.depth_image_input_port())
    builder.Connect(station.GetOutputPort(right_name_prefix + "_depth_image"),
                  right_dut.depth_image_input_port())

    # connect the rgb images to the PointCloudToPoseSystem
    builder.Connect(station.GetOutputPort(left_name_prefix + "_rgb_image"),
                  pc_to_pose.GetInputPort("camera_left_rgb"))
    builder.Connect(station.GetOutputPort(middle_name_prefix + "_rgb_image"),
                  pc_to_pose.GetInputPort("camera_middle_rgb"))
    builder.Connect(station.GetOutputPort(right_name_prefix + "_rgb_image"),
                  pc_to_pose.GetInputPort("camera_right_rgb"))

    # connect the XYZ point clouds to PointCloudToPoseSystem
    builder.Connect(left_dut.point_cloud_output_port(),
                  pc_to_pose.GetInputPort("left_point_cloud"))
    builder.Connect(middle_dut.point_cloud_output_port(),
                  pc_to_pose.GetInputPort("middle_point_cloud"))
    builder.Connect(right_dut.point_cloud_output_port(),
                  pc_to_pose.GetInputPort("right_point_cloud"))

    diagram = builder.Build()

    simulator = Simulator(diagram)

    if not use_hardware:
        X_WObject = Isometry3.Identity()
        X_WObject.set_translation([.6, 0, 0])
        station_context = diagram.GetMutableSubsystemContext(
            station, simulator.get_mutable_context())
        station.get_mutable_multibody_plant().tree().SetFreeBodyPoseOrThrow(
            station.get_mutable_multibody_plant().GetBodyByName(
                "base_link", brick),
                X_WObject,
                station.GetMutableSubsystemContext(
                    station.get_mutable_multibody_plant(), station_context))

    context = diagram.GetMutableSubsystemContext(pc_to_pose,
                                     simulator.get_mutable_context())

    # returns the pose of the brick, of type Isometry3
    return pc_to_pose.GetOutputPort("X_WObject").Eval(context)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
      "--config_file",
      required=True,
      help="The path to a .yml camera config file")
    parser.add_argument(
      "--viz",
      action="store_true",
      help="Save the aligned and segmented point clouds for visualization")
    args = parser.parse_args()

    print GetBrickPose(args.config_file, args.viz)
