import argparse

import numpy as np

from pydrake.examples.manipulation_station import ManipulationStation, _xyz_rpy
from pydrake.geometry import ConnectDrakeVisualizer
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.lcm import LcmPublisherSystem
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer, MeshcatPointCloudVisualizer, MeshcatContactVisualizer
from pydrake.systems.primitives import Demultiplexer, LogOutput
from pydrake.systems.sensors import ImageToLcmImageArrayT, PixelType
import pydrake.perception as mut

from dope_system import DopeSystem
from perception_tools.point_cloud_synthesis import PointCloudSynthesis
from pose_refinement import PoseRefinement, ObjectInfo
from perception_tools.visualization_utils import ThresholdArray
from perception_tools.file_utils import LoadCameraConfigFile
from sklearn.neighbors import NearestNeighbors

from plan_runner.demo_plans import GeneratePickAndPlaceObjectPlans, GeneratePickAndPlaceObjectTaskPlans
from plan_runner.manipulation_station_plan_runner import *
from plan_runner.open_left_door import (GenerateOpenLeftDoorPlansByTrajectory,
                                        GenerateOpenLeftDoorPlansByImpedanceOrPosition,)

from robotlocomotion import image_array_t

from pydrake.math import RigidTransform

model_file_base_path = "models/"
model_files = {
    "cracker": model_file_base_path + "003_cracker_box_textured.npy",
    "sugar": model_file_base_path + "004_sugar_box_textured.npy",
    "soup": model_file_base_path + "005_tomato_soup_can_textured.npy",
    "mustard": model_file_base_path + "006_mustard_bottle_textured.npy",
    # "gelatin": model_file_base_path + "009_gelatin_box_textured.npy",
    "meat": model_file_base_path + "010_potted_meat_can_textured.npy"
}

image_file_base_path = "/home/amazon/drake-build/install/share/drake/" \
                       "manipulation/models/ycb/meshes/"
image_files = {
    "cracker": image_file_base_path + "003_cracker_box_textured.png",
    "sugar": image_file_base_path + "004_sugar_box_textured.png",
    "soup": image_file_base_path + "005_tomato_soup_can_textured.png",
    "mustard": image_file_base_path + "006_mustard_bottle_textured.png",
    # "gelatin": image_file_base_path + "009_gelatin_box_textured.png",
    "meat": image_file_base_path + "010_potted_meat_can_textured.png"
}


def CreateYcbObjectClutter():
    ycb_object_pairs = []

    X_WCracker = _xyz_rpy([0.35, 0.14, 0.09], [0, -1.57, 4])
    ycb_object_pairs.append(
        ("drake/manipulation/models/ycb/sdf/003_cracker_box.sdf", X_WCracker))

    # The sugar box pose.
    X_WSugar = _xyz_rpy([0.28, -0.17, 0.03], [0, 1.57, 3.14])
    ycb_object_pairs.append(
        ("drake/manipulation/models/ycb/sdf/004_sugar_box.sdf", X_WSugar))

    # The tomato soup can pose.
    # After moving:
    # X_WSoup = RigidTransform(np.array([[-9.99982403e-01,  1.01915004e-16,  5.93238370e-03,  8.55377622e-01],
    #  [-5.93238370e-03, -4.44089210e-16, -9.99982403e-01, -2.38764523e-03],
    #  [-9.93129190e-17, -1.00000000e+00,  4.44089210e-16,  3.42077514e-01],
    # [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]))

    X_WSoup = _xyz_rpy([0.40, -0.07, 0.03], [-1.57, 0, 3.14])
    ycb_object_pairs.append(
        ("drake/manipulation/models/ycb/sdf/005_tomato_soup_can.sdf", X_WSoup))

    # The mustard bottle pose.
    X_WMustard = _xyz_rpy([0.44, -0.16, 0.09], [-1.57, 0, 3.3])
    '''
    [[-9.69829800e-01  1.44285253e-04 -2.43782975e-01  4.45281370e-01]
 [ 2.43782981e-01  2.15142685e-05 -9.69829809e-01 -1.61993966e-01]
 [-1.34687327e-04 -9.99999989e-01 -5.60394677e-05  8.22916025e-02]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]

    '''
    # X_WMustard = RigidTransform(np.array([[-9.69829800e-01,  1.44285253e-04, -2.43782975e-01,  4.45281370e-01],
    #                                       [ 2.43782981e-01,  2.15142685e-05, -9.69829809e-01, -1.61993966e-01],
    #                                      [-1.34687327e-04, -9.99999989e-01, -5.60394677e-05,  8.22916025e-02],
    #                                         [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]))
    ycb_object_pairs.append(
        ("drake/manipulation/models/ycb/sdf/006_mustard_bottle.sdf",
         X_WMustard))

    # The gelatin box pose.
    # X_WGelatin = _xyz_rpy([0.35, -0.32, 0.1], [-1.57, 0, 2.5])
    # ycb_object_pairs.append(
    #     ("drake/manipulation/models/ycb/sdf/009_gelatin_box.sdf", X_WGelatin))

    # The potted meat can pose.
    X_WMeat = _xyz_rpy([0.35, -0.32, 0.03], [-1.57, 0, 2.5])
    # X_WMeat = RigidTransform(np.array([[-9.93764499e-01, -7.63278329e-17, -1.11499422e-01,  8.67166894e-01],
    #  [ 1.11499422e-01, -2.22044605e-16, -9.93764499e-01, -1.33921891e-02],
    #  [ 6.24500451e-17, -1.00000000e+00,  1.11022302e-16,  6.25261549e-02],
    # [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]))

    ycb_object_pairs.append(
        ("drake/manipulation/models/ycb/sdf/010_potted_meat_can.sdf", X_WMeat))

    return ycb_object_pairs


def SegmentArea(scene_points, scene_colors, model, model_image, init_pose):
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
    @param init_pose An RigidTransform representing the initial guess of the
        pose of the object.

    @return segmented_points An Mx3 numpy array of segmented object points.
    @return segmented_colors An Mx3 numpy array of corresponding segmented
        object colors.
    """
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

    z_min = max(init_z - max_delta, 0)
    z_max = init_z + max_delta

    x_indices = ThresholdArray(scene_points[:, 0], x_min, x_max)
    y_indices = ThresholdArray(scene_points[:, 1], y_min, y_max)
    z_indices = ThresholdArray(scene_points[:, 2], z_min, z_max)

    indices = reduce(np.intersect1d, (x_indices, y_indices, z_indices))

    return scene_points[indices, :], scene_colors[indices, :]

def SegmentColor(color_thresholds, area_points, area_colors, model, model_image, init_pose):
    r_min = color_thresholds[0]
    r_max = color_thresholds[1]

    g_min = color_thresholds[2]
    g_max = color_thresholds[3]

    b_min = color_thresholds[4]
    b_max = color_thresholds[5]

    r_indices = ThresholdArray(area_colors[:, 0], r_min, r_max)
    g_indices = ThresholdArray(area_colors[:, 1], g_min, g_max)
    b_indices = ThresholdArray(area_colors[:, 2], b_min, b_max)

    indices = reduce(np.intersect1d, (r_indices, g_indices, b_indices))

    final_points = area_points[indices, :]
    final_colors = area_colors[indices, :]

    return final_points, final_colors

def PruneOutliers(points, colors, min_distance, num_neighbors, init_center=np.zeros(3), max_center_dist=3.0):
    nbrs = NearestNeighbors(n_neighbors=num_neighbors).fit(np.copy(points))
    distances, indices = nbrs.kneighbors(np.copy(points))

    outliers = []
    avg_dist = np.sum(distances, axis=1) / float(num_neighbors - 1)

    for i in range(points.shape[0]):
        center_dist = np.sqrt(np.sum((points[i] - init_center)**2))
        if avg_dist[i] < min_distance and center_dist < max_center_dist:
            outliers.append(i)

    return np.copy(points)[outliers, :], np.copy(colors)[outliers, :]

def SegmentCrackerBox(scene_points, scene_colors, model, model_image, init_pose):
    return scene_points, scene_colors
    # area_points, area_colors = SegmentArea(scene_points, scene_colors, model, model_image, init_pose)
    #
    # r_min = 100
    # r_max = 255
    #
    # g_min = 0
    # g_max = 255
    #
    # b_min = 0
    # b_max = 255
    #
    # color_thresholds = [r_min, r_max, g_min, g_max, b_min, b_max]
    #
    # segmented_points, segmented_colors = SegmentColor(color_thresholds, area_points, area_colors, model, model_image, init_pose)
    #
    # final_points, final_colors = PruneOutliers(segmented_points, segmented_colors, 0.01, 100, init_pose.matrix()[:3, 3], 0.15)
    #
    # return final_points, final_colors

def SegmentSugarBox(scene_points, scene_colors, model, model_image, init_pose):
    return scene_points, scene_colors
    # area_points, area_colors = SegmentArea(scene_points, scene_colors, model, model_image, init_pose)
    #
    # r_min = 0
    # r_max = 255
    #
    # g_min = 100
    # g_max = 255
    #
    # b_min = 0
    # b_max = 100
    #
    # color_thresholds = [r_min, r_max, g_min, g_max, b_min, b_max]
    #
    # segmented_points, segmented_colors = SegmentColor(color_thresholds, area_points, area_colors, model, model_image, init_pose)
    #
    # final_points, final_colors = PruneOutliers(segmented_points, segmented_colors, 0.01, 100, init_pose.matrix()[:3, 3], 0.11)
    #
    # return final_points, final_colors

def SegmentSoupCan(scene_points, scene_colors, model, model_image, init_pose):
    max_delta_x = np.abs(np.max(model[:, 0]) - np.min(model[:, 0]))
    max_delta_y = np.abs(np.max(model[:, 1]) - np.min(model[:, 1]))
    max_delta_z = np.abs(np.max(model[:, 2]) - np.min(model[:, 2]))

    # farther away from camera means more noise
    max_delta =  2 * np.max([max_delta_x, max_delta_y, max_delta_z])

    init_x = init_pose.matrix()[0, 3]
    init_y = init_pose.matrix()[1, 3]
    init_z = init_pose.matrix()[2, 3]

    x_min = init_x - max_delta
    x_max = init_x + max_delta

    y_min = init_y - max_delta
    y_max = init_y + max_delta

    # second shelf of the cabinet
    z_min = max(init_z - max_delta, 0)
    z_max = init_z + max_delta

    x_indices = ThresholdArray(scene_points[:, 0], x_min, x_max)
    y_indices = ThresholdArray(scene_points[:, 1], y_min, y_max)
    z_indices = ThresholdArray(scene_points[:, 2], z_min, z_max)

    indices = reduce(np.intersect1d, (x_indices, y_indices, z_indices))

    area_points, area_colors = scene_points[indices, :], scene_colors[indices, :]

    z_min = 0.292
    z_max = 0.6

    z_indices = ThresholdArray(area_points[:, 2], z_min, z_max)

    return area_points[z_indices, :], area_colors[z_indices, :]

    # r_min = 100
    # r_max = 255
    #
    # g_min = 0
    # g_max = 255
    #
    # b_min = 0
    # b_max = 255
    #
    # color_thresholds = [r_min, r_max, g_min, g_max, b_min, b_max]
    #
    # segmented_points, segmented_colors = SegmentColor(color_thresholds, area_points, area_colors, model, model_image, init_pose)
    #
    # final_points, final_colors = PruneOutliers(segmented_points, segmented_colors, 0.01, 20, init_pose.matrix()[:3, 3], 0.085)
    #
    # return final_points, final_colors

def SegmentMustardBottle(scene_points, scene_colors, model, model_image, init_pose):
    area_points, area_colors = SegmentArea(scene_points, scene_colors, model, model_image, init_pose)

    r_min = 100
    r_max = 255

    g_min = 100
    g_max = 255

    b_min = 0
    b_max = 100

    color_thresholds = [r_min, r_max, g_min, g_max, b_min, b_max]

    segmented_points, segmented_colors = SegmentColor(color_thresholds, area_points, area_colors, model, model_image, init_pose)

    final_points, final_colors = PruneOutliers(segmented_points, segmented_colors, 0.01, 40)

    return final_points, final_colors

def SegmentGelatinBox(scene_points, scene_colors, model, model_image, init_pose):
    area_points, area_colors = SegmentArea(scene_points, scene_colors, model, model_image, init_pose)

    r_min = 0
    r_max = 255

    g_min = 0
    g_max = 255

    b_min = 0
    b_max = 255

    color_thresholds = [r_min, r_max, g_min, g_max, b_min, b_max]

    segmented_points, segmented_colors = SegmentColor(color_thresholds, area_points, area_colors, model, model_image, init_pose)

    final_points, final_colors = PruneOutliers(segmented_points, segmented_colors, 0.01, 20)

    return final_points, final_colors

def SegmentMeatCan(scene_points, scene_colors, model, model_image, init_pose):
    max_delta_x = np.abs(np.max(model[:, 0]) - np.min(model[:, 0]))
    max_delta_y = np.abs(np.max(model[:, 1]) - np.min(model[:, 1]))
    max_delta_z = np.abs(np.max(model[:, 2]) - np.min(model[:, 2]))

    # farther away from camera means more noise
    max_delta =  2.5 * np.max([max_delta_x, max_delta_y, max_delta_z])

    init_x = init_pose.matrix()[0, 3]
    init_y = init_pose.matrix()[1, 3]
    init_z = init_pose.matrix()[2, 3]

    # bottom shelf of cabinet
    x_min = max(init_x - max_delta, 0.65)
    x_max = init_x + max_delta

    y_min = max(init_y - max_delta, -0.2)
    y_max = init_y + max_delta

    z_min = max(init_z - max_delta, 0.03)
    z_max = min(init_z + max_delta, 0.25)

    x_indices = ThresholdArray(scene_points[:, 0], x_min, x_max)
    y_indices = ThresholdArray(scene_points[:, 1], y_min, y_max)
    z_indices = ThresholdArray(scene_points[:, 2], z_min, z_max)

    indices = reduce(np.intersect1d, (x_indices, y_indices, z_indices))

    area_points, area_colors = scene_points[indices, :], scene_colors[indices, :]

    return area_points, area_colors


    # r_min = 20
    # r_max = 100
    #
    # g_min = 40
    # g_max = 255
    #
    # b_min = 10
    # b_max = 255
    #
    # color_thresholds = [r_min, r_max, g_min, g_max, b_min, b_max]
    #
    # segmented_points, segmented_colors = SegmentColor(color_thresholds, area_points, area_colors, model, model_image, init_pose)
    #
    # final_points, final_colors = PruneOutliers(segmented_points, segmented_colors, 0.01, 30, init_pose.matrix()[:3, 3], 0.074)
    #
    # return final_points, final_colors

seg_functions = {
    'cracker': SegmentCrackerBox,
    'sugar': SegmentSugarBox,
    'soup': SegmentSoupCan,
    'mustard': SegmentMustardBottle,
    # 'gelatin': SegmentGelatinBox,
    'meat': SegmentMeatCan,
}

def ConstructObjectInfoDict():
    object_info_dict = {}

    for object_name in model_files:
        info = ObjectInfo(object_name,
                          model_files[object_name],
                          image_files[object_name],
                          segment_scene_function=seg_functions[object_name])
        object_info_dict[object_name] = info

    return object_info_dict

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration", type=float, default=0.1,
        help="Desired duration of the simulation in seconds.")

    MeshcatVisualizer.add_argparse_argument(parser)
    args = parser.parse_args()

    builder = DiagramBuilder()

    station = builder.AddSystem(ManipulationStation())
    station.SetupDefaultStation()
    ycb_objects = CreateYcbObjectClutter()
    for model_file, X_WObject in ycb_objects:
        station.AddManipulandFromFile(model_file, X_WObject)
    station.Finalize()

    # Create the PoseRefinement system.
    object_info_dict = ConstructObjectInfoDict()
    pose_refinement_system = builder.AddSystem(
        PoseRefinement(object_info_dict))

    # Create the PointCloudSynthesis system.
    id_list = station.get_camera_names()
    left_serial = "1"
    middle_serial = "2"
    right_serial = "0"

    camera_configs = LoadCameraConfigFile(
        "/home/amazon/6-881-examples/perception/config/sim.yml")
    transform_dict = {}
    for id in id_list:
        transform_dict[id] = camera_configs[id]["camera_pose_world"].multiply(
            camera_configs[id]["camera_pose_internal"])
    pc_synth = builder.AddSystem(PointCloudSynthesis(transform_dict))

    X_WCamera = camera_configs[right_serial]["camera_pose_world"].multiply(
        camera_configs[right_serial]["camera_pose_internal"])

    # Use the right camera for DOPE.
    right_camera_info = camera_configs[right_serial]["camera_info"]
    right_name_prefix = "camera_" + str(right_serial)

    # Create the DOPE system
    weights_path = '/home/amazon/catkin_ws/src/dope/weights'
    dope_config_file = '/home/amazon/catkin_ws/src/dope/config/config_pose.yaml'
    dope_system = builder.AddSystem(DopeSystem(weights_path, dope_config_file, X_WCamera))

    # TODO(kmuhlrad): clean this up
    # Create the duts.
    # use scale factor of 1/1000 to convert mm to m
    duts = {}
    duts[right_serial] = builder.AddSystem(mut.DepthImageToPointCloud(
         right_camera_info, PixelType.kDepth16U, 1e-3,
         fields=mut.BaseField.kXYZs | mut.BaseField.kRGBs))
    duts[left_serial] = builder.AddSystem(mut.DepthImageToPointCloud(
         camera_configs[left_serial]["camera_info"], PixelType.kDepth16U, 1e-3,
         fields=mut.BaseField.kXYZs | mut.BaseField.kRGBs))
    duts[middle_serial] = builder.AddSystem(mut.DepthImageToPointCloud(
         camera_configs[middle_serial]["camera_info"], PixelType.kDepth16U, 1e-3,
         fields=mut.BaseField.kXYZs | mut.BaseField.kRGBs))

    # Connect the depth and rgb images to the dut
    for name in station.get_camera_names():
        builder.Connect(
            station.GetOutputPort("camera_" + name + "_rgb_image"),
            duts[name].color_image_input_port())
        builder.Connect(
            station.GetOutputPort("camera_" + name + "_depth_image"),
            duts[name].depth_image_input_port())

    builder.Connect(duts[left_serial].point_cloud_output_port(),
                    pc_synth.GetInputPort("point_cloud_P_" + left_serial))
    builder.Connect(duts[middle_serial].point_cloud_output_port(),
                    pc_synth.GetInputPort("point_cloud_P_" + middle_serial))
    builder.Connect(duts[right_serial].point_cloud_output_port(),
                    pc_synth.GetInputPort("point_cloud_P_" + right_serial))

    # Connect the rgb images to the DopeSystem.
    builder.Connect(station.GetOutputPort(right_name_prefix + "_rgb_image"),
                    dope_system.GetInputPort("rgb_input_image"))

    # Connect the PoseRefinement system.
    builder.Connect(pc_synth.GetOutputPort("combined_point_cloud_W"),
                    pose_refinement_system.GetInputPort("point_cloud_W"))
    builder.Connect(dope_system.GetOutputPort("pose_bundle_W"),
                    pose_refinement_system.GetInputPort("pose_bundle_W"))

    # Connect visualization stuff.
    if args.meshcat:
        meshcat = builder.AddSystem(MeshcatVisualizer(
            station.get_scene_graph(), zmq_url=args.meshcat,
            open_browser=args.open_browser))
        builder.Connect(station.GetOutputPort("pose_bundle"),
                        meshcat.get_input_port(0))

        contact_vis = builder.AddSystem(MeshcatContactVisualizer(
            meshcat, contact_force_scale=-7.5, plant=station.get_multibody_plant()))
        builder.Connect(station.GetOutputPort("pose_bundle"),
                        contact_vis.GetInputPort("pose_bundle"))
        builder.Connect(station.GetOutputPort("contact_results"),
                        contact_vis.GetInputPort("contact_results"))

        scene_pc_vis = builder.AddSystem(MeshcatPointCloudVisualizer(
            meshcat, name="scene_point_cloud"))
        builder.Connect(pc_synth.GetOutputPort("combined_point_cloud_W"),
                        scene_pc_vis.GetInputPort("point_cloud_P"))

        mustard_pc_vis = builder.AddSystem(MeshcatPointCloudVisualizer(
            meshcat, name="mustard_bottle_point_cloud"))
        mustard_pc_vis.set_name("other_system")
        builder.Connect(pose_refinement_system.GetOutputPort(
            "segmented_point_cloud_W_mustard"),
                        mustard_pc_vis.GetInputPort("point_cloud_P"))
    else:
        ConnectDrakeVisualizer(builder, station.get_scene_graph(),
                               station.GetOutputPort("pose_bundle"))

    q0 = np.array([0, -0.2136, 0, -2.094, 0, 0.463, 0]) #[0, 0, 0, -1.75, 0, 1.0, 0]

    # build diagram
    diagram = builder.Build()

    # construct simulator
    simulator = Simulator(diagram)

    station_context = diagram.GetMutableSubsystemContext(
        station, simulator.get_mutable_context())

    # set initial state of the robot
    station_context.FixInputPort(
        station.GetInputPort("iiwa_position").get_index(), q0)
    station_context.FixInputPort(
        station.GetInputPort("iiwa_feedforward_torque").get_index(), np.zeros(7))
    station_context.FixInputPort(
        station.GetInputPort("wsg_position").get_index(), [0.05])
    station_context.FixInputPort(
        station.GetInputPort("wsg_force_limit").get_index(), [50])

    # Door now starts open
    # left_hinge_joint = station.get_multibody_plant().GetJointByName("left_door_hinge")
    # left_hinge_joint.set_angle(station_context, angle=-np.pi/2)
    # right_hinge_joint = station.get_multibody_plant().GetJointByName("right_door_hinge")
    # right_hinge_joint.set_angle(station_context, angle=np.pi/2)

    # import cv2
    # context = diagram.GetMutableSubsystemContext(
    #     dope_system, simulator.get_mutable_context())
    # annotated_image = dope_system.GetOutputPort(
    #     "annotated_rgb_image").Eval(context).data
    # cv2.imshow("dope image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)

    simulator.set_publish_every_time_step(False)
    simulator.set_target_realtime_rate(0.0) # go as fast as possible

    simulator.Initialize()
    simulator.StepTo(2.0)



    # Check the poses.
    print("DOPE POSES")
    pose_bundle = dope_system.GetOutputPort("pose_bundle_W").Eval(context)
    for i in range(pose_bundle.get_num_poses()):
        if pose_bundle.get_name(i) == "mustard":
            import meshcat.geometry as g
            print pose_bundle.get_name(i), pose_bundle.get_pose(i).matrix()
            bounding_box = g.Box([9.6/100., 19.13/100., 5.82/100.])
            material = g.MeshBasicMaterial(color=0xffffff)
            mesh = g.Mesh(geometry=bounding_box, material=material)
            meshcat.vis["dope"].set_object(mesh)
            meshcat.vis["dope"].set_transform(pose_bundle.get_pose(i).matrix())

    print("\n\nICP POSES")
    colors = {
        'cracker': 0x0dff80,
        'sugar': 0xe8de0c,
        'soup': 0xff6500,
        'mustard': 0xd90ce8,
        # 'gelatin': 0xffffff,
        'meat': 0x0068ff
    }
    sizes = {
        'cracker': [16.4/100., 21.34/100., 7.18/100.],
        'sugar': [9.27/100., 17.63/100., 4.51/100.],
        'soup': [6.77/100., 10.18/100., 6.77/100.],
        'mustard': [9.6/100., 19.13/100., 5.82/100.],
        # 'gelatin': [8.92/100., 7.31/100., 3/100.],
        'meat': [10.16/100., 8.35/100., 5.76/100.]
    }
    p_context = diagram.GetMutableSubsystemContext(
        pose_refinement_system, simulator.get_mutable_context())
    pose_bundle = pose_refinement_system.GetOutputPort(
        "refined_pose_bundle_W").Eval(p_context)
    for obj_name in ["soup", "mustard", "meat"]:
        import meshcat.geometry as g
        for i in range(pose_bundle.get_num_poses()):
            if pose_bundle.get_name(i) == obj_name:
                pose = pose_bundle.get_pose(i)
                break
        bounding_box = g.Box(sizes[obj_name])
        material = g.MeshBasicMaterial(color=colors[obj_name])
        mesh = g.Mesh(geometry=bounding_box, material=material)
        meshcat.vis[obj_name].set_object(mesh)
        meshcat.vis[obj_name].set_transform(pose.matrix())
        print obj_name, pose.matrix().tolist()


    import cv2
    # context = diagram.GetMutableSubsystemContext(
    #     dope_system, simulator.get_mutable_context())
    annotated_image = dope_system.GetOutputPort(
        "annotated_rgb_image").Eval(context).data
    cv2.imshow("dope image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

    # calculate starting time for all plans.
    # t_plan = GetPlanStartingTimes(plan_list)
    # extra_time = 2.0
    # sim_duration = t_plan[-1]*duration_multiplier + extra_time
    # print "simulation duration", sim_duration
    # simulator.Initialize()
    # simulator.StepTo(sim_duration)
    #
    # output_dict = {}
    # output_dict["q0"] = q0
    # output_dict["iiwa_position_t"] = iiwa_position_command_log.sample_times()
    # output_dict["iiwa_position_data"] = iiwa_position_command_log.data()
    # output_dict["wsg_position_t"] = wsg_position_command_log.sample_times()
    # output_dict["wsg_position_data"] = wsg_position_command_log.data()
    #
    # import pickle
    # import time
    # with open("teleop_log_%d.pickle" % (time.time()*1000*1000), "wb") as f:
    #     pickle.dump(output_dict, f)



if __name__ == "__main__":
    main()  # This is what you would have, but the following is useful:

    # # These are temporary, for debugging, so meh for programming style.
    # import sys, trace
    #
    # # If there are segfaults, it's a good idea to always use stderr as it
    # # always prints to the screen, so you should get as much output as
    # # possible.
    # sys.stdout = sys.stderr
    #
    # # Now trace execution:
    # tracer = trace.Trace(trace=1, count=0, ignoredirs=["/usr", sys.prefix])
    # tracer.run('main()')
