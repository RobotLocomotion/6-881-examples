import argparse

import numpy as np

from pydrake.examples.manipulation_station import ManipulationStation, _xyz_rpy
from pydrake.geometry import ConnectDrakeVisualizer
from pydrake.perception import BaseField, DepthImageToPointCloud
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, AbstractValue
from pydrake.systems.meshcat_visualizer import (
    MeshcatVisualizer, MeshcatPointCloudVisualizer)
from pydrake.systems.perception import PointCloudConcatenation
from pydrake.systems.sensors import PixelType

from perception.dope_system import DopeSystem
from perception.pose_refinement import PoseRefinement, ObjectInfo
from perception_tools.file_utils import LoadCameraConfigFile
from perception_tools.visualization_utils import ThresholdArray
from sklearn.neighbors import NearestNeighbors


model_file_base_path = "models/"
model_files = {
    "cracker": model_file_base_path + "003_cracker_box_textured.npy",
    "sugar": model_file_base_path + "004_sugar_box_textured.npy",
    "soup": model_file_base_path + "005_tomato_soup_can_textured.npy",
    "mustard": model_file_base_path + "006_mustard_bottle_textured.npy",
    "meat": model_file_base_path + "010_potted_meat_can_textured.npy"
}


def CreateYcbObjectClutter():
    ycb_object_pairs = []

    X_WCracker = _xyz_rpy([0.35, 0.14, 0.09], [0, -1.57, 4.01])
    ycb_object_pairs.append(
        ("drake/manipulation/models/ycb/sdf/003_cracker_box.sdf", X_WCracker))

    X_WSugar = _xyz_rpy([0.28, -0.17, 0.03], [0, 1.57, 3.14])
    ycb_object_pairs.append(
        ("drake/manipulation/models/ycb/sdf/004_sugar_box.sdf", X_WSugar))

    X_WSoup = _xyz_rpy([0.40, -0.07, 0.03], [-1.57, 0, 3.14])
    ycb_object_pairs.append(
        ("drake/manipulation/models/ycb/sdf/005_tomato_soup_can.sdf", X_WSoup))

    X_WMustard = _xyz_rpy([0.44, -0.16, 0.09], [-1.57, 0, 3.3])
    ycb_object_pairs.append(
        ("drake/manipulation/models/ycb/sdf/006_mustard_bottle.sdf",
         X_WMustard))

    X_WMeat = _xyz_rpy([0.35, -0.32, 0.03], [-1.57, 0, 2.5])
    ycb_object_pairs.append(
        ("drake/manipulation/models/ycb/sdf/010_potted_meat_can.sdf", X_WMeat))

    return ycb_object_pairs


def SegmentArea(scene_points, scene_colors, model, init_pose):
    """
    Returns a subset of the scene point cloud representing the segmentation
    of the object of interest. Points are only included if they are within the
    size of the largest model dimension of either side of the initial pose. For
    example, if the largest dimension of the model was 2 and init_pose was
    located at (0, 3, 4), all points included in the segmentation mask have
    x-values between [-2, 2], y-values between [1, 5], and z-values between
    [2, 6].

    @param scene_points An Nx3 numpy array representing a scene.
    @param scene_colors An Nx3 numpy array of rgb values corresponding to
        the points in scene_points.
    @param model A Px3 numpy array representing the object model.
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


def SegmentColor(color_thresholds, area_points, area_colors, model, init_pose):
    """
    Removes all points from the given point clouds whose colors are not within
    the given thresholds.

    @param color_thresholds: A list of [r_min, r_max, g_min, g_max, b_min,
        b_max] thresholds
    @param area_points An Nx3 numpy array representing a scene.
    @param area_colors An Nx3 numpy array of rgb values corresponding to
        the points in scene_points.
    @param model A Px3 numpy array representing the object model.
    @param init_pose An RigidTransform representing the initial guess of the
        pose of the object.

    @return final_points An Mx3 numpy array of segmented object points.
    @return final_colors An Mx3 numpy array of corresponding segmented
        object colors.
    """
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


def PruneOutliers(points, colors, min_distance, num_neighbors,
                  init_center=np.zeros(3), max_center_dist=3.0):
    """
    Removes points that are farther than min_distance away on average from
    their num_neighbors nearest neighbors, and points that are farther than
    max_center_dist away from the init_center.

    @param points An Nx3 numpy array representing a scene.
    @param colors An Nx3 numpy array of rgb values corresponding to
        the points in scene_points.
    @param min_distance float. The minimum average distance in meters that a
        point can be away from its num_neighbors nearest neighbors.
    @param num_neighbors int. The number of nearest neighbors to check.
    @param init_center An 3x1 numpy array of the (x, y, z) center of an area of
        interest.
    :param max_center_dist float. The minimum distance in meters a point can be
        from init_center.

    @return segmented_points An Mx3 numpy array of segmented object points.
    @return segmented_colors An Mx3 numpy array of corresponding segmented
        object colors.
    """

    if not points.size or not colors.size:
        return points, colors
    nbrs = NearestNeighbors(n_neighbors=num_neighbors).fit(np.copy(points))
    distances, indices = nbrs.kneighbors(np.copy(points))

    outliers = []
    avg_dist = np.sum(distances, axis=1) / float(num_neighbors - 1)

    for i in range(points.shape[0]):
        center_dist = np.sqrt(np.sum((points[i] - init_center)**2))
        if avg_dist[i] < min_distance and center_dist < max_center_dist:
            outliers.append(i)

    return np.copy(points)[outliers, :], np.copy(colors)[outliers, :]


def SegmentCrackerBox(scene_points, scene_colors, model, init_pose):
    area_points, area_colors = SegmentArea(
        scene_points, scene_colors, model, init_pose)

    r_min = 100
    r_max = 255

    g_min = 0
    g_max = 255

    b_min = 0
    b_max = 255

    color_thresholds = [r_min, r_max, g_min, g_max, b_min, b_max]

    segmented_points, segmented_colors = SegmentColor(
        color_thresholds, area_points, area_colors, model, init_pose)

    final_points, final_colors = PruneOutliers(
        segmented_points, segmented_colors, 0.01, 100,
        init_pose.matrix()[:3, 3], 0.15)

    return final_points, final_colors


def SegmentSugarBox(scene_points, scene_colors, model, init_pose):
    area_points, area_colors = SegmentArea(
        scene_points, scene_colors, model, init_pose)

    r_min = 0
    r_max = 255

    g_min = 100
    g_max = 255

    b_min = 0
    b_max = 100

    color_thresholds = [r_min, r_max, g_min, g_max, b_min, b_max]

    segmented_points, segmented_colors = SegmentColor(
        color_thresholds, area_points, area_colors, model, init_pose)

    final_points, final_colors = PruneOutliers(
        segmented_points, segmented_colors, 0.01, 100,
        init_pose.matrix()[:3, 3], 0.11)

    return final_points, final_colors


def SegmentFarSoupCan(scene_points, scene_colors, model, init_pose):
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

    return area_points, area_colors


def SegmentMustardBottle(scene_points, scene_colors, model, init_pose):
    area_points, area_colors = SegmentArea(scene_points, scene_colors, model, init_pose)

    r_min = 100
    r_max = 255

    g_min = 100
    g_max = 255

    b_min = 0
    b_max = 100

    color_thresholds = [r_min, r_max, g_min, g_max, b_min, b_max]

    segmented_points, segmented_colors = SegmentColor(
        color_thresholds, area_points, area_colors, model, init_pose)

    final_points, final_colors = PruneOutliers(
        segmented_points, segmented_colors, 0.01, 40)

    return final_points, final_colors


def SegmentFarMeatCan(scene_points, scene_colors, model, init_pose):
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

    area_points, area_colors = (
        scene_points[indices, :], scene_colors[indices, :])

    return area_points, area_colors


seg_functions = {
    'cracker': None, #SegmentCrackerBox,
    'sugar': None, #SegmentSugarBox,
    'soup': None, #SegmentFarSoupCan,
    'mustard': None, #SegmentMustardBottle,
    'meat': None, #SegmentFarMeatCan,
}


def ConstructObjectInfoDict():
    object_info_dict = {}

    for object_name in model_files:
        info = ObjectInfo(object_name,
                          model_files[object_name],
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

    # Create the PointCloudConcatenation system.
    id_list = station.get_camera_names()
    left_serial = "1"
    middle_serial = "2"
    right_serial = "0"

    camera_configs = LoadCameraConfigFile(
        "/home/amazon/6-881-examples/perception/config/sim.yml")
    pc_concat = builder.AddSystem(PointCloudConcatenation(id_list))

    X_WCamera = camera_configs[right_serial]["camera_pose_world"].multiply(
        camera_configs[right_serial]["camera_pose_internal"])

    # Use the right camera for DOPE.
    right_camera_info = camera_configs[right_serial]["camera_info"]
    right_name_prefix = "camera_" + str(right_serial)

    # Create the DOPE system
    weights_path = 'weights'
    dope_config_file = 'config/dope_config.yml'
    dope_system = builder.AddSystem(
        DopeSystem(weights_path, dope_config_file, X_WCamera))

    # Create the DepthImageToPointClouds.
    # use scale factor of 1/1000 to convert mm to m
    di2pcs = {}
    di2pcs[right_serial] = builder.AddSystem(DepthImageToPointCloud(
        right_camera_info, PixelType.kDepth16U, 1e-3,
        fields=BaseField.kXYZs | BaseField.kRGBs))
    di2pcs[left_serial] = builder.AddSystem(DepthImageToPointCloud(
        camera_configs[left_serial]["camera_info"], PixelType.kDepth16U, 1e-3,
        fields=BaseField.kXYZs | BaseField.kRGBs))
    di2pcs[middle_serial] = builder.AddSystem(DepthImageToPointCloud(
        camera_configs[middle_serial]["camera_info"],
        PixelType.kDepth16U, 1e-3,
        fields=BaseField.kXYZs | BaseField.kRGBs))

    # Connect the depth and rgb images to the dut
    for name in station.get_camera_names():
        builder.Connect(
            station.GetOutputPort("camera_" + name + "_rgb_image"),
            di2pcs[name].color_image_input_port())
        builder.Connect(
            station.GetOutputPort("camera_" + name + "_depth_image"),
            di2pcs[name].depth_image_input_port())

    for id in id_list:
        builder.Connect(di2pcs[id].point_cloud_output_port(),
                        pc_concat.GetInputPort(
                            "point_cloud_CiSi_{}".format(id)))

    # Connect the rgb images to the DopeSystem.
    builder.Connect(station.GetOutputPort(right_name_prefix + "_rgb_image"),
                    dope_system.GetInputPort("rgb_input_image"))

    # Connect the PoseRefinement system.
    builder.Connect(pc_concat.GetOutputPort("point_cloud_FS"),
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

        scene_pc_vis = builder.AddSystem(MeshcatPointCloudVisualizer(
            meshcat, name="scene_point_cloud"))
        builder.Connect(pc_concat.GetOutputPort("point_cloud_FS"),
                        scene_pc_vis.GetInputPort("point_cloud_P"))

        for obj_name in model_files:
            pc_vis = builder.AddSystem(MeshcatPointCloudVisualizer(
                meshcat, name="{}_point_cloud".format(obj_name)))
            builder.Connect(pose_refinement_system.GetOutputPort(
                "segmented_point_cloud_W_{}".format(obj_name)),
                pc_vis.GetInputPort("point_cloud_P"))
    else:
        ConnectDrakeVisualizer(builder, station.get_scene_graph(),
                               station.GetOutputPort("pose_bundle"))

    q0 = np.array([0, 0, 0, -1.75, 0, 1.0, 0])

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
        station.GetInputPort("iiwa_feedforward_torque").get_index(),
        np.zeros(7))
    station_context.FixInputPort(
        station.GetInputPort("wsg_position").get_index(), [0.05])
    station_context.FixInputPort(
        station.GetInputPort("wsg_force_limit").get_index(), [50])

    pc_concat_context = diagram.GetMutableSubsystemContext(
        pc_concat, simulator.get_mutable_context())
    for id in id_list:
        X_WP = camera_configs[id]["camera_pose_world"].multiply(
            camera_configs[id]["camera_pose_internal"])
        pc_concat_context.FixInputPort(
            pc_concat.GetInputPort("X_FCi_{}".format(id)).get_index(),
            AbstractValue.Make(X_WP))

    # Door now starts open
    left_hinge_joint = station.get_multibody_plant().GetJointByName(
        "left_door_hinge")
    left_hinge_joint.set_angle(station_context, angle=-np.pi/2)
    right_hinge_joint = station.get_multibody_plant().GetJointByName(
        "right_door_hinge")
    right_hinge_joint.set_angle(station_context, angle=np.pi/2)

    context = diagram.GetMutableSubsystemContext(
        dope_system, simulator.get_mutable_context())

    simulator.set_publish_every_time_step(False)
    simulator.set_target_realtime_rate(0.0) # go as fast as possible

    simulator.Initialize()
    simulator.StepTo(2.0)

    # Check the poses.
    colors = {
        'cracker': 0x0dff80,
        'sugar': 0xe8de0c,
        'soup': 0xff6500,
        'mustard': 0xd90ce8,
        'meat': 0x0068ff
    }
    sizes = {
        'cracker': [16.4/100., 21.34/100., 7.18/100.],
        'sugar': [9.27/100., 17.63/100., 4.51/100.],
        'soup': [6.77/100., 10.18/100., 6.77/100.],
        'mustard': [9.6/100., 19.13/100., 5.82/100.],
        'meat': [10.16/100., 8.35/100., 5.76/100.]
    }

    import meshcat.geometry as g

    print("DOPE POSES")
    pose_bundle = dope_system.GetOutputPort("pose_bundle_W").Eval(context)
    for i in range(pose_bundle.get_num_poses()):
        if pose_bundle.get_name(i) in ["mustard", "soup", "meat", "cracker", "sugar"]:
            print pose_bundle.get_name(i), pose_bundle.get_pose(i).matrix()
            bounding_box = g.Box(sizes[pose_bundle.get_name(i)])
            material = g.MeshBasicMaterial(colors[pose_bundle.get_name(i)])
            mesh = g.Mesh(geometry=bounding_box, material=material)
            meshcat.vis["{}_dope".format(pose_bundle.get_name(i))].set_object(mesh)
            meshcat.vis["{}_dope".format(pose_bundle.get_name(i))].set_transform(pose_bundle.get_pose(i).matrix())

    print("\n\nICP POSES")
    p_context = diagram.GetMutableSubsystemContext(
        pose_refinement_system, simulator.get_mutable_context())
    pose_bundle = pose_refinement_system.GetOutputPort(
        "refined_pose_bundle_W").Eval(p_context)
    for obj_name in ["soup", "mustard", "meat"]:
        for i in range(pose_bundle.get_num_poses()):
            if pose_bundle.get_name(i) == obj_name:
                pose = pose_bundle.get_pose(i)
                break
        bounding_box = g.Box(sizes[obj_name])
        material = g.MeshBasicMaterial(color=colors[obj_name])
        mesh = g.Mesh(geometry=bounding_box, material=material)
        meshcat.vis["{}_icp".format(obj_name)].set_object(mesh)
        meshcat.vis["{}_icp".format(obj_name)].set_transform(pose.matrix())
        print obj_name, pose.matrix().tolist()

    scene_pc = pose_refinement_system.GetInputPort("point_cloud_W").Eval(p_context)
    np.save("scene_points", scene_pc.xyzs())
    np.save("scene_points", scene_pc.rgbs())

    import cv2
    annotated_image = dope_system.GetOutputPort(
        "annotated_rgb_image").Eval(context).data
    cv2.imshow("dope image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
