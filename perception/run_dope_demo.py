import argparse

import numpy as np

from pydrake.common.eigen_geometry import Isometry3
from pydrake.examples.manipulation_station import ManipulationStation, _xyz_rpy
from pydrake.geometry import ConnectDrakeVisualizer
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.lcm import LcmPublisherSystem
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.systems.primitives import Demultiplexer, LogOutput
from pydrake.systems.sensors import ImageToLcmImageArrayT, PixelType
import pydrake.perception as mut

from dope_system import DopeSystem
from point_cloud_synthesis import PointCloudSynthesis
from pose_refinement import PoseRefinement
from perception_tools.visualization_utils import ThresholdArray
from sklearn.neighbors import NearestNeighbors

from plan_runner.demo_plans import GeneratePickAndPlaceObjectPlans
from plan_runner.manipulation_station_plan_runner import *
from plan_runner.open_left_door import (GenerateOpenLeftDoorPlansByTrajectory,
                                        GenerateOpenLeftDoorPlansByImpedanceOrPosition,)

from robotlocomotion import image_array_t


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
    X_WSoup = _xyz_rpy([0.40, -0.07, 0.03], [-1.57, 0, 3.14])
    ycb_object_pairs.append(
        ("drake/manipulation/models/ycb/sdf/005_tomato_soup_can.sdf", X_WSoup))

    # The mustard bottle pose.
    X_WMustard = _xyz_rpy([0.44, -0.16, 0.09], [-1.57, 0, 3.3])
    ycb_object_pairs.append(
        ("drake/manipulation/models/ycb/sdf/006_mustard_bottle.sdf",
         X_WMustard))

    # The gelatin box pose.
    X_WGelatin = _xyz_rpy([0.35, -0.32, 0.1], [-1.57, 0, 2.5])
    ycb_object_pairs.append(
        ("drake/manipulation/models/ycb/sdf/009_gelatin_box.sdf", X_WGelatin))

    # The potted meat can pose.
    X_WMeat = _xyz_rpy([0.35, -0.32, 0.03], [-1.57, 0, 2.5])
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
    @param init_pose An Isometry3 representing the initial guess of the
        pose of the object.

    @return segmented_points An Mx3 numpy array of segmented object points.
    @return segmented_colors An Mx3 numpy array of corresponding segmented
        object colors.
    """
    # Filter by area around initial pose guess
    max_delta_x = np.abs(np.max(model[:, 0]) - np.min(model[:, 0]))
    max_delta_y = np.abs(np.max(model[:, 1]) - np.min(model[:, 1]))
    max_delta_z = np.abs(np.max(model[:, 2]) - np.min(model[:, 2]))

    # CHANGED FROM MAX
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
    area_points, area_colors = SegmentArea(scene_points, scene_colors, model, model_image, init_pose)

    r_min = 100
    r_max = 255

    g_min = 0
    g_max = 255

    b_min = 0
    b_max = 255

    color_thresholds = [r_min, r_max, g_min, g_max, b_min, b_max]

    segmented_points, segmented_colors = SegmentColor(color_thresholds, area_points, area_colors, model, model_image, init_pose)

    final_points, final_colors = PruneOutliers(segmented_points, segmented_colors, 0.01, 20, init_pose.matrix()[:3, 3], 0.085)

    return final_points, final_colors

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
    area_points, area_colors = SegmentArea(scene_points, scene_colors, model, model_image, init_pose)

    r_min = 20
    r_max = 100

    g_min = 40
    g_max = 255

    b_min = 10
    b_max = 255

    color_thresholds = [r_min, r_max, g_min, g_max, b_min, b_max]

    segmented_points, segmented_colors = SegmentColor(color_thresholds, area_points, area_colors, model, model_image, init_pose)

    final_points, final_colors = PruneOutliers(segmented_points, segmented_colors, 0.01, 30, init_pose.matrix()[:3, 3], 0.074)

    return final_points, final_colors

seg_functions = {
    'cracker': SegmentCrackerBox,
    'sugar': SegmentSugarBox,
    'soup': SegmentSoupCan,
    'mustard': SegmentMustardBottle,
    'gelatin': SegmentGelatinBox,
    'meat': SegmentMeatCan,
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration", type=float, default=0.1,
        help="Desired duration of the simulation in seconds.")

    MeshcatVisualizer.add_argparse_argument(parser)
    args = parser.parse_args()

    builder = DiagramBuilder()

    # Create the ManipulationStation.
    # manip_station_sim = ManipulationStationSimulator(time_step=2e-3, objects_and_poses=CreateYcbObjectClutter())
    # station = builder.AddSystem(manip_station_sim.station)
    station = builder.AddSystem(ManipulationStation())
    station.SetupDefaultStation()
    ycb_objects = CreateYcbObjectClutter()
    for model_file, X_WObject in ycb_objects:
        station.AddManipulandFromFile(model_file, X_WObject)
    station.Finalize()

    # Create the PoseRefinement systems.
    camera_config_file = '/home/amazon/6-881-examples/perception/config/sim.yml'
    pose_refinement_systems = {}
    for obj in ["cracker", "sugar", "soup", "mustard", "gelatin", "meat"]:
        pose_refinement_systems[obj] = builder.AddSystem(PoseRefinement(
            camera_config_file, model_files[obj], image_files[obj], obj, segment_scene_function=seg_functions[obj], viz=False))

    # Create the PointCloudSynthesis system.
    pc_synth = builder.AddSystem(PointCloudSynthesis(camera_config_file, False))

    # Use the right camera for DOPE.
    right_camera_info = pose_refinement_systems["cracker"].camera_configs["right_camera_info"]
    right_name_prefix = \
        "camera_" + pose_refinement_systems["cracker"].camera_configs["right_camera_serial"]

    # Create the DOPE system
    weights_path = '/home/amazon/catkin_ws/src/dope/weights'
    dope_config_file = '/home/amazon/catkin_ws/src/dope/config/config_pose.yaml'
    dope_system = builder.AddSystem(DopeSystem(weights_path, dope_config_file))

    # TODO(kmuhlrad): figure out if I need to combine point clouds
    # Create the duts.
    # use scale factor of 1/1000 to convert mm to m
    duts = {}
    duts[pose_refinement_systems["cracker"].camera_configs["right_camera_serial"]] = builder.AddSystem(mut.DepthImageToPointCloud(
        right_camera_info, PixelType.kDepth16U, 1e-3,
        fields=mut.BaseField.kXYZs | mut.BaseField.kRGBs))
    duts[pose_refinement_systems["cracker"].camera_configs["left_camera_serial"]] = builder.AddSystem(mut.DepthImageToPointCloud(
        pose_refinement_systems["cracker"].camera_configs["left_camera_info"], PixelType.kDepth16U, 1e-3,
        fields=mut.BaseField.kXYZs | mut.BaseField.kRGBs))
    duts[pose_refinement_systems["cracker"].camera_configs["middle_camera_serial"]] = builder.AddSystem(mut.DepthImageToPointCloud(
        pose_refinement_systems["cracker"].camera_configs["middle_camera_info"], PixelType.kDepth16U, 1e-3,
        fields=mut.BaseField.kXYZs | mut.BaseField.kRGBs))

    # Connect the depth and rgb images to the dut
    for name in station.get_camera_names():
        builder.Connect(
            station.GetOutputPort("camera_" + name + "_rgb_image"),
            duts[name].color_image_input_port())
        builder.Connect(
            station.GetOutputPort("camera_" + name + "_depth_image"),
            duts[name].depth_image_input_port())

    builder.Connect(duts[pose_refinement_systems["cracker"].camera_configs["left_camera_serial"]].point_cloud_output_port(),
                    pc_synth.GetInputPort("left_point_cloud"))
    builder.Connect(duts[pose_refinement_systems["cracker"].camera_configs["middle_camera_serial"]].point_cloud_output_port(),
                    pc_synth.GetInputPort("middle_point_cloud"))
    builder.Connect(duts[pose_refinement_systems["cracker"].camera_configs["right_camera_serial"]].point_cloud_output_port(),
                    pc_synth.GetInputPort("right_point_cloud"))

    # Connect the rgb images to the DopeSystem.
    builder.Connect(station.GetOutputPort(right_name_prefix + "_rgb_image"),
                    dope_system.GetInputPort("rgb_input_image"))

    # Connect the PoseRefinement systems.
    for pose_refinement in pose_refinement_systems.values():
        builder.Connect(pc_synth.GetOutputPort("combined_point_cloud"),
                        pose_refinement.GetInputPort("point_cloud"))
        builder.Connect(dope_system.GetOutputPort("pose_bundle"),
                        pose_refinement.GetInputPort("pose_bundle"))

    # Connect visualization stuff.
    if args.meshcat:
        meshcat = builder.AddSystem(MeshcatVisualizer(
            station.get_scene_graph(), zmq_url=args.meshcat,
            open_browser=args.open_browser))
        builder.Connect(station.GetOutputPort("pose_bundle"),
                        meshcat.get_input_port(0))
    else:
        ConnectDrakeVisualizer(builder, station.get_scene_graph(),
                               station.GetOutputPort("pose_bundle"))

    # Plan Runner Stuff
    # Generate plans.

    # TODO(kmuhlrad): change this between runs
    ########################################
    q0 = np.array([0, -0.2136, 0, -2.094, 0, 0.463, 0]) #[0, 0, 0, -1.75, 0, 1.0, 0]
    # plan_list, gripper_setpoint_list = GenerateOpenLeftDoorPlansByImpedanceOrPosition(
    #     open_door_method="Impedance", is_open_fully=True, q0=q0)
    plan_list, gripper_setpoint_list = GeneratePickAndPlaceObjectPlans(
        [0.40, -0.07, 0.08], [0.40, -0.07, 0.2], is_printing=True)
    ########################################

    plan_runner = ManipStationPlanRunner(
        station=station,
        kuka_plans=plan_list,
        gripper_setpoint_list=gripper_setpoint_list)
    duration_multiplier = plan_runner.kPlanDurationMultiplier

    builder.AddSystem(plan_runner)
    builder.Connect(plan_runner.GetOutputPort("gripper_setpoint"),
                    station.GetInputPort("wsg_position"))
    builder.Connect(plan_runner.GetOutputPort("force_limit"),
                    station.GetInputPort("wsg_force_limit"))


    demux = builder.AddSystem(Demultiplexer(14, 7))
    builder.Connect(
        plan_runner.GetOutputPort("iiwa_position_and_torque_command"),
        demux.get_input_port(0))
    builder.Connect(demux.get_output_port(0),
                    station.GetInputPort("iiwa_position"))
    builder.Connect(demux.get_output_port(1),
                    station.GetInputPort("iiwa_feedforward_torque"))
    builder.Connect(station.GetOutputPort("iiwa_position_measured"),
                    plan_runner.GetInputPort("iiwa_position"))
    builder.Connect(station.GetOutputPort("iiwa_velocity_estimated"),
                    plan_runner.GetInputPort("iiwa_velocity"))


    # Add loggers
    publish_period = 0.001
    iiwa_position_command_log = LogOutput(demux.get_output_port(0), builder)
    iiwa_position_command_log._DeclarePeriodicPublish(publish_period)

    wsg_position_command_log = LogOutput(
        plan_runner.GetOutputPort("gripper_setpoint"), builder)
    wsg_position_command_log._DeclarePeriodicPublish(publish_period)


    # build diagram
    diagram = builder.Build()

    # construct simulator
    simulator = Simulator(diagram)

    # context = diagram.GetMutableSubsystemContext(
    #     dope_system, simulator.get_mutable_context())

    # Check the poses.
    # X_WCamera = pose_refinement_systems['cracker'].camera_configs["right_camera_pose_world"].multiply(
    #     pose_refinement_systems['cracker'].camera_configs["right_camera_pose_internal"])
    #
    # print("DOPE POSES")
    # pose_bundle = dope_system.GetOutputPort("pose_bundle").Eval(context)
    # for i in range(pose_bundle.get_num_poses()):
    #     if pose_bundle.get_name(i):
    #         print pose_bundle.get_name(i), X_WCamera.multiply(pose_bundle.get_pose(i))

    print("\n\nICP POSES")
    colors = {
        'cracker': 0x0dff80,
        'sugar': 0xe8de0c,
        'soup': 0xff6500,
        'mustard': 0xd90ce8,
        'gelatin': 0xffffff,
        'meat': 0x0068ff
    }
    sizes = {
        'cracker': [16.4/100., 21.34/100., 7.18/100.],
        'sugar': [9.27/100., 17.63/100., 4.51/100.],
        'soup': [6.77/100., 10.18/100., 6.77/100.],
        'mustard': [9.6/100., 19.13/100., 5.82/100.],
        'gelatin': [8.92/100., 7.31/100., 3/100.],
        'meat': [10.16/100., 8.35/100., 5.76/100.]
    }
    for obj_name in seg_functions:
        import meshcat.geometry as g
        p_context = diagram.GetMutableSubsystemContext(
            pose_refinement_systems[obj_name], simulator.get_mutable_context())
        pose = pose_refinement_systems[obj_name].GetOutputPort(
            "X_WObject_refined").Eval(p_context)
        bounding_box = g.Box(sizes[obj_name])
        material = g.MeshBasicMaterial(color=colors[obj_name])
        mesh = g.Mesh(geometry=bounding_box, material=material)
        meshcat.vis[obj_name].set_object(mesh)
        meshcat.vis[obj_name].set_transform(pose.matrix())
        print obj_name, pose

    station_context = diagram.GetMutableSubsystemContext(
        station, simulator.get_mutable_context())

    # set initial state of the robot
    station.SetIiwaPosition(station_context, q0)
    station.SetIiwaVelocity(station_context, np.zeros(7))
    station.SetWsgPosition(station_context, 0.05)
    station.SetWsgVelocity(station_context, 0)

    simulator.set_publish_every_time_step(False)
    simulator.set_target_realtime_rate(0.0) # go as fast as possible

    # calculate starting time for all plans.
    t_plan = GetPlanStartingTimes(plan_list)
    extra_time = 2.0
    sim_duration = t_plan[-1]*duration_multiplier + extra_time
    print "simulation duration", sim_duration
    simulator.Initialize()
    simulator.StepTo(sim_duration)

    output_dict = {}
    output_dict["q0"] = q0
    output_dict["iiwa_position_t"] = iiwa_position_command_log.sample_times()
    output_dict["iiwa_position_data"] = iiwa_position_command_log.data()
    output_dict["wsg_position_t"] = wsg_position_command_log.sample_times()
    output_dict["wsg_position_data"] = wsg_position_command_log.data()

    import pickle
    import time
    with open("teleop_log_%d.pickle" % (time.time()*1000*1000), "wb") as f:
        pickle.dump(output_dict, f)



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
