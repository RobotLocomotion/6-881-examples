import argparse

import numpy as np

from pydrake.common.eigen_geometry import Isometry3
from pydrake.examples.manipulation_station import ManipulationStation, _xyz_rpy
from pydrake.geometry import ConnectDrakeVisualizer
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.lcm import LcmPublisherSystem
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.systems.sensors import ImageToLcmImageArrayT, PixelType
import pydrake.perception as mut

from dope_system import DopeSystem
from pose_refinement import PoseRefinement
from perception_tools.visualization_utils import ThresholdArray

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

    X_WCracker = _xyz_rpy([0.35, 0.15, 0.09], [0, -1.57, 4])
    ycb_object_pairs.append(
        ("drake/manipulation/models/ycb/sdf/003_cracker_box.sdf", X_WCracker))

    # The sugar box pose.
    X_WSugar = _xyz_rpy([0.25, -0.2, 0.03], [0, 1.57, 3.14])
    ycb_object_pairs.append(
        ("drake/manipulation/models/ycb/sdf/004_sugar_box.sdf", X_WSugar))

    # The tomato soup can pose.
    X_WSoup = _xyz_rpy([0.40, -0.07, 0.03], [-1.57, 0, 3.14])
    ycb_object_pairs.append(
        ("drake/manipulation/models/ycb/sdf/005_tomato_soup_can.sdf", X_WSoup))

    # The mustard bottle pose.
    X_WMustard = _xyz_rpy([0.45, -0.16, 0.07], [-1.57, 0, 3.3])
    ycb_object_pairs.append(
        ("drake/manipulation/models/ycb/sdf/006_mustard_bottle.sdf",
         X_WMustard))

    # The gelatin box pose.
    X_WGelatin = _xyz_rpy([0.35, -0.32, 0.1], [-1.57, 0, 3.7])
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

def SegmentCrackerBox(scene_points, scene_colors, model, model_image, init_pose):
    area_points, area_colors = SegmentArea(scene_points, scene_colors, model, model_image, init_pose)

    r_min = 0
    r_max = 255

    g_min = 0
    g_max = 255

    b_min = 0
    b_max = 255

    r_indices = ThresholdArray(area_colors[:, 0], r_min, r_max)
    g_indices = ThresholdArray(area_colors[:, 1], g_min, g_max)
    b_indices = ThresholdArray(area_colors[:, 2], b_min, b_max)

    indices = reduce(np.intersect1d, (r_indices, g_indices, b_indices))

    final_points = area_points[indices, :]
    final_colors = area_colors[indices, :]

    return final_points, final_colors

def SegmentSugarBox(scene_points, scene_colors, model, model_image, init_pose):
    area_points, area_colors = SegmentArea(scene_points, scene_colors, model, model_image, init_pose)

    r_min = 0
    r_max = 255

    g_min = 0
    g_max = 255

    b_min = 0
    b_max = 255

    r_indices = ThresholdArray(area_colors[:, 0], r_min, r_max)
    g_indices = ThresholdArray(area_colors[:, 1], g_min, g_max)
    b_indices = ThresholdArray(area_colors[:, 2], b_min, b_max)

    indices = reduce(np.intersect1d, (r_indices, g_indices, b_indices))

    final_points = area_points[indices, :]
    final_colors = area_colors[indices, :]

    return final_points, final_colors

def SegmentSoupCan(scene_points, scene_colors, model, model_image, init_pose):
    area_points, area_colors = SegmentArea(scene_points, scene_colors, model, model_image, init_pose)

    r_min = 0
    r_max = 255

    g_min = 0
    g_max = 255

    b_min = 0
    b_max = 255

    r_indices = ThresholdArray(area_colors[:, 0], r_min, r_max)
    g_indices = ThresholdArray(area_colors[:, 1], g_min, g_max)
    b_indices = ThresholdArray(area_colors[:, 2], b_min, b_max)

    indices = reduce(np.intersect1d, (r_indices, g_indices, b_indices))

    final_points = area_points[indices, :]
    final_colors = area_colors[indices, :]

    return final_points, final_colors

def SegmentMustardBottle(scene_points, scene_colors, model, model_image, init_pose):
    area_points, area_colors = SegmentArea(scene_points, scene_colors, model, model_image, init_pose)

    r_min = 0
    r_max = 255

    g_min = 0
    g_max = 255

    b_min = 0
    b_max = 255

    r_indices = ThresholdArray(area_colors[:, 0], r_min, r_max)
    g_indices = ThresholdArray(area_colors[:, 1], g_min, g_max)
    b_indices = ThresholdArray(area_colors[:, 2], b_min, b_max)

    indices = reduce(np.intersect1d, (r_indices, g_indices, b_indices))

    final_points = area_points[indices, :]
    final_colors = area_colors[indices, :]

    return final_points, final_colors

def SegmentGelatinBox(scene_points, scene_colors, model, model_image, init_pose):
    area_points, area_colors = SegmentArea(scene_points, scene_colors, model, model_image, init_pose)

    r_min = 0
    r_max = 255

    g_min = 0
    g_max = 255

    b_min = 0
    b_max = 255

    r_indices = ThresholdArray(area_colors[:, 0], r_min, r_max)
    g_indices = ThresholdArray(area_colors[:, 1], g_min, g_max)
    b_indices = ThresholdArray(area_colors[:, 2], b_min, b_max)

    indices = reduce(np.intersect1d, (r_indices, g_indices, b_indices))

    final_points = area_points[indices, :]
    final_colors = area_colors[indices, :]

    return final_points, final_colors

def SegmentMeatCan(scene_points, scene_colors, model, model_image, init_pose):
    area_points, area_colors = SegmentArea(scene_points, scene_colors, model, model_image, init_pose)

    r_min = 0
    r_max = 255

    g_min = 0
    g_max = 255

    b_min = 0
    b_max = 255

    r_indices = ThresholdArray(area_colors[:, 0], r_min, r_max)
    g_indices = ThresholdArray(area_colors[:, 1], g_min, g_max)
    b_indices = ThresholdArray(area_colors[:, 2], b_min, b_max)

    indices = reduce(np.intersect1d, (r_indices, g_indices, b_indices))

    final_points = area_points[indices, :]
    final_colors = area_colors[indices, :]

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
            camera_config_file, model_files[obj], image_files[obj], obj, segment_scene_function=seg_functions[obj]))

    # Use the right camera for DOPE.
    right_camera_info = pose_refinement_systems["cracker"].camera_configs["right_camera_info"]
    right_name_prefix = \
        "camera_" + pose_refinement_systems["cracker"].camera_configs["right_camera_serial"]

    # Create the DOPE system
    weights_path = '/home/amazon/catkin_ws/src/dope/weights'
    dope_config_file = '/home/amazon/catkin_ws/src/dope/config/config_pose.yaml'
    dope_system = builder.AddSystem(DopeSystem(weights_path, dope_config_file))

    # TODO(kmuhlrad): figure out if I need to combine point clouds
    # Create the dut.
    # use scale factor of 1/1000 to convert mm to m
    dut = builder.AddSystem(mut.DepthImageToPointCloud(
        right_camera_info, PixelType.kDepth16U, 1e-3,
        fields=mut.BaseField.kXYZs | mut.BaseField.kRGBs))

    # Connect the depth and rgb images to the dut
    builder.Connect(station.GetOutputPort(right_name_prefix + "_depth_image"),
                    dut.depth_image_input_port())
    builder.Connect(station.GetOutputPort(right_name_prefix + "_rgb_image"),
                    dut.color_image_input_port())

    # Connect the rgb images to the DopeSystem.
    builder.Connect(station.GetOutputPort(right_name_prefix + "_rgb_image"),
                    dope_system.GetInputPort("rgb_input_image"))

    # Connect the PoseRefinement systems.
    for pose_refinement in pose_refinement_systems.values():
        builder.Connect(dut.point_cloud_output_port(),
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

        # image_to_lcm_image_array = builder.AddSystem(ImageToLcmImageArrayT())
        # image_to_lcm_image_array.set_name("converter")
        # cam_port = (
        #     image_to_lcm_image_array.DeclareImageInputPort[PixelType.kRgba8U](
        #         "dope"))
        # # builder.Connect(dope_system.GetOutputPort("annotated_rgb_image"),
        # #                 cam_port)
        # builder.Connect(station.GetOutputPort("camera_0_rgb_image"),
        #                 cam_port)
        #
        # image_array_lcm_publisher = builder.AddSystem(
        #     LcmPublisherSystem.Make(
        #         channel="DRAKE_RGBD_CAMERA_IMAGES",
        #         lcm_type=image_array_t,
        #         lcm=None,
        #         publish_period=0.1,
        #         use_cpp_serializer=True))
        # image_array_lcm_publisher.set_name("rgbd_publisher")
        # builder.Connect(image_to_lcm_image_array.image_array_t_msg_output_port(),
        #                 image_array_lcm_publisher.get_input_port(0))

    diagram = builder.Build()
    simulator = Simulator(diagram)

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

    context = diagram.GetMutableSubsystemContext(
        dope_system, simulator.get_mutable_context())

    # Check the poses.

    X_WCamera = pose_refinement_systems['cracker'].camera_configs["right_camera_pose_world"].multiply(
        pose_refinement_systems['cracker'].camera_configs["right_camera_pose_internal"])

    print("DOPE POSES")
    pose_bundle = dope_system.GetOutputPort("pose_bundle").Eval(context)
    for i in range(pose_bundle.get_num_poses()):
        if pose_bundle.get_name(i):
            print pose_bundle.get_name(i), X_WCamera.multiply(pose_bundle.get_pose(i))

    print("\n\nICP POSES")
    #for obj_name in pose_refinement_systems:
    for obj_name in ["mustard"]:
        p_context = diagram.GetMutableSubsystemContext(
            pose_refinement_systems[obj_name], simulator.get_mutable_context())
        mustard_pose = pose_refinement_systems[obj_name].GetOutputPort(
            "X_WObject_refined").Eval(p_context)
        print obj_name, mustard_pose


    if args.meshcat:
        import meshcat.geometry as g
        mustard_box = g.Box([9.6/100., 19.13/100., 5.82/100.])
        mustard_transform = mustard_pose.matrix()
        meshcat.vis["mustard"].set_object(mustard_box)
        meshcat.vis["mustard"].set_transform(mustard_transform)

    import cv2
    # Show the annotated image.
    annotated_image = dope_system.GetOutputPort(
        "annotated_rgb_image").Eval(context).data
    cv2.imshow("dope image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

if __name__ == "__main__":
    main()  # This is what you would have, but the following is useful:

    # These are temporary, for debugging, so meh for programming style.
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
