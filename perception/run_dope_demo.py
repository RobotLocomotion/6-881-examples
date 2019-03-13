import argparse

import numpy as np

from pydrake.examples.manipulation_station import (
    ManipulationStation, _xyz_rpy)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.sensors import PixelType
import pydrake.perception as mut

from dope_system import DopeSystem
from pose_refinement import PoseRefinement


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

    X_WCracker = _xyz_rpy([-0.3, -0.55, 0.36], [-1.57, 0, 3])
    ycb_object_pairs.append(
        ("drake/manipulation/models/ycb/sdf/003_cracker_box.sdf", X_WCracker))

    # The sugar box pose.
    X_WSugar = _xyz_rpy([-0.3, -0.7, 0.33], [1.57, 1.57, 0])
    ycb_object_pairs.append(
        ("drake/manipulation/models/ycb/sdf/004_sugar_box.sdf", X_WSugar))

    # The tomato soup can pose.
    X_WSoup = _xyz_rpy([-0.03, -0.57, 0.31], [-1.57, 0, 3.14])
    ycb_object_pairs.append(
        ("drake/manipulation/models/ycb/sdf/005_tomato_soup_can.sdf", X_WSoup))

    # The mustard bottle pose.
    X_WMustard = _xyz_rpy([0.05, -0.66, 0.35], [-1.57, 0, 3.3])
    ycb_object_pairs.append(
        ("drake/manipulation/models/ycb/sdf/006_mustard_bottle.sdf",
         X_WMustard))

    # The gelatin box pose.
    X_WGelatin = _xyz_rpy([-0.15, -0.62, 0.38], [-1.57, 0, 3.7])
    ycb_object_pairs.append(
        ("drake/manipulation/models/ycb/sdf/009_gelatin_box.sdf", X_WGelatin))

    # The potted meat can pose.
    X_WMeat = _xyz_rpy([-0.15, -0.62, 0.3], [-1.57, 0, 2.5])
    ycb_object_pairs.append(
        ("drake/manipulation/models/ycb/sdf/010_potted_meat_can.sdf", X_WMeat))

    return ycb_object_pairs


if __name__ == "__main__":
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
            camera_config_file, model_files[obj], image_files[obj], obj))

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
        right_camera_info, PixelType.kDepth16U, 1e-3))

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

    # TODO(kmuhlrad): figure out how to connect DopeSystem and PoseRefinement
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

    print "done"
