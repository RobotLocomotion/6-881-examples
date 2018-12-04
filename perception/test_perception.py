import unittest
import numpy as np

from pydrake.examples.manipulation_station import ManipulationStation
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.common.eigen_geometry import Isometry3
import pydrake.perception as mut

from perception.point_cloud_to_pose_system import PointCloudToPoseSystem


class TestPointCloudToPoseSystem(unittest.TestCase):
    def setUp(self):
        builder = DiagramBuilder()

        station = builder.AddSystem(ManipulationStation())
        station.Finalize()

        # create the PointCloudToPoseSystem
        config_file = "perception/config/sim.yml"
        self.pc_to_pose = builder.AddSystem(
            PointCloudToPoseSystem(config_file, viz=False))

        # add systems to convert the depth images from ManipulationStation to
        # PointClouds
        left_camera_info = self.pc_to_pose.camera_configs["left_camera_info"]
        middle_camera_info = \
            self.pc_to_pose.camera_configs["middle_camera_info"]
        right_camera_info = self.pc_to_pose.camera_configs["right_camera_info"]

        left_dut = builder.AddSystem(
            mut.DepthImageToPointCloud(camera_info=left_camera_info))
        middle_dut = builder.AddSystem(
            mut.DepthImageToPointCloud(camera_info=middle_camera_info))
        right_dut = builder.AddSystem(
            mut.DepthImageToPointCloud(camera_info=right_camera_info))

        left_name_prefix = "camera_" + \
            self.pc_to_pose.camera_configs["left_camera_serial"]
        middle_name_prefix = "camera_" + \
            self.pc_to_pose.camera_configs["middle_camera_serial"]
        right_name_prefix = "camera_" + \
            self.pc_to_pose.camera_configs["right_camera_serial"]

        # connect the depth images to the DepthImageToPointCloud converters
        builder.Connect(
            station.GetOutputPort(left_name_prefix + "_depth_image"),
                                  left_dut.depth_image_input_port())
        builder.Connect(
            station.GetOutputPort(middle_name_prefix + "_depth_image"),
                                  middle_dut.depth_image_input_port())
        builder.Connect(
            station.GetOutputPort(right_name_prefix + "_depth_image"),
                                  right_dut.depth_image_input_port())

        # connect the rgb images to the PointCloudToPoseSystem
        builder.Connect(station.GetOutputPort(left_name_prefix + "_rgb_image"),
                        self.pc_to_pose.GetInputPort("camera_left_rgb"))
        builder.Connect(
            station.GetOutputPort(
                middle_name_prefix + "_rgb_image"),
                self.pc_to_pose.GetInputPort("camera_middle_rgb"))
        builder.Connect(station.GetOutputPort(right_name_prefix + "_rgb_image"),
                        self.pc_to_pose.GetInputPort("camera_right_rgb"))

        # connect the XYZ point clouds to PointCloudToPoseSystem
        builder.Connect(left_dut.point_cloud_output_port(),
                        self.pc_to_pose.GetInputPort("left_point_cloud"))
        builder.Connect(middle_dut.point_cloud_output_port(),
                        self.pc_to_pose.GetInputPort("middle_point_cloud"))
        builder.Connect(right_dut.point_cloud_output_port(),
                        self.pc_to_pose.GetInputPort("right_point_cloud"))

        diagram = builder.Build()

        simulator = Simulator(diagram)

        self.context = diagram.GetMutableSubsystemContext(self.pc_to_pose,
                                             simulator.get_mutable_context())

    def test_blank_system(self):
        pose = self.pc_to_pose.GetOutputPort("X_WObject").Eval(self.context)
        self.assertTrue(
            np.allclose(pose.matrix(), Isometry3.Identity().matrix()))
