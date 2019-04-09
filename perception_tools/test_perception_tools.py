import unittest
import numpy as np

from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import AbstractValue, DiagramBuilder
from pydrake.math import RigidTransform
import pydrake.perception as mut

from perception_tools.point_cloud_synthesis import PointCloudSynthesis

class TestPointCloudSynthesis(unittest.TestCase):
    def setUp(self):
        builder = DiagramBuilder()

        X_W0 = RigidTransform.Identity()
        X_W1 = RigidTransform.Identity()
        X_W1.set_translation([1.0, 0, 0])

        transform_dict = {
            "0": X_W0,
            "1": X_W1,
        }

        self.pc_synth = builder.AddSystem(
            PointCloudSynthesis(transform_dict))

        num_points = 100000
        xyzs = np.random.uniform(-0.1, 0.1, (3, num_points))
        # only go to 254 to distinguish between point clouds with and without
        # color
        rgbs = np.random.uniform(0., 254.0, (3, num_points))

        self.pc = mut.PointCloud(
            num_points, mut.Fields(mut.BaseField.kXYZs | mut.BaseField.kRGBs))
        self.pc.mutable_xyzs()[:] = xyzs
        self.pc.mutable_rgbs()[:] = rgbs

        self.pc_no_rgbs = mut.PointCloud(
            num_points, mut.Fields(mut.BaseField.kXYZs))
        self.pc_no_rgbs.mutable_xyzs()[:] = xyzs

        diagram = builder.Build()

        simulator = Simulator(diagram)

        self.context = diagram.GetMutableSubsystemContext(
            self.pc_synth, simulator.get_mutable_context())

    def test_no_rgb(self):
        self.context.FixInputPort(
            self.pc_synth.GetInputPort("point_cloud_P_0").get_index(),
            AbstractValue.Make(self.pc_no_rgbs))
        self.context.FixInputPort(
            self.pc_synth.GetInputPort("point_cloud_P_1").get_index(),
            AbstractValue.Make(self.pc_no_rgbs))

        fused_pc = self.pc_synth.GetOutputPort("combined_point_cloud_W").Eval(
            self.context)

        self.assertEqual(fused_pc.size(), 200000)

        # the first point cloud should be from [-0.1 to 0.1]
        # the second point cloud should be from [0.9 to 1.1]
        self.assertTrue(np.max(fused_pc.xyzs()[0, :]) >= 1.0)
        self.assertTrue(np.min(fused_pc.xyzs()[0, :]) <= 0.0)

        # even if both input point clouds don't have rgbs, the fused point
        # cloud should contain rgbs of the default color
        self.assertTrue(fused_pc.has_rgbs())
        self.assertTrue(
            np.all(fused_pc.rgbs()[:, 0] == np.array([255, 255, 255])))
        self.assertTrue(
            np.all(fused_pc.rgbs()[:, -1] == np.array([255, 255, 255])))

    def test_rgb(self):
        self.context.FixInputPort(
            self.pc_synth.GetInputPort("point_cloud_P_0").get_index(),
            AbstractValue.Make(self.pc))
        self.context.FixInputPort(
            self.pc_synth.GetInputPort("point_cloud_P_1").get_index(),
            AbstractValue.Make(self.pc))

        fused_pc = self.pc_synth.GetOutputPort("combined_point_cloud_W").Eval(
            self.context)

        self.assertEqual(fused_pc.size(), 200000)

        # the first point cloud should be from [-0.1 to 0.1]
        # the second point cloud should be from [0.9 to 1.1]
        self.assertTrue(np.max(fused_pc.xyzs()[0, :]) >= 1.0)
        self.assertTrue(np.min(fused_pc.xyzs()[0, :]) <= 0.0)

        self.assertTrue(fused_pc.has_rgbs())
        self.assertTrue(
            np.all(fused_pc.rgbs()[:, 0] != np.array([255, 255, 255])))
        self.assertTrue(
            np.all(fused_pc.rgbs()[:, -1] != np.array([255, 255, 255])))

    def test_mix_rgb(self):
        self.context.FixInputPort(
            self.pc_synth.GetInputPort("point_cloud_P_0").get_index(),
            AbstractValue.Make(self.pc))
        self.context.FixInputPort(
            self.pc_synth.GetInputPort("point_cloud_P_1").get_index(),
            AbstractValue.Make(self.pc_no_rgbs))

        fused_pc = self.pc_synth.GetOutputPort("combined_point_cloud_W").Eval(
            self.context)

        self.assertEqual(fused_pc.size(), 200000)

        # the first point cloud should be from [-0.1 to 0.1]
        # the second point cloud should be from [0.9 to 1.1]
        self.assertTrue(np.max(fused_pc.xyzs()[0, :]) >= 1.0)
        self.assertTrue(np.min(fused_pc.xyzs()[0, :]) <= 0.0)


        self.assertTrue(fused_pc.has_rgbs())

        # We don't know what order the two point clouds will be combined
        rgb_first = np.all(fused_pc.rgbs()[:, 0] != np.array([255, 255, 255]))
        rgb_last = np.all(fused_pc.rgbs()[:, -1] != np.array([255, 255, 255]))
        no_rgb_first = np.all(
            fused_pc.rgbs()[:, 0] == np.array([255, 255, 255]))
        no_rgb_last = np.all(
            fused_pc.rgbs()[:, -1] == np.array([255, 255, 255]))

        self.assertTrue(
            (rgb_first and no_rgb_last) or (no_rgb_first and rgb_last))
