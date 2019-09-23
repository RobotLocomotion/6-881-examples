import os

import graphviz
import numpy as np

from pydrake.examples.manipulation_station import ManipulationStation
from pydrake.systems.analysis import Simulator
from pydrake.systems.perception import PointCloudConcatenation
from pydrake.perception import DepthImageToPointCloud, BaseField
from pydrake.systems.framework import DiagramBuilder, AbstractValue
from pydrake.systems.sensors import PixelType
from pydrake.systems.meshcat_visualizer import (MeshcatVisualizer,
    MeshcatPointCloudVisualizer)
from pydrake.common import FindResourceOrThrow

from pydrake.math import RigidTransform, RollPitchYaw

import meshcat

from perception_tools.file_utils import LoadCameraConfigFile


def RenderSystemWithGraphviz(system, output_file="system_view.gz"):
    """ Renders the Drake system (presumably a diagram,
    otherwise this graph will be fairly trivial) using
    graphviz to a specified file. """
    from graphviz import Source
    string = system.GetGraphvizString()
    src = Source(string)
    src.render(output_file, view=False)


def UpdateTriad(vis, X_WT, name, prefix):
    """
    Draws coordinate axes of frame T whose transformation in world frame is
    X_WT. The triad needs to be initialized by calling AddTriad.
    Args:
        vis: a meshcat.Visualizer object.
        X_WT: a RigidTransform object.
        name: (string) the name of the triad in meshcat.
        prefix: (string) name of the node in the meshcat tree to which this
            triad is added.
    """
    vis[prefix][name].set_transform(X_WT.matrix())


def DrawTriad(vis, X_WT, name="triad", scale=1., opacity=1.):
    length = 1 * scale
    radius = 0.040 * scale
    delta_xyz = np.array([[length / 2, 0, 0],
                          [0, length / 2, 0],
                          [0, 0, length / 2]])

    axes_name = ['x', 'y', 'z']
    colors = [0xff0000, 0x00ff00, 0x0000ff]
    rotation_axes = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]

    for i in range(3):
        material = meshcat.geometry.MeshLambertMaterial(
            color=colors[i], opacity=opacity)
        vis[name][axes_name[i]].set_object(
            meshcat.geometry.Cylinder(length, radius), material)
        X = meshcat.transformations.rotation_matrix(
            np.pi/2, rotation_axes[i])
        X[0:3, 3] = delta_xyz[i]
        vis[name][axes_name[i]].set_transform(X)

    vis[name].set_transform(X_WT.matrix())

#%%
camera_configs = LoadCameraConfigFile(
    os.path.join(os.getcwd(), "config", "sim.yml"))


#%%

builder = DiagramBuilder()

# Add manipulation station
station = ManipulationStation()
builder.AddSystem(station)
station.SetupManipulationClassStation()

# add manipuland
foam_brick_sdf = "drake/examples/manipulation_station/models/061_foam_brick.sdf"

X_WO = RigidTransform()
X_WO.set_translation([0.4, -0.2, 0])

station.AddManipulandFromFile(foam_brick_sdf, X_WO)
station.Finalize()


camera_name_list = station.get_camera_names()

pc_concat = builder.AddSystem(PointCloudConcatenation(camera_name_list))


# Create the DepthImageToPointClouds.
# use scale factor of 1/1000 to convert mm to m
left_serial = "2"
middle_serial = "1"
right_serial = "0"

di2pcs = {}
for serial_num in [left_serial, middle_serial, right_serial]:
    di2pcs[serial_num] = builder.AddSystem(DepthImageToPointCloud(
        camera_configs[serial_num]["camera_info"], PixelType.kDepth16U, 1e-3,
        fields=BaseField.kXYZs | BaseField.kRGBs))


# Connect the depth and rgb images to manipulation station
for name in station.get_camera_names():
    builder.Connect(
        station.GetOutputPort("camera_" + name + "_rgb_image"),
        di2pcs[name].color_image_input_port())
    builder.Connect(
        station.GetOutputPort("camera_" + name + "_depth_image"),
        di2pcs[name].depth_image_input_port())


for camera_name in camera_name_list:
    builder.Connect(di2pcs[camera_name].point_cloud_output_port(),
                    pc_concat.GetInputPort(
                        "point_cloud_CiSi_{}".format(camera_name)))

# visualizer
frames_to_draw = {"iiwa": {"iiwa_link_7", "iiwa_link_6"},
                  "foam_brick": {"base_link"}}


vis = builder.AddSystem(MeshcatVisualizer(
    station.get_scene_graph(), zmq_url="tcp://127.0.0.1:6000",
    frames_to_draw=frames_to_draw))

builder.Connect(station.GetOutputPort("pose_bundle"),
                vis.get_input_port(0))

scene_pc_vis = builder.AddSystem(MeshcatPointCloudVisualizer(
    vis, name="scene_point_cloud"))
builder.Connect(pc_concat.GetOutputPort("point_cloud_FS"),
                scene_pc_vis.GetInputPort("point_cloud_P"))


diagram = builder.Build()

q0 = np.array([0, 0, 0, -1.75, 0, 1.0, 0])


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

for camera_name in camera_name_list:
    X_WP = camera_configs[camera_name]["camera_pose_world"].multiply(
        camera_configs[camera_name]["camera_pose_internal"])
    pc_concat_context.FixInputPort(
        pc_concat.GetInputPort("X_FCi_{}".format(camera_name)).get_index(),
        AbstractValue.Make(X_WP))

# Set door angles
left_hinge_joint = station.get_multibody_plant().GetJointByName(
    "left_door_hinge")
left_hinge_joint.set_angle(station_context, angle=-np.pi / 2)
right_hinge_joint = station.get_multibody_plant().GetJointByName(
    "right_door_hinge")
right_hinge_joint.set_angle(station_context, angle=0)


simulator.set_publish_every_time_step(False)
simulator.set_target_realtime_rate(0.0)  # go as fast as possible

simulator.Initialize()
simulator.AdvanceTo(0.05)

#%% show camera frames
for name in station.get_camera_names():
    X = camera_configs[name]["camera_pose_world"].multiply(
        camera_configs[name]["camera_pose_internal"])
    print(name)
    print(camera_configs[name]["camera_pose_world"].translation())
    print(X.translation())
    print(RollPitchYaw(X.rotation()).vector())

    DrawTriad(vis.vis, X, name=name, scale=0.15)


#%% get point cloud from output port
pc_concat_context = diagram.GetSubsystemContext(
    pc_concat, simulator.get_context())

pc = pc_concat.GetOutputPort("point_cloud_FS").Eval(pc_concat_context)

color = pc.rgbs()
position = pc.xyzs()

np.save("position", position)
np.save("color", color)