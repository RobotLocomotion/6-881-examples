import argparse
import numpy as np

from pydrake.examples.manipulation_station import (
    ManipulationStation, ManipulationStationHardwareInterface,
    CreateDefaultYcbObjectList)
from pydrake.geometry import ConnectDrakeVisualizer
from pydrake.multibody.plant import MultibodyPlant
from pydrake.manipulation.planner import (
    DifferentialInverseKinematicsParameters)
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import (BasicVector, DiagramBuilder,
                                       LeafSystem)
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.systems.primitives import FirstOrderLowPassFilter, SignalLogger, TrajectorySource
from pydrake.trajectories import PiecewisePolynomial

import sys

try:
    import pygame
    from pygame.locals import *
except ImportError:
    print("ERROR: missing pygame.  Please install pygame to use this example.")
    # Fail silently (until pygame is supported in python 3 on all platforms)
    sys.exit(0)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--target_realtime_rate", type=float, default=1.0,
    help="Desired rate relative to real time.  See documentation for "
         "Simulator::set_target_realtime_rate() for details.")
parser.add_argument(
    "--duration", type=float, default=np.inf,
    help="Desired duration of the simulation in seconds.")
parser.add_argument(
    "--hardware", action='store_true',
    help="Use the ManipulationStationHardwareInterface instead of an "
         "in-process simulation.")
parser.add_argument(
    "--test", action='store_true',
    help="Disable opening the gui window for testing.")
parser.add_argument(
    "--filter_time_const", type=float, default=0.005,
    help="Time constant for the first order low pass filter applied to"
         "the teleop commands")
parser.add_argument(
    "--velocity_limit_factor", type=float, default=1.0,
    help="This value, typically between 0 and 1, further limits the iiwa14 "
         "joint velocities. It multiplies each of the seven pre-defined "
         "joint velocity limits. "
         "Note: The pre-defined velocity limits are specified by "
         "iiwa14_velocity_limits, found in this python file.")
parser.add_argument(
    '--setup', type=str, default='default',
    help="The manipulation station setup to simulate. ",
    choices=['default', 'clutter_clearing'])
parser.add_argument("log", type=str, help="Pickle file of logged run.")

MeshcatVisualizer.add_argparse_argument(parser)
args = parser.parse_args()

def _xyz_rpy(p, rpy):
    return RigidTransform(rpy=RollPitchYaw(rpy), p=p)

def CreateYcbObjectClutter():
    ycb_object_pairs = []

    X_WCracker = _xyz_rpy([0.35, 0.15, 0.09], [0, -1.57, 4])
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
    X_WMustard = _xyz_rpy([0.45, -0.16, 0.09], [-1.57, 0, 3.3])
    ycb_object_pairs.append(
        ("drake/manipulation/models/ycb/sdf/006_mustard_bottle.sdf",
         X_WMustard))

    # The gelatin box pose.
    # X_WGelatin = _xyz_rpy([0.35, -0.32, 0.1], [-1.57, 0, 3.7])
    # ycb_object_pairs.append(
    #    ("drake/manipulation/models/ycb/sdf/009_gelatin_box.sdf", X_WGelatin))

    # The potted meat can pose.
    X_WMeat = _xyz_rpy([0.35, -0.32, 0.03], [-1.57, 0, 2.5])
    ycb_object_pairs.append(
        ("drake/manipulation/models/ycb/sdf/010_potted_meat_can.sdf", X_WMeat))

    return ycb_object_pairs

import pickle
with open(args.log, "rb") as f:
    input_dict = pickle.load(f)

builder = DiagramBuilder()

if args.hardware:
    station = builder.AddSystem(ManipulationStationHardwareInterface())
    station.Connect(wait_for_cameras=False)
else:
    station = builder.AddSystem(ManipulationStation())

    # Initializes the chosen station type.
    if args.setup == 'default':
        station.SetupDefaultStation()
    elif args.setup == 'clutter_clearing':
        station.SetupDefaultStation()
        ycb_objects = CreateYcbObjectClutter()
        for model_file, X_WObject in ycb_objects:
            station.AddManipulandFromFile(model_file, X_WObject)

    station.Finalize()
    ConnectDrakeVisualizer(builder, station.get_scene_graph(),
                           station.GetOutputPort("pose_bundle"))
    if args.meshcat:
        meshcat = builder.AddSystem(MeshcatVisualizer(
            station.get_scene_graph(), zmq_url=args.meshcat, open_browser=args.open_browser))
        builder.Connect(station.GetOutputPort("pose_bundle"),
                        meshcat.get_input_port(0))

robot = station.get_controller_plant()

time_step = 0.001

end_time = 55
end_index = int(end_time * 1./time_step)

iiwa_position_ppt = PiecewisePolynomial.ZeroOrderHold(
    input_dict["iiwa_position_t"][:end_index],
    input_dict["iiwa_position_data"][:, :end_index])
iiwa_position_trajsource = builder.AddSystem(TrajectorySource(
    trajectory=iiwa_position_ppt,
    output_derivative_order=0,
    zero_derivatives_beyond_limits=True))

wsg_position_ppt = PiecewisePolynomial.ZeroOrderHold(
    input_dict["wsg_position_t"][:end_index],
    input_dict["wsg_position_data"][:, :end_index])
wsg_position_trajsource = builder.AddSystem(TrajectorySource(
    trajectory=wsg_position_ppt,
    output_derivative_order=0,
    zero_derivatives_beyond_limits=True))

builder.Connect(iiwa_position_trajsource.get_output_port(0),
                station.GetInputPort("iiwa_position"))

builder.Connect(wsg_position_trajsource.get_output_port(0),
                station.GetInputPort("wsg_position"))

diagram = builder.Build()
simulator = Simulator(diagram)

station_context = diagram.GetMutableSubsystemContext(
    station, simulator.get_mutable_context())

station_context.FixInputPort(station.GetInputPort(
    "iiwa_feedforward_torque").get_index(), np.zeros(7))
station_context.FixInputPort(station.GetInputPort(
    "wsg_force_limit").get_index(), np.zeros(1)+40)

station.SetIiwaPosition(station_context, input_dict["q0"])
q0 = station.GetOutputPort("iiwa_position_measured").Eval(station_context)
q0_initial = q0.copy()

# This is important to avoid duplicate publishes to the hardware interface:
simulator.set_publish_every_time_step(False)

simulator.set_target_realtime_rate(args.target_realtime_rate)

try:
    simulator.StepTo(args.duration)

    pose_bundle = station.GetOutputPort("pose_bundle").Eval(station_context)
    for i in range(pose_bundle.get_num_poses()):
        if pose_bundle.get_name(i):
            print pose_bundle.get_name(i), pose_bundle.get_pose(i)
except KeyboardInterrupt:
    print("Terminated")
