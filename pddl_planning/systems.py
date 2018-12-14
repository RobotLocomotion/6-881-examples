from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import Demultiplexer, LogOutput

def connect_plan_runner(builder, station, plan):
    from plan_runner.manipulation_station_plan_runner import ManipStationPlanRunner
    plan_list, gripper_setpoints = plan

    # Add plan runner.
    plan_runner = ManipStationPlanRunner(plan_list, gripper_setpoints)

    builder.AddSystem(plan_runner)
    builder.Connect(plan_runner.hand_setpoint_output_port,
                    station.GetInputPort("wsg_position"))
    builder.Connect(plan_runner.gripper_force_limit_output_port,
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
                    plan_runner.iiwa_position_input_port)
    builder.Connect(station.GetOutputPort("iiwa_velocity_estimated"),
                    plan_runner.iiwa_velocity_input_port)
    builder.Connect(station.GetOutputPort("iiwa_torque_external"),
                    plan_runner.GetInputPort("iiwa_torque_external"))

    # Add logger
    iiwa_position_command_log = LogOutput(demux.get_output_port(0), builder)
    iiwa_position_command_log._DeclarePeriodicPublish(0.005)

    iiwa_external_torque_log = LogOutput(
        station.GetOutputPort("iiwa_torque_external"), builder)
    iiwa_external_torque_log._DeclarePeriodicPublish(0.005)

    iiwa_position_measured_log = LogOutput(
        station.GetOutputPort("iiwa_position_measured"), builder)
    iiwa_position_measured_log._DeclarePeriodicPublish(0.005)

    wsg_state_log = LogOutput(
        station.GetOutputPort("wsg_state_measured"), builder)
    wsg_state_log._DeclarePeriodicPublish(0.1)

    wsg_command_log = LogOutput(
        plan_runner.hand_setpoint_output_port, builder)
    wsg_command_log._DeclarePeriodicPublish(0.1)

    return plan_runner

def build_manipulation_station(station, plan=None, visualize=False):
    from underactuated.meshcat_visualizer import MeshcatVisualizer

    builder = DiagramBuilder()
    builder.AddSystem(station)

    plan_runner = None
    if plan is not None:
        plan_runner = connect_plan_runner(builder, station, plan)

    # Add meshcat visualizer
    plant = station.get_mutable_multibody_plant()
    scene_graph = station.get_mutable_scene_graph()
    if visualize:
        viz = MeshcatVisualizer(scene_graph,
                                is_drawing_contact_force = True,
                                plant = plant)
        builder.AddSystem(viz)
        builder.Connect(station.GetOutputPort("pose_bundle"),
                        viz.GetInputPort("lcm_visualization"))
        builder.Connect(station.GetOutputPort("contact_results"),
                        viz.GetInputPort("contact_results"))

    # build diagram
    diagram = builder.Build()
    if visualize:
        viz.load()
    #RenderSystemWithGraphviz(diagram)
    return diagram, plan_runner

##################################################


def RenderSystemWithGraphviz(system, output_file="system_view.gz"):
    ''' Renders the Drake system (presumably a diagram,
    otherwise this graph will be fairly trivial) using
    graphviz to a specified file. '''
    from graphviz import Source
    string = system.GetGraphvizString()
    src = Source(string)
    src.render(output_file, view=False)
