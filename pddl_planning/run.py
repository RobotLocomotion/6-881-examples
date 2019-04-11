from __future__ import print_function

import argparse
import random
import numpy as np
import cProfile
import pstats

from pddl_planning.generators import Pose, Conf, get_ik_gen_fn, get_reachable_grasp_gen_fn, \
    get_reachable_pose_gen_fn, get_motion_fn, get_pull_fn, get_collision_test, \
    get_open_trajectory, Trajectory, get_force_pull_fn
from iiwa_utils import get_door_positions, DOOR_OPEN
from pddl_planning.simulation import compute_duration, convert_controls, step_trajectories, dump_plans, ForceControl
from pddl_planning.problems import load_station, load_dope, DOPE_PATH, get_sdf_path
from pddl_planning.systems import RenderSystemWithGraphviz
from pddl_planning.utils import get_world_pose, get_configuration, get_model_name, get_joint_positions, get_parent_joints, \
    get_state, set_state, get_movable_joints

from pddlstream.algorithms.focused import solve_focused
from pddlstream.language.constants import And, PDDLProblem
from pddlstream.language.function import FunctionInfo
from pddlstream.language.generator import from_gen_fn, from_fn
from pddlstream.utils import print_solution, read, INF, get_file_path

from plan_runner.manipulation_station_simulator import ManipulationStationSimulator
from plan_runner.open_left_door import GenerateOpenLeftDoorPlansByImpedanceOrPosition

from pydrake.math import RigidTransform
from pydrake.common.eigen_geometry import Isometry3

def get_pddlstream_problem(task, context, collisions=True, use_impedance=False):
    domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
    stream_pddl = read(get_file_path(__file__, 'stream.pddl'))
    constant_map = {}

    plant = task.mbp
    robot = task.robot
    robot_name = get_model_name(plant, robot)

    world = plant.world_frame() # mbp.world_body()
    robot_joints = get_movable_joints(plant, robot)
    robot_conf = Conf(robot_joints, get_configuration(plant, context, robot))
    init = [
        ('Robot', robot_name),
        ('CanMove', robot_name),
        ('Conf', robot_name, robot_conf),
        ('AtConf', robot_name, robot_conf),
        ('HandEmpty', robot_name),
    ]
    goal_literals = []
    if task.reset_robot:
        goal_literals.append(('AtConf', robot_name, robot_conf),)

    for obj in task.movable:
        obj_name = get_model_name(plant, obj)
        obj_pose = Pose(plant, world, obj, get_world_pose(plant, context, obj))
        print("OBJECT POSE", get_world_pose(plant, context, obj))
        init += [('Graspable', obj_name),
                 ('Pose', obj_name, obj_pose),
                 ('InitPose', obj_name, obj_pose),
                 ('AtPose', obj_name, obj_pose)]
        for surface in task.surfaces:
            init += [('Stackable', obj_name, surface)]

    for surface in task.surfaces:
        surface_name = get_model_name(plant, surface.model_index)
        if 'sink' in surface_name:
            init += [('Sink', surface)]
        if 'stove' in surface_name:
            init += [('Stove', surface)]

    for door in task.doors:
        door_body = plant.get_body(door)
        door_name = door_body.name()
        door_joints = get_parent_joints(plant, door_body)
        door_conf = Conf(door_joints, get_joint_positions(door_joints, context))
        init += [
            ('Door', door_name),
            ('Conf', door_name, door_conf),
            ('AtConf', door_name, door_conf),
        ]
        for positions in [get_door_positions(door_body, DOOR_OPEN)]:
            conf = Conf(door_joints, positions)
            init += [('Conf', door_name, conf)]
        if task.reset_doors:
            goal_literals += [('AtConf', door_name, door_conf)]

    for obj, transform in task.goal_poses.items():
        obj_name = get_model_name(plant, obj)
        obj_pose = Pose(plant, world, obj, transform)
        init += [
            ('Pose', obj_name, obj_pose),
        ]
        goal_literals.append(('AtPose', obj_name, obj_pose))
    for obj in task.goal_holding:
        goal_literals.append(('Holding', robot_name, get_model_name(plant, obj)))
    for obj, surface in task.goal_on:
        goal_literals.append(('On', get_model_name(plant, obj), surface))
    for obj in task.goal_cooked:
        goal_literals.append(('Cooked', get_model_name(plant, obj)))

    goal = And(*goal_literals)
    print('Initial:', init)
    print('Goal:', goal)

    stream_map = {
        'sample-reachable-grasp': from_gen_fn(get_reachable_grasp_gen_fn(task, context, collisions=collisions)),
        'sample-reachable-pose': from_gen_fn(get_reachable_pose_gen_fn(task, context, collisions=collisions)),
        'plan-ik': from_gen_fn(get_ik_gen_fn(task, context, collisions=collisions)),
        'plan-motion': from_fn(get_motion_fn(task, context, collisions=collisions)),
        'TrajPoseCollision': get_collision_test(task, context, collisions=collisions),
        'TrajConfCollision': get_collision_test(task, context, collisions=collisions),
    }
    if use_impedance:
        stream_map['plan-pull'] = from_gen_fn(get_force_pull_fn(task, context, collisions=collisions))
    else:
        stream_map['plan-pull'] = from_gen_fn(get_pull_fn(task, context, collisions=collisions))
    #stream_map = 'debug' # Runs PDDLStream with "placeholder streams" for debugging

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)


def postprocess_plan(plant, gripper, plan):
    trajectories = []
    if plan is None:
        return trajectories
    open_traj = get_open_trajectory(plant, gripper)
    close_traj = open_traj.reverse()

    for name, args in plan:
        if name in ['clean', 'cook']:
            continue
        traj = args[-1]
        if name == 'pick':
            trajectories.extend([Trajectory(reversed(traj.path)), close_traj, traj])
        elif name == 'place':
            trajectories.extend([traj.reverse(), open_traj, Trajectory(traj.path)])
        elif name == 'pull':
            trajectories.extend([close_traj, traj, open_traj])
        elif name == 'move':
            trajectories.append(traj)
        else:
            raise NotImplementedError(name)
    return trajectories


def plan_trajectories(task, context, collisions=True, use_impedance=False, max_time=180):
    stream_info = {
        'TrajPoseCollision': FunctionInfo(p_success=1e-3),
        'TrajConfCollision': FunctionInfo(p_success=1e-3),
    }
    pr = cProfile.Profile()
    pr.enable()
    problem = get_pddlstream_problem(task, context, collisions=collisions, use_impedance=use_impedance)
    solution = solve_focused(problem, stream_info=stream_info, planner='ff-wastar2',
                             max_time=max_time, search_sampling_ratio=0)
    pr.disable()
    pstats.Stats(pr).sort_stats('tottime').print_stats(5)
    print_solution(solution)
    plan, cost, evaluations = solution
    if plan is None:
        print('Unable to find a solution in under {} seconds'.format(max_time))
        return None
    return postprocess_plan(task.mbp, task.gripper, plan)

##################################################

def replan(task, context, visualize=True, collisions=True, use_impedance=False):
    initial_state = get_state(task.plant, context)
    trajectories = plan_trajectories(task, context, collisions=collisions, use_impedance=use_impedance)
    if trajectories is None:
        return

    set_state(task.plant, context, initial_state)
    if visualize:
        step_trajectories(task.diagram, task.diagram_context, context, trajectories,
                         time_step=None, teleport=True)
                         # time_step=0.001)
    splines, gripper_setpoints = convert_controls(
        task.plant, task.robot, task.gripper, context, trajectories)
    np.save("splines", splines, allow_pickle=True)
    np.save("gripper_setpoints", gripper_setpoints, allow_pickle=True)

    return splines, gripper_setpoints

##################################################

def main():
    time_step = 2e-3

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfree', action='store_true',
                        help='Disables collisions when planning')
    parser.add_argument('-d', '--deterministic', action='store_true',
                        help='Manually sets the random seeds used by the stream generators')
    parser.add_argument('-s', '--simulate', action='store_true',
                        help='Simulates the system')
    parser.add_argument('-e', '--execute', action='store_true',
                        help='Executes the system')
    parser.add_argument('-l', '--load', action='store_true',
                        help='Loads the last plan')
    parser.add_argument('-f', '--force_control', action='store_true',
                        help='Use impedance control to open the door')
    parser.add_argument('-p', '--poses', default=DOPE_PATH,
                        help='The path to the dope poses file')
    args = parser.parse_args()

    if args.deterministic:
        random.seed(0)
        np.random.seed(0)

    goal_name = 'soup'
    if args.poses == 'none':
        task, diagram, state_machine = load_station(time_step=time_step)
    else:
        task, diagram, state_machine = load_dope(time_step=time_step, dope_path=args.poses, goal_name=goal_name)
    print(task)

    plant = task.mbp
    RenderSystemWithGraphviz(diagram) # Useful for getting port names

    task.publish()
    context = diagram.GetMutableSubsystemContext(plant, task.diagram_context)

    world_frame = plant.world_frame()
    X_WSoup = plant.CalcRelativeTransform(
        context, frame_A=world_frame, frame_B=plant.GetFrameByName("base_link_soup"))
    print("SOUP_POSE", X_WSoup.matrix())
    if not args.load:
        replan(task, context, visualize=True, collisions=not args.cfree, use_impedance=args.force_control)

    ##################################################

    if not args.simulate and not args.execute:
        return

    manip_station_sim = ManipulationStationSimulator(
        time_step=time_step,
        object_file_path=get_sdf_path(goal_name),
        object_base_link_name="base_link_soup",
        X_WObject=X_WSoup)

    plan_list = []
    gripper_setpoints = []
    splines = np.load("splines.npy")
    setpoints = np.load("gripper_setpoints.npy")
    for control, setpoint in zip(splines, setpoints):
        if isinstance(control, ForceControl):
            new_plans, new_setpoints = \
                GenerateOpenLeftDoorPlansByImpedanceOrPosition("Impedance", is_open_fully=True)
            plan_list.extend(new_plans)
            gripper_setpoints.extend(new_setpoints)
        else:
            plan_list.append(control.plan())
            gripper_setpoints.append(setpoint)

    dump_plans(plan_list, gripper_setpoints)
    sim_duration = compute_duration(plan_list)
    print('Splines: {}\nDuration: {:.3f} seconds'.format(len(plan_list), sim_duration))

    if args.execute:
        raw_input('Execute on hardware?')
        iiwa_position_command_log, iiwa_position_measured_log, iiwa_external_torque_log = \
            manip_station_sim.RunRealRobot(plan_list, gripper_setpoints)
        #PlotExternalTorqueLog(iiwa_external_torque_log)
        #PlotIiwaPositionLog(iiwa_position_command_log, iiwa_position_measured_log)
    else:
        raw_input('Execute in simulation?')
        q0 = [0, 0, 0, -1.75, 0, 1.0, 0]
        iiwa_position_command_log, iiwa_position_measured_log, iiwa_external_torque_log, \
            plant_state_log = \
            manip_station_sim.RunSimulation(plan_list, gripper_setpoints,
                                            extra_time=2.0, real_time_rate=1.0, q0_kuka=q0)
        #PlotExternalTorqueLog(iiwa_external_torque_log)
        #PlotIiwaPositionLog(iiwa_position_command_log, iiwa_position_measured_log)


if __name__ == '__main__':
    main()
