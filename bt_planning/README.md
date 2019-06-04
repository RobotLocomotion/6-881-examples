# Behavior Tree Planning

This directory was an experiment of controlling the robot using Behavior Trees
instead of PDDL. The code isn't fully functional, doesn't have tests, relies on
a fork of Drake that hasn't merged into master, and uses deprecated functions.

The main use of these files is for an example of a different way of controlling
the robot, and a starting point of writing a `System` that interfaces with the
newer version of `RobotPlanRunner`.

These files cannot currently be run in a Docker image.

## Files

- `behavior_tree.py`: `BehaviorTree` is a `LeafSystem` that sends plans to the
robot through a `RobotPlanRunner`, available on [this fork](
https://github.com/pangtao22/drake/tree/robot_plan_runner/manipulation/robot_plan_runner) 
of Drake. It wraps a `py_trees.BehaviourTree` object in a Drake `System` to run 
a stowing task of opening a cabinet door, picking up an object, and placing it
on one of the cabinet shelves.

- `behaviors.py` contains many `py_trees.Behaviour` objects to construct various
trees to perform the stowing task.

- `planning.py` contains helper methods to plan trajectories the robot can
follow.

## Using This Module

### py_trees
In addition to relying on Drake Python bindings built from the previously listed
`RobotPlanRunner` [fork](
https://github.com/pangtao22/drake/tree/robot_plan_runner/manipulation/robot_plan_runner), 
these files depend on version 0.5 of the 
[py_trees](https://github.com/splintered-reality/py_trees/tree/release/0.5.x) 
library written by Daniel Stonier.
 
### Running the BT
`behavior_tree.py` is an executable file to run a subset of the stowing task and
visualize it in Meshcat. There are three different options of subtasks to run,
which are selected by modifying the line selecting the `root` variable towards
the top of the `main()` function. The line currently reads
```py
root = pick_test(obj_name="gelatin")
```
`pick_test` can be replaced with either `open_door_test` or `make_root`.

To run the behavior tree, just call the script from the command line:
```sh
$ python behavior_tree.py
```

This will print out a link to a meshcat URL to visit to visualize the robot, as
well as print out the status of the behavior tree as its running. `pick_test` is
currently the only working subtask. A video of a successful run can be found 
[here](https://youtu.be/5cm3LV2dMOo).