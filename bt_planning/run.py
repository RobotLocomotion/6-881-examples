from behaviors import (
    RobotMoving, Holding, On, DoorOpen, Pick, Place, OpenDoor)

import py_trees
from py_trees.composites import Selector, Sequence
from py_trees.meta import inverter
from py_trees.common import Status

import time
import numpy as np

def make_root():

    # conditions
    # robot_moving = RobotMoving()
    holding_soup = Holding("soup")
    soup_on_shelf = On("soup", "bottom_shelf")
    left_door_open = DoorOpen("left_door")

    # actions
    pick_soup = Pick("soup")
    place_soup = Place("soup", "bottom_shelf")
    open_left_door = OpenDoor("left_door")

    root = Selector(name="Root")

    soup_on_shelf_seq = Sequence(name="SoupOnShelfSeq")

    open_door_sel = Selector(name="OpenDoorSel")
    open_door_seq = Sequence(name="OpenDoorSeq")

    pick_soup_sel = Selector(name="PickSoupSel")
    pick_soup_seq = Sequence(name="PickSoupSeq")

    moving_inverter = inverter(RobotMoving)("MovingInverter")
    gripper_empty = inverter(Holding)("soup", "GripperEmpty")

    soup_on_shelf_seq.add_children([open_door_sel, pick_soup_sel, place_soup])
    open_door_sel.add_children([left_door_open, open_door_seq])
    open_door_seq.add_children(
        [gripper_empty, moving_inverter, open_left_door])

    pick_soup_sel.add_children([holding_soup, pick_soup_seq])
    pick_soup_seq.add_children([gripper_empty, moving_inverter, pick_soup])

    root.add_children([soup_on_shelf, soup_on_shelf_seq])

    return root


def init_blackboard():
    blackboard = py_trees.blackboard.Blackboard()

    blackboard.set("robot_moving", False)
    blackboard.set("robot_holding", None)
    blackboard.set("soup_on", "table")
    blackboard.set("left_door_open", False)
    blackboard.set("right_door_open", True)


if __name__ == "__main__":
    py_trees.logging.level = py_trees.logging.Level.INFO
    root = make_root()
    root.setup(timeout=15)

    # py_trees.display.render_dot_tree(root)

    # for node in root.tick():
    #     print(node, "\n")

    init_blackboard()

    i = 0
    while not root.status == Status.SUCCESS:
        print("\n--------- Tick {0} ---------\n".format(i))
        root.tick_once()
        print("\n")
        print("{}".format(py_trees.display.print_ascii_tree(root, show_status=True)))
        print(py_trees.blackboard.Blackboard())
        time.sleep(1.0)
        i += 1