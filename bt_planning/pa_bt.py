import copy
import time
from collections import OrderedDict
import pdb
import numpy as np

from behaviors import (
    RobotMoving, Holding, On, DoorOpen, Pick, Place, OpenDoor)

import py_trees
from py_trees.trees import BehaviourTree
from py_trees.composites import Sequence, Selector
from py_trees.meta import inverter
from py_trees.visitors import VisitorBase
from py_trees.behaviour import Behaviour
from py_trees.common import Status
from py_trees.blackboard import Blackboard


class PPA:
    def __init__(self, action_name, pre, post):
        self.action_name = action_name
        self.pre = pre
        self.post = post

    def __repr__(self):        
        pre_text = ""
        for pre in self.pre:
            pre_text += "\t{}: {}\n".format(pre, self.pre[pre])

        post_text = ""
        for post in self.post:
            post_text += "\t{}: {}\n".format(post, self.post[post])

        return ("{}\n"
                "Preconditions:\n"
                "{}"
                "Postconditions:\n"
                "{}").format(self.action_name, pre_text, post_text)


class Condition(Behaviour):
    def __init__(self, condition_key, condition_value):
        super(Condition, self).__init__(name="{}_is_{}_Condition".format(condition_key, condition_value))
        self.condition_key = condition_key
        self.condition_value = condition_value
        self.blackboard = Blackboard()

    def update(self):
        if self.blackboard.get(self.condition_key) == self.condition_value:
            self.feedback_message = "{} is {}".format(self.condition_key, self.condition_value)
            return Status.SUCCESS
        self.feedback_message = "{} is not {}".format(self.condition_key, self.condition_value)
        return Status.FAILURE


class Action(Behaviour):
    def __init__(self, action):
        super(Action, self).__init__(name="{}_Action".format(action.action_name))
        self.blackboard = Blackboard()
        self.action = action

    def initialise(self):
        self.counter = 0

    def update(self):
        if self.counter < 1:
            for key in self.action.pre:
                if not key in self.action.post or self.action.post[key] == self.action.pre[key]:
                    self.blackboard.set(key, not self.action.pre[key])
            self.counter += 1
            return Status.RUNNING

        for key in self.action.post:
            self.blackboard.set(key, self.action.post[key])

        return Status.FAILURE
        # return Status.SUCCESS


class ConditionBFSVisitor(VisitorBase):
    def __init__(self, full=False):
        super(ConditionBFSVisitor, self).__init__(full=full)
        self.nodes = {}
        self.failing_nodes = []
        self.previously_failing_nodes = []

    def initialise(self):
        self.nodes = {}
        self.previously_failing_nodes = self.failing_nodes
        self.failing_nodes = []

    def run(self, behaviour):
        self.nodes[behaviour.id] = behaviour.status
        if (type(behaviour) == Condition
            and behaviour.status == Status.FAILURE):
            self.failing_nodes.append(behaviour)


class LastActionVisitor(VisitorBase):
    def __init__(self, full=False):
        super(LastActionVisitor, self).__init__(full=full)
        self.last_action = None
        self.level = 0

    def initialise(self):
        self.last_action = None
        self.level = 0

    def run(self, behaviour):
        if type(behaviour) == Action:
            self.last_action = behaviour


def get_node_level(node):
    level = 0
    while node.parent:
        level += 1
        node = node.parent
    return level


def get_all_action_templates_for(ppa_set, failed_condition):
    '''
    Finds all actions whose postconditions satisfies the failed_condition
    
    Args:
        ppa_set: a set of PPA objects
        failed_condition: a condition Behaviour that failed

    Returns:
        action_templates: a set of PPA objects that satisfy failed_condition
    '''
    action_templates = set()
    for action in ppa_set:
        if failed_condition.condition_key in action.post:
            if failed_condition.condition_value == action.post[failed_condition.condition_key]:
                action_templates.add(action)
    return action_templates


def expand_tree(tree, ppa_set, failed_condition, last_action):
    '''
    The ExpandTree function of the PA-BT algorithm. 'Algorithm 2'.

    Args:
        tree: the current BehaviourTree
        failed_condition: a condition Behavior that failed while ticking
            tree

    Returns:
        tree: the new BehaviourTree
        tree_sel: the new subtree with a Selector node
    '''
    conflict = False
    action_templates = get_all_action_templates_for(ppa_set, failed_condition)
    tree_sel = Selector(name=failed_condition.name + "_Selector")
    tree_sel.add_child(copy.copy(failed_condition))
    for action in action_templates:
        tree_seq = Sequence(name=action.action_name + "_Sequence")
        for key in action.pre:
            if last_action and key in last_action.action.post:
                if not last_action.action.post[key] == action.pre[key]:
                    conflict = True
            tree_seq.add_child(Condition(key, action.pre[key]))
        tree_seq.add_child(Action(action))
        tree_sel.add_child(tree_seq)
    # print "OUT OF TREE", tree_sel.parent
    tree_sel.parent
    # pdb.set_trace()
    tree.replace_subtree(failed_condition.id, tree_sel)
    tree_sel.parent = failed_condition.parent
    # print "IN TREE", tree_sel.parent
    tree_sel.parent
    return tree, tree_sel, conflict


def get_condition_to_expand(tree, tree_visitor, expanded_nodes):
    '''
    The GetConditionToExpand function of the PA-BT algorithm. 'Algorithm 3'.

    Args:
        tree: the current BehaviourTree
        tree_visitor: A VisitorBase that records failed nodes
        expanded_nodes: the list of previously expanded conditions

    Returns:
        the next condition to expand, or None if there aren't any
    '''
    # for condition in tree_visitor.failing_nodes:
    #     print "FAILED NODE", condition.name
    for condition in tree_visitor.failing_nodes[::-1]:
        if condition.name not in expanded_nodes:
            expanded_nodes.append(condition.name)
            return condition
    return None


def conflict(last_action):
    '''
    conflict - executing an action creates a mismatch between effects and
    preconditions the progress of the plan

    analyze conditions of new action added with effects of actions that subtree
    executes before executing the new action
    '''
    # TODO(kmuhlrad): replace this with something useful
    return False


def increase_priority(tree, subtree):
    '''
    Move a conflicting subtree leftward. If the subtree is already the
    left_most tree on a certain level, move it up a level and to left of its original
    parent.
    '''
    
    sub = subtree
    index_mod = 0
    while sub.parent:
        index = sub.parent.children.index(sub)
        if index or index_mod:
            tree.prune_subtree(subtree.id)
            tree.insert_subtree(subtree, sub.parent.id, index + index_mod - 1)
            break
        else:
            sub = sub.parent
            index_mod = -1


def drop_soup(blackboard):
    blackboard.set("robot_holding", None)
    blackboard.set("soup_on", "table")

def left_door_closed(blackboard):
    blackboard.set("left_door_open", False)

def random_event(blackboard):
    prob = np.random.rand()
    if prob >= 0.8:
        print "OOPS, THE DOOR CLOSED"
        left_door_closed(blackboard)
    elif prob >= 0.6:
        print "OOPS, I DROPPED THE SOUP"
        drop_soup(blackboard)

def unstack_block(blackboard):
    blackboard.set("a_on_table", True)
    blackboard.set("a_on", None)
    blackboard.set("a_clear", True)
    blackboard.set("b_clear", True)
    blackboard.set("b_on_table", True)
    blackboard.set("b_on", None)
    blackboard.set("holding", False)
    blackboard.set("arm-empty", True)
    blackboard.set("c_on_table", True)
    blackboard.set("c_clear", True)

def pa_bt(ppa_set, goal_conditions):
    '''
    The main loop of the PA-BT Algorithm. 'Algorithm 1' in the paper.

    Args:
        goal_conditions: a list of conditions (Behaviours) that must
            be true in the goal state.

    '''
    import time
    start = time.time()
    root_seq = Sequence(name="Root")

    for condition in goal_conditions:
        root_seq.add_child(condition)

    tree = BehaviourTree(root_seq)
    condition_bfs = ConditionBFSVisitor()
    last_action_visitor = LastActionVisitor()

    tree.visitors.append(condition_bfs)
    tree.visitors.append(last_action_visitor)

    # for debugging
    snapshot_visitor = py_trees.visitors.SnapshotVisitor()
    tree.visitors.append(snapshot_visitor)

    blackboard = py_trees.blackboard.Blackboard()

    expanded_nodes = []
    iteration = 1
    door_counter = 0
    while True:
        # py_trees.display.render_dot_tree(tree.root, name="iteration_{}".format(iteration))
        # print "ITERATION", iteration
        status = Status.INVALID
        while not status == Status.FAILURE and not status == Status.SUCCESS:
            tree.tick()
            status = tree.root.status
            ascii_tree = py_trees.display.ascii_tree(
            tree.root,
            snapshot_information=snapshot_visitor)
            # print ascii_tree
            # time.sleep(1.0)
            if door_counter < 3:
                if blackboard.get("b_on") == "c":
                    unstack_block(blackboard)
                    door_counter += 1
            # random_event(blackboard)
            # print blackboard
        if status == Status.SUCCESS:
            break
        failed_condition = get_condition_to_expand(tree, condition_bfs, expanded_nodes)
        if not failed_condition:
            continue
        tree, tree_new_subtree, conflict = expand_tree(
            tree, ppa_set, failed_condition, last_action_visitor.last_action)
        # TODO(kmuhlrad): for now ignore conflicts, since they are irrelevant
        # in this examples
        if conflict:
            # print "CONFLICT"
            import pdb
            # pdb.set_trace()
            ascii_tree = py_trees.display.ascii_tree(tree.root)
            # print "LAST ACTION", last_action_visitor.last_action.name
            # print "SUBTREE ROOT", tree_new_subtree.name
            failed_condition_index = ascii_tree.index(last_action_visitor.last_action.name)
            subtree_index = ascii_tree.index(tree_new_subtree.name)

            # print "FAILED INDEX", failed_condition_index
            # print "SUBTREE INDEX", subtree_index

            # import pdb
            # pdb.set_trace()

            while failed_condition_index < subtree_index:
                
                # print "CONFLICT"
                # print
                # print "BEFORE TREE"
                # print py_trees.display.ascii_tree(tree.root)
                # print
                # print
                # print
                # print "SUBTREE"
                # print py_trees.display.ascii_tree(tree_new_subtree)
                # print
                # print
                # print
                increase_priority(tree, tree_new_subtree)
                # print
                # print
                # print
                # print "NEW TREE"
                # print py_trees.display.ascii_tree(tree.root)
                # print
                # print
                # print
                ascii_tree = py_trees.display.ascii_tree(tree.root)
                failed_condition_index = ascii_tree.index(last_action_visitor.last_action.name)
                subtree_index = ascii_tree.index(tree_new_subtree.name)
                # pdb.set_trace()
        iteration += 1
        # time.sleep(1.0)
    end = time.time()
    # print "TOTAL ITERATIONS:", iteration
    # print "TOTAL TIME:", end - start
    return end - start

def make_soup_can_ppa_set():
    '''
    Make the precondition-postcondition-action set for the
    soup can placing demo. The preconditions must be ordered because
    we want to make sure certain conditions are true before checking
    anything else.
    '''

    # Pick
    pick_soup_pre = OrderedDict([
        ("robot_holding", None),
        ("robot_moving", False)
    ])
    pick_soup_post = {
        "robot_holding": "soup",
        "robot_moving": False,
        "soup_on": None
    }
    pick = PPA("PickSoup", pick_soup_pre, pick_soup_post)

    # Place
    place_soup_on_shelf_lower_pre = OrderedDict([
        ("left_door_open", True),
        ("robot_holding", "soup"),
        ("robot_moving", False)
    ])
    place_soup_on_shelf_lower_post = {
        "robot_holding": None,
        "soup_on": "shelf_lower",
        "robot_moving": False
    }
    place = PPA("PlaceSoupOnShelfLower", 
                place_soup_on_shelf_lower_pre, 
                place_soup_on_shelf_lower_post)

    # Open Door
    open_left_door_pre = OrderedDict([
        ("left_door_open", False),
        ("robot_moving", False)
    ])
    open_left_door_post = {
        "left_door_open": True,
        "robot_moving": False
    }
    open_door = PPA("OpenLeftDoor", 
                    open_left_door_pre, 
                    open_left_door_post)

    return set([pick, place, open_door])


def init_blackboard():
    blackboard = py_trees.blackboard.Blackboard()

    blackboard.set("robot_moving", False)
    blackboard.set("robot_holding", None)
    blackboard.set("soup_on", "table")
    blackboard.set("left_door_open", False)
    blackboard.set("right_door_open", True)


def test_increase_priority():
    root = py_trees.composites.Sequence(name="Root")

    count1 = py_trees.behaviours.Count(name="1")
    count2 = py_trees.behaviours.Count(name="2")
    count3 = py_trees.behaviours.Count(name="3")
    count4 = py_trees.behaviours.Count(name="4")

    seq1 = py_trees.composites.Sequence(name="Sequence")
    sel1 = py_trees.composites.Selector(name="Selector")
    seq2 = py_trees.composites.Sequence(name="Nested")
    
    root.add_children([seq1, sel1])
    seq1.add_children([count1, count2])
    sel1.add_children([seq2, count3])
    seq2.add_child(count4)

    tree = BehaviourTree(root)

    print py_trees.display.ascii_tree(root)

    print get_node_level(sel1)

    print py_trees.display.ascii_tree(root)


if __name__ == "__main__":
    import sys
    args = sys.argv
    if len(args) != 2:
        total_runs = 1
    else:
        total_runs = int(args[1])

    py_trees.logging.level = py_trees.logging.Level.INFO

    ppa_set = make_soup_can_ppa_set()

    avg_time = 0
    for i in range(total_runs):
        init_blackboard()
        goal_conditions = [Condition("soup_on", "shelf_lower")]
        avg_time += pa_bt(ppa_set, goal_conditions)

    avg_time /= total_runs

    print "AVERAGE TIME", avg_time

    # test_increase_priority()