from behaviors import *
import py_trees
from py_trees.trees import BehaviourTree
from py_trees.composites import Sequence, Selector
from py_trees.meta import inverter
from py_trees.visitors import VisitorBase

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
        super(Condition, self).__init__(name="{} is {}".format(condition_key, condition_value))
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
        super(Action, self).__init__(name=action.action_name)
        self.blackboard = Blackboard()
        self.action = action

    def initialise(self):
        self.counter = 0

    def update(self):
        if self.counter < 1:
            for key in self.action.pre:
                self.blackboard.set(key, not self.action.pre[key])
            self.counter += 1
            return Status.RUNNING

        for key in self.action.post:
            self.blackboard.set(key, self.action.post[key])

        return Status.SUCCESS


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
    print "FAILED CONDITION", failed_condition
    for action in ppa_set:
        if failed_condition.condition_key in action.post:
            if failed_condition.condition_value == action.post[failed_condition.condition_key]:
                action_templates.add(action)
    return action_templates


def expand_tree(tree, ppa_set, failed_condition):
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
    action_templates = get_all_action_templates_for(ppa_set, failed_condition)
    print "ACTION TEMPLATES", action_templates
    tree_sel = Selector(name=failed_condition.name + " Selector")
    tree_sel.add_child(failed_condition)
    for action in action_templates:
        tree_seq = Sequence(name=action.action_name)
        for key in action.pre:
            tree_seq.add_child(Condition(key, action.pre[key]))
        tree_seq.add_child(Action(action))
        tree_sel.add_child(tree_seq)
    print tree_sel
    # ascii_tree = py_trees.display.ascii_tree(
            # tree.root)
    print tree.root.children
    print failed_condition
    # print ascii_tree
    success = tree.replace_subtree(failed_condition.id, tree_sel)
    print "REPLACED", success
    print tree.root.children
    # ascii_tree = py_trees.display.ascii_tree(
    #         tree.root)
    # print ascii_tree
    return tree, tree_sel


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
    for condition in tree_visitor.failing_nodes:
        if condition not in expanded_nodes:
            expanded_nodes.append(condition)
            return condition
    return None


def conflict(tree):
    '''
    conflict - executing an action creates a missmatch between effects and
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
    while sub.parent:
        index = sub.parent.children.index(sub)
        if index:
            tree.insert_subtree(subtree, sub.parent.id, index - 1)
            break
        else:
            sub = sub.parent


def pa_bt(ppa_set, goal_conditions):
    '''
    The main loop of the PA-BT Algorithm. 'Algorithm 1' in the paper.

    Args:
        goal_conditions: a list of conditions (Behaviours) that must
            be true in the goal state.

    '''
    root_seq = Sequence(name="Root")
    tree = BehaviourTree(root_seq)
    condition_bfs = ConditionBFSVisitor()
    tree.visitors.append(condition_bfs)

    # for debugging
    snapshot_visitor = py_trees.visitors.SnapshotVisitor()
    tree.visitors.append(snapshot_visitor)

    for condition in goal_conditions:
        root_seq.add_child(condition)

    expaned_nodes = []
    while True:
        status = Status.INVALID
        while not status == Status.FAILURE:
            tree.tick()
            status = tree.root.status
            ascii_tree = py_trees.display.ascii_tree(
            tree.root,
            snapshot_information=snapshot_visitor)
            print ascii_tree
        failed_condition = get_condition_to_expand(tree, condition_bfs, expaned_nodes)
        print "MAIN FAILED CONDITION", failed_condition
        tree, tree_new_subtree = expand_tree(tree, ppa_set, failed_condition)
        while conflict(tree):
            tree = increase_priority(tree, tree_new_subtree)


def make_ppa_set():
    '''
    Make the precondition-postcondition-action set for the
    soup can placing demo.
    '''

    # Pick
    pick_soup_pre = {
        "robot_moving": False,
        "robot_holding": None
    }
    pick_soup_post = {
        "robot_holding": "soup",
        "robot_moving": False
    }
    pick = PPA("PickSoup", pick_soup_pre, pick_soup_post)

    # Place
    place_soup_on_shelf_lower_pre = {
        "robot_moving": False,
        "robot_holding": "soup"
    }
    place_soup_on_shelf_lower_post = {
        "robot_moving": False,
        "robot_holding": None,
        "soup_on": "shelf_lower"
    }
    place = PPA("PlaceSoupOnShelfLower", 
                place_soup_on_shelf_lower_pre, 
                place_soup_on_shelf_lower_post)

    # Open Door
    open_left_door_pre = {
        "robot_moving": False,
        "left_door_open": False
    }
    open_left_door_post = {
        "robot_moving": False,
        "left_door_open": True
    }
    open_door = PPA("OpenLeftDoor", 
                    open_left_door_pre, 
                    open_left_door_post)

    return set([pick, place, open_door])


# TODO(kmuhlrad): delete, just for reference
def make_root():
    # conditions
    holding_soup = Holding("soup")
    soup_on_shelf = On("soup", "shelf_lower")
    left_door_open = DoorOpen("left_door")
    moving_inverter = inverter(RobotMoving)("MovingInverter")
    gripper_empty = inverter(Holding)("soup", "GripperEmpty")

    # actions
    pick_soup = Pick("soup")
    place_soup = Place("soup", "shelf_lower")
    open_left_door = OpenDoor("left_door")

    root = Selector(name="Root")

    soup_on_shelf_seq = Sequence(name="SoupOnShelfSeq")

    open_door_sel = Selector(name="OpenDoorSel")
    open_door_seq = Sequence(name="OpenDoorSeq")

    pick_soup_sel = Selector(name="PickSoupSel")
    pick_soup_seq = Sequence(name="PickSoupSeq")

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
    py_trees.logging.level = py_trees.logging.Level.DEBUG

    ppa_set = make_ppa_set()

    init_blackboard()

    goal_conditions = [Condition("soup_on", "shelf_lower")]

    pa_bt(ppa_set, goal_conditions)