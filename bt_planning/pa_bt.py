from behaviors import *
import py_trees


class PPA:
    def __init__(self, action_name, pre, post):
        self.action_name = action_name
        self.pre = pre
        self.post = post


def make_ppa_set():
    '''
    Make the precondition-postcondition-action set for the
    soup can placing demo.
    '''

    # Pick
    pick_soup_pre = {
        "robot_moving": False
        "robot_holding": None
    }
    pick_soup_post = {
        "robot_holding": "soup"
        "robot_moving": False
    }
    pick = PPA("PickSoup", pick_soup_pre, pick_soup_post)

    # Place
    place_soup_on_shelf_lower_pre = {
        "robot_moving": False
        "robot_holding": "soup"
    }
    place_soup_on_shelf_lower_post = {
        "robot_moving": False
        "robot_holding": None
        "soup_on": "shelf_lower"
    }
    place = PPA("PlaceSoupOnShelfLower", 
                place_soup_on_shelf_lower_pre, 
                place_soup_on_shelf_lower_post)

    # Open Door
    open_left_door_pre = {
        "robot_moving": False
        "left_door_open": False
    }
    open_left_door_post = {
        "robot_moving": False
        "left_door_open": True
    }
    open_door = PPA("OpenLeftDoor", 
                    open_left_door_pre, 
                    open_left_door_post)

    return set(pick, place, open_door)


def pa_bt():
    pass
    '''
    The main loop of the PA-BT Algorithm. 'Algorithm 1' in the paper.

    pseudocode:

    tree = None
    for condition in goal_conditions:
        tree = sequence_node(tree, condition)

    while True:
        tree = refine_actions(tree)
        status = Status.INVALID
        while not status == Status.FAILURE:
            status = tree.tick_once()
        failed_condition = get_condition_to_expand(tree)
        tree, tree_new_subtree = expand_tree(tree, failed_condition)
        while conflict(tree):
            tree = increase_priority(tree_new_subtree)

    '''


def expand_tree(tree, failed_condition):
    '''
    The ExpandTree function of the PA-BT algorithm. 'Algorithm 2'.

    pseudocode:

    action_templates = GetAllActionTemplatesFor(failed_condition)
    tree_fall = failed_condition
    for action in action_templates:
        tree_seq = None
        for condition in action.pre:
            tree_seq = sequence_node(tree_seq, condition)
        tree_seq = sequence_node(tree_seq, action)
        tree_fall = selector_node(tree_fall, tree_seq)
    tree = substitute(tree, failed_condition, tree_fall)
    return tree, tree_fall
    '''
    pass


def get_condition_to_expand(tree):
    '''
    The GetConditionToExpand function of the PA-BT algorithm. 'Algorithm 3'.

    pseudocode:

    for condition in GetConditionsBFS():
        if (condition.status == Status.FAILURE 
            and condition not in expanded_nodes):

            expanded_nodes.append(condition)
            return condition

    return None
    '''
    pass

if __name__ == "__main__":
    ppa_set = make_ppa_set()
    pa_bt()