import py_trees
from pa_bt import Condition, pa_bt, PPA

from collections import OrderedDict

def make_blocks_world_ppa_set(block_names):
    ppa_set = set()

    # Blocks
    # block_names = ["a", "b", "c"]

    # Pick Up
    for ob in block_names:
        pre = OrderedDict([
            ("{}_clear".format(ob), True),
            ("{}_on_table".format(ob), True),
            ("arm_empty", True)
        ])
        post = {
            "{}_on_table".format(ob): False,
            "{}_clear".format(ob): False,
            "arm_empty": False,
            "holding": ob
        }
        ppa_set.add(PPA("PickUp_{}".format(ob), pre, post))

    # PutDown
    for ob in block_names:
        pre = OrderedDict([
            ("holding", ob)
        ])
        post = {
            "{}_on_table".format(ob): True,
            "{}_clear".format(ob): True,
            "arm_empty": True,
            "holding": None 
        }
        ppa_set.add(PPA("PutDown_{}".format(ob), pre, post))

    # Stack
    for i in range(len(block_names)):
        for j in range(len(block_names)):
            if not i == j:
                ob = block_names[i]
                underob = block_names[j]

                pre = OrderedDict([
                    ("{}_clear".format(underob), True),
                    ("holding", ob)
                ])
                post = {
                    "{}_clear".format(underob): False,
                    "{}_clear".format(ob): True,
                    "arm_empty": True,
                    "{}_on".format(ob): underob 
                }
                ppa_set.add(PPA("Stack_{}_On_{}".format(ob, underob), pre, post))

    # Unstack
    for i in range(len(block_names)):
        for j in range(len(block_names)):
            if not i == j:
                ob = block_names[i]
                underob = block_names[j]

                pre = OrderedDict([
                    ("{}_clear".format(ob), True),
                    ("{}_on".format(ob), underob),
                    ("arm_empty", True)
                ])
                post = {
                    "{}_clear".format(ob): False,
                    "{}_clear".format(underob): True,
                    "arm_empty": False,
                    "{}_on".format(ob): None,
                    "holding": ob
                }
                ppa_set.add(PPA("Unstack_{}_From_{}".format(ob, underob), pre, post))

    return ppa_set


def init_blackboard_simple():
    blackboard = py_trees.blackboard.Blackboard()

    blackboard.set("a_clear", True)
    blackboard.set("arm_empty", True)
    blackboard.set("a_on", "b")
    blackboard.set("b_on_table", True)


def init_blackboard_tall(tall_blocks):
    blackboard = py_trees.blackboard.Blackboard()

    blackboard.set("arm_empty", True)

    for block in tall_blocks:
        blackboard.set("{}_clear".format(block), True)
        blackboard.set("{}_on_table".format(block), True)


def init_blackboard_sussman():
    blackboard = py_trees.blackboard.Blackboard()

    blackboard.set("b_clear", True)
    blackboard.set("c_clear", True)
    blackboard.set("arm_empty", True)
    blackboard.set("c_on", "a")
    blackboard.set("a_on_table", True)
    blackboard.set("b_on_table", True)

def init_blackboard_large_a():
    blackboard = py_trees.blackboard.Blackboard()

    blackboard.set("arm_empty", True)
    
    blackboard.set("3_clear", True)
    blackboard.set("3_on", "2")
    blackboard.set("2_on", "1")
    blackboard.set("1_on_table", True)

    blackboard.set("5_clear", True)
    blackboard.set("5_on", "4")
    blackboard.set("4_on_table", True)

    blackboard.set("9_clear", True)
    blackboard.set("9_on", "8")
    blackboard.set("8_on", "7")
    blackboard.set("7_on", "6")
    blackboard.set("6_on_table", True)


def init_blackboard_12_step():
    blackboard = py_trees.blackboard.Blackboard()

    blackboard.set("arm_empty", True)

    blackboard.set("a_clear", True)
    blackboard.set("a_on_table", True)

    blackboard.set("b_clear", True)
    blackboard.set("b_on_table", True)

    blackboard.set("c_clear", True)
    blackboard.set("c_on", "d")
    blackboard.set("d_on", "e")
    blackboard.set("e_on", "f")
    blackboard.set("f_on", "g")
    blackboard.set("g_on_table", True)

def clear_black_board_vars(blocks):
    blackboard = py_trees.blackboard.Blackboard()

    blackboard.set("arm_empty", None)
    blackboard.set("holding", None)

    for block in blocks:
        blackboard.set("{}_on".format(block), None)
        blackboard.set("{}_clear".format(block), None)
        blackboard.set("{}_on_table".format(block), None)

if __name__ == "__main__":
    import sys
    args = sys.argv
    if len(args) != 2:
        total_runs = 1
    else:
        total_runs = int(args[1])

    py_trees.logging.level = py_trees.logging.Level.INFO

    simple_blocks = ["a", "b", "c"]
    more_blocks = ["a", "b", "c", "d", "e", "f", "g"]
    large_blocks = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    ppa_set = make_blocks_world_ppa_set(simple_blocks)

    # Simple
    goal_conditions = [Condition("a_on_table", True), Condition("b_clear", True)]

    # Tall

    # 2 blocks
    # tall_blocks = ["a", "b"]
    # goal_conditions = [Condition("a_on", "b")]

    # 3 blocks

    # 4 blocks

    # 5 blocks

    # 10 blocks

    # ppa_set = make_blocks_world_ppa_set(tall_blocks)

    avg_time = 0
    tall_num = 10
    all_tall_blocks = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    all_goal_conditions = [
        Condition("a_clear", True),
        Condition("a_on", "b"),
        Condition("b_on", "c"),
        Condition("c_on", "d"),
        Condition("d_on", "e"),
        Condition("e_on", "f"),
        Condition("f_on", "g"),
        Condition("g_on", "h"),
        Condition("h_on", "i"),
        Condition("i_on", "j"),
    ]
    ppa_set = make_blocks_world_ppa_set(all_tall_blocks[:tall_num])


    for i in range(total_runs):
        # Simple
        # clear_black_board_vars(simple_blocks)
        # init_blackboard_simple()
        # goal_conditions = [Condition("a_on_table", True), Condition("b_clear", True)]
        
        # Tall
        clear_black_board_vars(all_tall_blocks[:tall_num])
        init_blackboard_tall(all_tall_blocks[:tall_num])
        goal_conditions = list(all_goal_conditions[:tall_num])[::-1]
        
        avg_time += pa_bt(ppa_set, goal_conditions)

    avg_time /= total_runs

    print "AVERAGE TIME", avg_time

    # Sussman
    # goal_conditions = [Condition("b_on", "a"), Condition("c_on", "b")]

    # Large A
    # goal_conditions = [
    #     Condition("1_on", "5"), 
    #     Condition("5_on_table", True),
    #     Condition("8_on", "9"),
    #     Condition("9_on", "4"),
    #     Condition("4_on_table", True),
    #     Condition("2_on", "3"),
    #     Condition("3_on", "7"),
    #     Condition("7_on", "6"),
    #     Condition("6_on_table", True),
    #     Condition("1_clear", True),
    #     Condition("8_clear", True),
    #     Condition("2_clear", True)
    # ]

    # 12-step
    # goal_conditions = [
    #     Condition("a_on_table", True),
    #     Condition("f_on", "a"),
    #     Condition("c_on", "d"),
    #     Condition("b_on", "c"), 
    # ]
