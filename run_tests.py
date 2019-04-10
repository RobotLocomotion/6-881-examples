import unittest
# from pddl_planning.test_pddl_planning import TestPDDLPlanning

from perception_tools.test_perception_tools import TestPointCloudSynthesis
# from plan_runner.test_plan_runner import TestOpenDoor

if __name__ == "__main__":
    # Run tests
    #test_cases = [TestPointCloudToPoseSystem, TestPDDLPlanning, TestOpenDoor]
    test_cases = [TestPointCloudSynthesis]
    suite = unittest.TestSuite()
    for test_case in test_cases:
        tests = unittest.defaultTestLoader.loadTestsFromTestCase(test_case)
        suite.addTests(tests)

    result = unittest.TextTestRunner(verbosity=2).run(suite)
    exit(not result.wasSuccessful())
