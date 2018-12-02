import unittest
from perception.test_perception import TestPointCloudToPoseSystem
from plan_runner.test_plan_runner import TestOpenDoor

if __name__ == "__main__":
    # Run tests
    test_cases = [TestOpenDoor, TestPointCloudToPoseSystem]
    suite = unittest.TestSuite()
    for test_case in test_cases:
        tests = unittest.defaultTestLoader.loadTestsFromTestCase(test_case)
        suite.addTests(tests)

    result = unittest.TextTestRunner(verbosity=2).run(suite)
    exit(not result.wasSuccessful())
