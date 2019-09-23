import numpy as np
import yaml

from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.systems.rendering import PoseBundle
from pydrake.systems.sensors import CameraInfo

def ReadPosesFromFile(filename):
    """
    Reads in 4x4 poses from a text file and wraps them into a PoseBundle.

    @param filename str. The path of the .txt file to read.
    @return a PoseBundle with all of the poses.
    """
    pose_dict = {}
    row_num = 0
    object_name = ""
    cur_matrix = np.eye(4)
    with open(filename, "r") as f:
        for line in f:
            line = line.rstrip()
            if not line.lstrip(" ").startswith("["):
                object_name = line
            else:
                row = np.matrix(line)
                cur_matrix[row_num, :] = row
                row_num += 1
                if row_num == 4:
                    pose_dict[object_name] = RigidTransform(cur_matrix)
                    cur_matrix = np.eye(4)
                row_num %= 4

    pose_bundle = PoseBundle(num_poses=len(pose_dict))
    i = 0
    for object_name in pose_dict:
        pose_bundle.set_name(i, object_name)
        pose_bundle.set_pose(i, pose_dict[object_name])
        i += 1

    return pose_bundle


def _xyz_rpy(xyz, rpy):
    return RigidTransform(RollPitchYaw(rpy), xyz)


def _ParseCameraConfig(camera_config):
    # extract serial number
    serial_no = camera_config["serial_no"]

    # construct the transformation matrix
    world_transform = camera_config["world_transform"]
    X_WCamera = _xyz_rpy([world_transform["x"],
                          world_transform["y"],
                          world_transform["z"]],
                         [world_transform["roll"],
                          world_transform["pitch"],
                          world_transform["yaw"]])

    # construct the transformation matrix
    internal_transform = camera_config["internal_transform"]
    X_CameraDepth = _xyz_rpy([internal_transform["x"],
                              internal_transform["y"],
                              internal_transform["z"]],
                             [internal_transform["roll"],
                              internal_transform["pitch"],
                              internal_transform["yaw"]])

    # construct the camera info
    camera_info_data = camera_config["camera_info"]
    if "fov_y" in camera_info_data:
        camera_info = CameraInfo(camera_info_data["width"],
                                 camera_info_data["height"],
                                 camera_info_data["fov_y"])
    else:
        camera_info = CameraInfo(
            camera_info_data["width"], camera_info_data["height"],
            camera_info_data["focal_x"], camera_info_data["focal_y"],
            camera_info_data["center_x"], camera_info_data["center_y"])

    return serial_no, X_WCamera, X_CameraDepth, camera_info


def LoadCameraConfigFile(config_file):
    """
    Loads camera information from a .yml file into a dictionary. The
    information is first keyed on the camera's serial number, and then by
    the type of information. For example, if a camera's serial number is 123,
    the entry containing its information looks is under camera_configs[123].
    The subfields are camera_pose_world, camera_pose_internal, and camera_info.
    camera_pose_world and camera_pose_internal are both RigidTransforms.
    camera_pose_world is the transform between the optical frame and the world.
    camera_pose_internal is the transform between the optical frame and depth
    frame.
    camera_info is the CameraInfo of the camera.

    @param config_file str. The path to the .yml file with the camera info.
    @return a dict containing the camera information
    """
    camera_configs = {}
    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            for camera in config:
                serial_no, X_WCamera, X_CameraDepth, camera_info = \
                    _ParseCameraConfig(config[camera])
                id = str(serial_no)
                camera_configs[id] = {}
                camera_configs[id]["camera_pose_world"] = X_WCamera
                camera_configs[id]["camera_pose_internal"] = X_CameraDepth
                camera_configs[id]["camera_info"] = camera_info
        except yaml.YAMLError as exc:
            print("could not parse config file")
            print(exc)

    return camera_configs