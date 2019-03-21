import yaml

from pydrake.systems.sensors import CameraInfo
import meshcat.transformations as tf

def LoadConfigFile(config_file):
    with open(config_file, 'r') as stream:
        try:
            config = yaml.load(stream)
            camera_configs = {}
            for camera in config:
                serial_no, X_WCamera, X_CameraDepth, camera_info = \
                    _ParseCameraConfig(config[camera])
                id = str(serial_no)
                camera_configs[id] = {}
                camera_configs[id]["camera_pose_world"] = X_WCamera
                camera_configs[id]["camera_pose_internal"] = \
                    X_CameraDepth
                camera_configs[id]["camera_info"] = camera_info
        except yaml.YAMLError as exc:
            print "could not parse config file"
            print exc

    return camera_configs


# TODO(kmuhlrad): drakify
def _ParseCameraConfig(camera_config):
    # extract serial number
    serial_no = camera_config["serial_no"]

    # construct the transformation matrix
    world_transform = camera_config["world_transform"]
    X_WCamera = tf.euler_matrix(world_transform["roll"],
                                world_transform["pitch"],
                                world_transform["yaw"])
    X_WCamera[:3, 3] = \
        [world_transform["x"], world_transform["y"], world_transform["z"]]

    # construct the transformation matrix
    internal_transform = camera_config["internal_transform"]
    X_CameraDepth = tf.euler_matrix(internal_transform["roll"],
                                    internal_transform["pitch"],
                                    internal_transform["yaw"])
    X_CameraDepth[:3, 3] = ([internal_transform["x"],
                             internal_transform["y"],
                             internal_transform["z"]])

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