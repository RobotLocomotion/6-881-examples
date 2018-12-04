# Perception

This module contains classes and examples for working with data from the three Realsense D415 cameras.

## Files
- `point_cloud_to_pose_system.py`: `PointCloudToPoseSystem` is a Drake `LeafSystem` that takes in point clouds and RGB images and returns the pose of a specific object. The user supplies a camera configuration file, a function to segment an object from the aligned point cloud, and a function to compute the pose of the object from the segmented point cloud.
- `run_perception_system.py` contains an example of using a `PointCloudToPoseSystem` to compute the pose of the foam brick.
- `test_perception.py` is a script to test basic functionality of `point_cloud_to_pose_system.py` for continuous integration.
- `PointCloudVisualization.ipynb` is a Jupyter notebook with examples of visualizing point cloud data retrieved with a `PointCloudToPoseSystem`.

### Model Files
Files under the `models/` directory are serialized numpy arrays (`.npy` files) used as models or ground truth for point cloud alignment.

### Camera Configuration Files
[YAML]((https://learn.getgrav.org/advanced/yaml)) camera configuration files live in the `config/` directory. The files describe information about each of the three cameras both in simulation and on the physical workstations. The top level contains three cameras, `left_camera`, `middle_camera`, and `right_camera`. Each of these contains the serial number of the camera, the pose of the camera in world frame, the transform between the camera's base frame and its optical frame, and the camera instrinsics.

- `sim.yml` contains the configuration of the three cameras defined in [`ManipulationStation`](https://drake.mit.edu/doxygen_cxx/classdrake_1_1examples_1_1manipulation__station_1_1_manipulation_station.html).
- `station_1.yml` contains the configuration of the three cameras on the physical station closest to the chalkboard.
- [TODO] `station_2.yml` contains the configuration of the three cameras on the physical station closest to the chalkboard.

## Using This Module
To run the example of getting the pose of the foam brick, execute `run_perception_system.py` with a camera configuration file. For example:

```sh
$ python run_perception_system.py --config_file=config/sim.yml
```

If the given configuration file is `sim.yml`, the simulated [`ManipulationStation`](https://drake.mit.edu/doxygen_cxx/classdrake_1_1examples_1_1manipulation__station_1_1_manipulation_station.html) will be created. If configuration file is one of the station configurations, a [`ManipulationStationHardwareInterface`](https://drake.mit.edu/doxygen_cxx/classdrake_1_1examples_1_1manipulation__station_1_1_manipulation_station_hardware_interface.html) will be created, and the script will wait to receive LCM messages from the robot, gripper, and cameras. More information about running the system on the hardware will be explained in the lab.

### Running in Docker
If these scripts are running in a docker container, in order to enable graphics, run the following lines in a docker bash session:

```sh
$ Xvfb :100 -ac -screen 0 800x600x24 &
$ DISPLAY=:100 python run_perception_system.py --config_file=config/sim.yml
```

### Visualization
There is currently no way to automatically visualize these point clouds in meshcat. The best way to visualize the point clouds is to save the numpy arrays created in `point_cloud_to_pose_system.py` and load them in a Jupyter notebook, similar to psets 2 and 3. To do that, initialize a `PointCloudToPoseSystem` with the parameter `viz=True`. The point clouds from each of the cameras, the aligned scene point cloud, and the segmented point cloud will all be saved as `.npy` files under a directory called `saved_point_clouds`. Examples of viewing them are in `PointCloudVisualization.ipynb`. To run that file in a docker container (that's already in the `perception/` directory):

```sh
$ Xvfb :100 -ac -screen 0 800x600x24 &
$ mkdir saved_point_clouds
$ DISPLAY=:100 jupyter notebook --ip 0.0.0.0 --port 8080 --allow-root --no-browser
```

Follow the instructions in the terminal to open up the notebook, which can be run just like in the psets.