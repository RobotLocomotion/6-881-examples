# Perception

This module contains classes and examples for working with data from the three
Realsense D415 cameras.

## Files
- `dope_system.py`: `DopeSystem` is a `LeafSystem` that runs NVidia's Deep 
Object Pose Estimation. See below for more details.

- `weights/` is a directory to store DOPE weights. See below for more details.

- `inference/` is a directory containing code provided by NVidia for DOPE. For
more information and the license, see below.

- `pose_refinement.py`: `PoseRefinment` is a Drake `LeafSystem` that, given a
point cloud and initial guesses of poses of objects in the point cloud, will
compute refined pose estimates. The user can supply custom object
segmentation and pose alignment functions.

- `run_dope_demo.py`: A script that will create a `ManipulationStation` with
many YCB objects, run DOPE to estimate all of their poses, and uses a
`PoseRefinement` to get better pose estimates of the specified objects.

- `test_perception.py` is a blank script where perception tests can be added.

### Model Files
Files under the `models/` directory are serialized numpy arrays (`.npy` files) 
used as models or ground truth for point cloud alignment.

These files were created using the `convert_obj_to_npy.py` script in the
`perception_tools/` directory on the `.obj` files of each of the YCB objects 
from the original dataset.

### Configuration Files
[YAML](https://learn.getgrav.org/advanced/yaml) configuration files live in the 
`config/` directory.

#### Camera Configuration Files
These files describe information about each of the three cameras both in 
simulation and on the physical workstations. The top level contains three 
cameras, `left_camera`, `middle_camera`, and `right_camera`. Each of these 
contains the serial number of the camera, the pose of the camera in world 
frame, the transform between the camera's base frame and its optical frame, 
and the camera intrinsics.

- `sim.yml` contains the configuration of the three cameras defined in 
[`ManipulationStation`](
https://drake.mit.edu/doxygen_cxx/classdrake_1_1examples_1_1manipulation__station_1_1_manipulation_station.html).
- `station_1.yml` contains the configuration of the three cameras on the 
physical station closest to the chalkboard.
- `station_2.yml` contains the configuration of the three cameras on the 
physical station closest to the entryway.

#### Dope Configuration Files
These files contain information used by DOPE to recognize YCB objects and draw
bounding boxes around them in a new RGB image.

- `dope_config.yml` contains information to use DOPE with one of the simulated
`ManipulationStation` cameras.

## Using This Module

### DOPE
`DopeSystem` is a Drake `System` that can run 
[DOPE](https://github.com/NVlabs/Deep_Object_Pose). The arguments are a 
configuration file and a path to a directory of weight files. 
`config/dope_config.yml` is an example config file set up for a 
Drake `RgbdCamera`. The `DopeSystem` has one input port that takes in an RGB 
image. It also has two output ports: one that outputs a `PoseBundle` of any 
recognized objects, and one that produces a new RGB image with bounding boxes 
drawn over every detected object. See the 
[DOPE documentation](https://github.com/NVlabs/Deep_Object_Pose) for more 
details about DOPE.

Note that currently DOPE does not work in the docker image, since it requires 
GPU access to run. It also does not have CI tests. You must download the 
weights to the `weights/` directory yourself, following the DOPE setup 
instructions.

The code was modified from NVidia's provided code, and is licensed under a 
[Creative-Commons Attribution-NonCommercial-ShareAlike 4.0 International](
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license. Any 
modifications were indicated in the relevant files.

#### Running DOPE
There is an example of using `DopeSystem` with a `ManipulationStation` at the 
end of `dope_system.py`. Just run it via
```sh
$ python dope_system.py
```
Poses of recognized objects will be printed in the terminal window, and a 
separate window should pop up containing the annotated RGB image.

### run\_dope\_demo.py

This script was used to create the bounding box estimates in [this video](
https://youtu.be/zUS33rvbRsc). It goes through the whole process of creating a
`ManipulationStation` with YCB objects, merging the images of all of the cameras
into a single point cloud, running DOPE on one of the RGB images, and getting
refined pose estimates of some of the objects DOPE recognized. Note that this 
only gives pose estimates at a single timestep. In order to produce the updated
bounding boxes in the video, this script was run again with the poses of all of
the objects manually changed if needed, and possibly different object
segmentation functions as well. Currently the default segmentation function is
used for all objects, but there are examples of custom ones to use.

The custom ones had various success, but the default ones work very well for
the current object configuration. However, when objects moved farther away, a new
segmentation function such as `SegmentFarMeatCan` was used. The reasoning behind
this is that when the object is farther away from the camera, the DOPE pose is
less accurate, so a larger initial area should be inspected. Then, since we know
we just placed the meat can on the bottom shelf and we know the location of the
bottom shelf, we can ignore points outside of that region.

To run this script and visualize the output in Meshcat, call
```sh
$ python run_dope_demo.py --meshcat --open_browser
```
from the command line. A browser window should open with Meshcat and the system
will simulate for a few seconds to allow the objects to settle and the arm to
move into its initial position. Then, DOPE will estimate the pose of every YCB
object it sees, and solid-colored boxes will appear on the point cloud at all of
the estimated locations. The colors are the same as DOPE uses to draw bounding
boxes around the objects in the images. The DOPE pose estimates will also be
printed out to the screen.

Then, `PoseRefinement` will run ICP over the specific objects in the script, and
more boxes will show up in meshcat representing the refined poses. Additionally,
all of the segmented point clouds will be shown in the drop down meshcat menu.
The updated poses will also be printed out to the screen.

At the end of the script, the annotated DOPE image will be displayed. Press any
key with the window active to close it and stop the script.
