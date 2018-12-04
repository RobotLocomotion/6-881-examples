# Perception Tools

This directory contains scripts that may be useful for perception-related projects.


## Files

- `convert_obj_to_npy.py` will sample points from an .obj mesh file and store the (x, y, z) points in a serialized numpy array (.npy format).
- `convert_ply_to_npy.py` will save points and colors from a .ply mesh file and store the (x, y, z) points and (r, g, b) colors in two serialized numpy arrays (.npy format).
- `iterative_closest_point.py` is a library to run ICP on numpy point clouds and visualize them in meshcat. It was used in pset 2. 
- `optimization_based_point_cloud_registration.py` is a library to run 2d point cloud alignment on numpy point clouds and visualize them in meshcat. It was used in pset 3.
- `visualization_utils.py` contains methods used throughout the labs and psets that are useful for visualizing perception data.

## Using these scripts

### convert\_obj\_to\_npy.py

This script will sample points from an .obj mesh file and store the (x, y, z) points in a serialized numpy array (.npy format). It takes in a path to an .obj file, the number of points to sample, and the directory to save the resulting .npy file.

#### Example command-line usage:

```sh
$ python convert_obj_to_npy.py mesh_models/cup_model.obj 10000 numpy_models/
```
This will create a file named `cup_model.npy` in the `numpy_models` directory. The file can be used in a Python script to load a numpy array with the following line:

```py
cup_model_points = numpy.load("numpy_models/cup_model.npy")
```

### convert\_ply\_to\_npy.py

This script will save points and colors from a .ply mesh file and store the (x, y, z) points and (r, g, b) colors in two serialized numpy arrays (.npy format). It takes in a path to a .ply file and the directory to save the resulting .npy files.

#### Example command-line usage:

```sh
$ python convert_ply_to_npy.py mesh_models/cup_model.ply numpy_models/
```
This will create a file named `cup_model_points.npy` and a file named `cup_model_colors.npy` in the `numpy_models` directory. These files can be used in a Python script to load a numpy array with the following lines:

```py
cup_model_points = numpy.load("numpy_models/cup_model_points.npy")
cup_model_colors = numpy.load("numpy_models/cup_model_colors.npy")
```