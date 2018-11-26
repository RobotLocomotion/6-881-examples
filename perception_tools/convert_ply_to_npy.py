# Read in an .ply file and convert it to two numpy arrays,
# one containing the (x, y, z) points and the other containing
# the corresponding (r, g, b) colors.

import sys
import numpy as np

from plyfile import PlyData, PlyElement

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Please supply the path of the .ply file to convert and the location to save the .npy file."
        sys.exit()

    ply_file = sys.argv[1]
    npy_file_path = sys.argv[2]

    mesh = PlyData.read(ply_file)

    num_points = mesh['vertex']['x'].shape[0]

    mesh_points = np.zeros((3, num_points))
    mesh_points[0, :] = mesh['vertex']['x']
    mesh_points[1, :] = mesh['vertex']['y']
    mesh_points[2, :] = mesh['vertex']['z']

    colors = np.zeros((3, num_points))
    colors[0, :] = mesh['vertex']['red']/255.
    colors[1, :] = mesh['vertex']['green']/255.
    colors[2, :] = mesh['vertex']['blue']/255.

    file_name = ply_file.split("/")[-1]
    file_name = file_name.split(".")[0]

    np.save(npy_file_path + "/" + file_name + "_points", mesh_points)
    np.save(npy_file_path + "/" + file_name + "_colors", colors)
