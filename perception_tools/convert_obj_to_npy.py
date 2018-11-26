# Read in an .obj file and sample points on the surface and randomly sample
# points with probabilities proportional to the area of each triangle.

import sys
import numpy as np

import trimesh

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print "Please supply the path of the .obj file to convert, the number of points to sample, and the location to save the .npy file."
        sys.exit()

    obj_file = sys.argv[1]
    num_points = int(sys.argv[2])
    npy_file_path = sys.argv[3]

    mesh = trimesh.load(obj_file, file_type='obj')
    samples, _ = trimesh.sample.sample_surface(mesh, num_points)

    file_name = obj_file.split("/")[-1]
    file_name = file_name.split(".")[0]

    np.save(npy_file_path + "/" + file_name, samples)