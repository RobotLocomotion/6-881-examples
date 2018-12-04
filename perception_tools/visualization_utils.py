import numpy as np

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf


def ThresholdArray(arr, min_val, max_val):
    """
    Finds where the values of arr are between min_val and max_val (inclusive).

    @param arr An (N, ) numpy array containing number values.
    @param min_val number. The minimum value threshold.
    @param max_val number. The maximum value threshold.

    @return An (M, ) numpy array of the integer indices in arr with values that
        are between min_val and max_val.
    """
    return np.where(
        abs(arr - (max_val + min_val) / 2.) < (max_val - min_val) / 2.)[0]


def MakeMeshcatColorArray(N, r, g, b):
    """Constructs a color array to visualize a point cloud in meshcat.

    @param N int. Number of points to generate. Must be >= number of points in
        the point cloud to color.
    @param r float. The red value of the points, 0.0 <= r <= 1.0.
    @param g float. The green value of the points, 0.0 <= g <= 1.0.
    @param b float. The blue value of the points, 0.0 <= b <= 1.0.

    @return Nx3 numpy array of the same color.
    """
    color = np.zeros((3, N))
    color[0, :] = r
    color[1, :] = g
    color[2, :] = b

    return color.T


def PlotMeshcatPointCloud(meshcat_vis, point_cloud_name, points, colors):
    """A wrapper function to plot meshcat point clouds.

    Args:
    @param meshcat_vis An instance of a meshcat visualizer.
    @param point_cloud_name string. The name of the meshcat point clouds.
    @param points An Nx3 numpy array of (x, y, z) points.
    @param colors An Nx3 numpy array of (r, g, b) colors corresponding to
        points.
    """

    meshcat_vis[point_cloud_name].set_object(g.PointCloud(points.T, colors.T))


def ClearVis(meshcat_vis):
    """
    Removes model, observations, and transformed_observations objects
    from meshcat.

    @param meshcat_vis An instance of a meshcat visualizer.
    """

    meshcat_vis['model'].delete()
    meshcat_vis['observations'].delete()
    meshcat_vis['transformed_observations'].delete()


def VisualizeTransform(meshcat_vis, points, transform):
    """Visualizes the points transformed by transform in yellow.

    Args:
    @param meshcat_vis An instance of a meshcat visualizer.
    @param points An Nx3 numpy array representing a point cloud.
    @param transform a 4x4 numpy array representing a homogeneous
        transformation.
    """

    ClearVis(meshcat_vis)

    N = points.shape[0]
    yellow = MakeMeshcatColorArray(N, 1, 1, 0)

    homogenous_points = np.ones((N, 4))
    homogenous_points[:, :3] = np.copy(points)

    transformed_points = transform.dot(homogenous_points.T)

    PlotMeshcatPointCloud(meshcat_vis,
                          'transformed_observations',
                          transformed_points[:3, :].T,
                          yellow)


def VisualizeTransformedSceneAndModel(meshcat_vis, scene, model, X_MS):
    """
    Visualizes ground truth (red), observation (blue), and transformed
    (yellow) point clouds in meshcat.

    @param meshcat_vis An instance of a meshcat visualizer.
    @param scene An Nx3 numpy array representing the scene point cloud.
    @param model An Mx3 numpy array representing the model point cloud.
    @param X_MS A 4x4 numpy array of the homogeneous transformation from the
            scene point cloud to the model point cloud.
    """

    ClearVis(meshcat_vis)

    # Make meshcat color arrays.
    N = scene.shape[0]
    M = model.shape[0]

    red = MakeMeshcatColorArray(M, 0.5, 0, 0)
    blue = MakeMeshcatColorArray(N, 0, 0, 0.5)
    yellow = MakeMeshcatColorArray(N, 1, 1, 0)

    # Create red and blue meshcat point clouds for visualization.
    PlotMeshcatPointCloud(meshcat_vis, 'model', model, red)
    PlotMeshcatPointCloud(meshcat_vis, 'observations', scene, blue)

    # Create a copy of the scene point cloud that is homogenous
    # so we can apply a 4x4 homogenous transform to it.
    homogenous_scene = np.ones((N, 4))
    homogenous_scene[:, :3] = np.copy(scene)

    # Apply the returned transformation to the scene samples to align the
    # scene point cloud with the ground truth point cloud.
    transformed_scene = X_MS.dot(homogenous_scene.T)

    # Create a yellow meshcat point cloud for visualization.
    PlotMeshcatPointCloud(meshcat_vis,
                          'transformed_observations',
                          transformed_scene[:3, :].T,
                          yellow)