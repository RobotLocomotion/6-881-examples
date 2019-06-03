# Copyright 2016 Clay Flannigan
#
# Style, documentation, and nearest neighbor algorithm changes made by Katy
# Muhlrad 2018-2019
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from scipy.spatial import cKDTree


def FindNearestNeighbors(point_cloud_A, point_cloud_B):
    """
    Finds the nearest (Euclidean) neighbor in point_cloud_B for each
    point in point_cloud_A.

    @param point_cloud_A An Nx3 numpy array of points.
    @param point_cloud_B An Mx3 numpy array of points.

    @return distances An (N, ) numpy array of Euclidean distances from each
        point in point_cloud_A to its nearest neighbor in point_cloud_B.
    @return indices An (N, ) numpy array of the indices in point_cloud_B of
        each point_cloud_A point's nearest neighbor - these are the c_i's.
    """

    kd_tree = cKDTree(point_cloud_B, balanced_tree=False, compact_nodes=False)
    distances, indices = kd_tree.query(point_cloud_A, k=1, n_jobs=-1)

    return distances, indices


def CalcLeastSquaresTransform(point_cloud_A, point_cloud_B):
    """
    Calculates the least-squares best-fit transform that maps corresponding
    points point_cloud_A to point_cloud_B.

    @param point_cloud_A An Nx3 numpy array of corresponding points.
    @param point_cloud_B An Nx3 numpy array of corresponding points.

    @returns X_BA A 4x4 numpy array of the homogeneous transformation matrix
        that maps point_cloud_A on to point_cloud_B such that

            X_BA x point_cloud_Ah ~= point_cloud_B,

        where point_cloud_Ah is a homogeneous version of point_cloud_A.
    """

    # The number of dimensions.
    m = 3

    # Translate points to their centroids.
    centroid_A = np.mean(point_cloud_A, axis=0)
    centroid_B = np.mean(point_cloud_B, axis=0)
    centered_point_cloud_A = point_cloud_A - centroid_A
    centered_point_cloud_B = point_cloud_B - centroid_B

    # Calculate the rotation matrix.
    H = np.dot(centered_point_cloud_A.T, centered_point_cloud_B)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Handle the special reflection case.
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Calculate the translation vector.
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # Construct the homogeneous transformation.
    X_BA = np.identity(m + 1)
    X_BA[:m, :m] = R
    X_BA[:m, m] = t

    return X_BA


def RunICP(point_cloud_A, point_cloud_B, init_guess=None, max_iterations=20,
           tolerance=1e-3):
    """Finds best-fit transform that maps point_cloud_A on to point_cloud_B.

    @param point_cloud_A. An Nx3 numpy array of points to match to
        point_cloud_B.
    @param point_cloud_B An Mx3 numpy array of points.
    @param init_guess A 4x4 homogeneous transformation representing an initial
        guess of the transform. If one isn't provided, the 4x4 identity matrix
        will be used.
    @param max_iterations: int. If the algorithm hasn't converged after
        max_iterations, exit the algorithm.
    @param tolerance: float. The maximum difference in the error between two
        consecutive iterations before stopping.

    @return X_BA: A 4x4 numpy array of the homogeneous transformation matrix
        that maps point_cloud_A on to point_cloud_B such that

            X_BA x point_cloud_Ah ~= point_cloud_B,

        where point_cloud_Ah is a homogeneous version of point_cloud_A.
    @return mean_error: float. The mean of the Euclidean distances from each
        point in the transformed point_cloud_A to its nearest neighbor in
        point_cloud_B.
    @return num_iters: int. The total number of iterations run.
    """

    mean_error = 0
    num_iters = 0

    # The number of dimensions.
    m = 3

    # Make homogeneous copies of both point clouds.
    point_cloud_Ah = np.ones((4, point_cloud_A.shape[0]))
    point_cloud_Bh = np.ones((4, point_cloud_B.shape[0]))
    point_cloud_Ah[:m, :] = np.copy(point_cloud_A.T)
    point_cloud_Bh[:m, :] = np.copy(point_cloud_B.T)

    # Apply the initial pose estimation.
    if init_guess is not None:
        point_cloud_Ah = np.dot(init_guess, point_cloud_Ah)

    prev_error = 0

    for num_iters in range(1, max_iterations + 1):
        # Find the nearest neighbors between the current source and destination
        # points.
        distances, indices = FindNearestNeighbors(point_cloud_Ah[:m, :].T,
                                                  point_cloud_Bh[:m, :].T)

        # Compute the transformation between the current source and nearest
        # destination points.
        T = CalcLeastSquaresTransform(point_cloud_Ah[:m, :].T,
                                      point_cloud_Bh[:m, indices].T)

        # Update the current point_cloud_A.
        point_cloud_Ah = np.dot(T, point_cloud_Ah)

        # Check the error.
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # Calculate the final transformation.
    X_BA = CalcLeastSquaresTransform(point_cloud_A, point_cloud_Ah[:m, :].T)

    return X_BA, mean_error, num_iters
