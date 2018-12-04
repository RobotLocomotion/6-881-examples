# Code adapted from Clay Flannigan
import numpy as np

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

from sklearn.neighbors import NearestNeighbors


def FindNearestNeighbors(point_cloud_A, point_cloud_B):
    """
    Finds the nearest (Euclidean) neighbor in point_cloud_B for each
    point in point_cloud_A.

    @param point_cloud_A An Nx3 numpy array of points.
    @param point_cloud_B An Mx3 numpy array of points.

    @return distances An (N, ) numpy array of Euclidean distances from each
        point in point_cloud_A to its nearest neighbor in point_cloud_B.
    @return indices An (N, ) numpy array of the indices in point_cloud_B of each
        point_cloud_A point's nearest neighbor - these are the c_i's.
    """

    distances = np.zeros(point_cloud_A.shape[1])
    indices = np.zeros(point_cloud_A.shape[1])

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(point_cloud_B)
    distances, indices = neigh.kneighbors(point_cloud_A, return_distance=True)

    distances = distances.ravel()
    indices = indices.ravel()

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

    # number of dimensions
    m = 3
    X_BA = np.identity(4)

    # 1) translate points to their centroids
    centroid_A = np.mean(point_cloud_A, axis=0)
    centroid_B = np.mean(point_cloud_B, axis=0)
    centered_point_cloud_A = point_cloud_A - centroid_A
    centered_point_cloud_B = point_cloud_B - centroid_B

    # 2.1) rotation matrix
    H = np.dot(centered_point_cloud_A.T, centered_point_cloud_B)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # 2.2) special reflection case
    if np.linalg.det(R) < 0:
       Vt[m - 1, :] *= -1
       R = np.dot(Vt.T, U.T)

    # 3) translation vector
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # 4) construct the homogeneous transformation
    X_BA = np.identity(m + 1)
    X_BA[:m, :m] = R
    X_BA[:m, m] = t

    return X_BA


def RunICP(point_cloud_A, point_cloud_B,
        init_guess=None, max_iterations=20, tolerance=1e-3):
    """Finds best-fit transform that maps point_cloud_A on to point_cloud_B.

    @param point_cloud_A. An Nx3 numpy array of points to match to
        point_cloud_B.
    @param point_cloud_B An Nx3 numpy array of points
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

    # Transform from point_cloud_B to point_cloud_A
    # Overwrite this with ICP results.
    X_BA = np.identity(4)

    mean_error = 0
    num_iters = 0


    # Number of dimensions
    m = 3

    # Make homogeneous copies of boht point clouds
    point_cloud_Ah = np.ones((4, point_cloud_A.shape[0]))
    point_cloud_Bh = np.ones((4, point_cloud_B.shape[0]))
    point_cloud_Ah[:m, :] = np.copy(point_cloud_A.T)
    point_cloud_Bh[:m, :] = np.copy(point_cloud_B.T)

    # apply the initial pose estimation
    if init_guess is not None:
        point_cloud_Ah = np.dot(init_guess, point_cloud_Ah)

    prev_error = 0

    for num_iters in range(1, max_iterations + 1):
        # find the nearest neighbors between the current source and destination
        # points
        distances, indices = FindNearestNeighbors(point_cloud_Ah[:m, :].T,
                                               point_cloud_Bh[:m, :].T)

        # compute the transformation between the current source and nearest
        # destination points
        T = CalcLeastSquaresTransform(point_cloud_Ah[:m, :].T,
                                    point_cloud_Bh[:m, indices].T)

        # update the current source
        point_cloud_Ah = np.dot(T, point_cloud_Ah)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    X_BA = CalcLeastSquaresTransform(point_cloud_A, point_cloud_Ah[:m, :].T)

    return X_BA, mean_error, num_iters


def RepeatICPUntilGoodFit(point_cloud_A,
                              point_cloud_B,
                              error_threshold,
                              max_tries,
                              init_guess=None,
                              max_iterations=20,
                              tolerance=0.001):
    """Runs ICP until it converges to a "good" fit.

    Args:
    @param point_cloud_A An Nx3 numpy array of points to match to point_cloud_B.
    @param point_cloud_B An Nx3 numpy array of points.
    @param error_threshold float. The maximum allowed mean ICP error before
        stopping.
    @param max_tries int. Stop running ICP after max_tries if it hasn't produced
        a transform with an error < error_threshold.
    @param init_guess A 4x4 homogeneous transformation representing an initial
        guess of the transform. If one isn't provided, the 4x4 identity matrix
        will be used.
    @param max_iterations: int. If the algorithm hasn't converged after
            max_iterations, exit the algorithm.
    @param tolerance: float. The maximum difference in the error between two
            consecutive iterations before stopping.

    Returns:
    @return X_BA: A 4x4 numpy array of the homogeneous transformation matrix
        that maps point_cloud_A on to point_cloud_B such that

            X_BA x point_cloud_Ah ~= point_cloud_B,

        where point_cloud_Ah is a homogeneous version of point_cloud_A.
    @return mean_error: float. The mean of the Euclidean distances from each
        point in the transformed point_cloud_A to its nearest neighbor in
        point_cloud_B.
    @return num_runs: int. The total number of times ICP ran - not the total
        number of ICP iterations.
    """

    # Transform from point_cloud_B to point_cloud_A
    # Overwrite this with ICP results.
    X_BA = np.identity(4)

    mean_error = 1e8
    num_runs = 0

    while mean_error > error_threshold:
        X_BA, mean_error, num_iters = \
            RunICP(point_cloud_A,
                point_cloud_B,
                init_guess=init_guess,
                max_iterations=max_iterations,
                tolerance=tolerance)
        num_runs += 1
        print "mean_error", mean_error
        print "iters", num_iters
        if mean_error <= error_threshold or num_runs >= max_tries:
            break
        rand_R = tf.random_rotation_matrix()
        init_guess = np.dot(rand_R, X_BA)

    return X_BA, mean_error, num_runs