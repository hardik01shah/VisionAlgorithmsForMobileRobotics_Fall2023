import numpy as np

from distort_points import distort_points


def project_points(points_3d: np.ndarray,
                   K: np.ndarray,
                   D: np.ndarray) -> np.ndarray:
    """
    Projects 3d points to the image plane, given the camera matrix,
    and distortion coefficients.

    Args:
        points_3d: 3d points (3xN)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)

    Returns:
        projected_points: 2d points (2xN)
    """
    pass
    # TODO: Your code here
    projected_points_lambda = np.matmul(K, points_3d)
    projected_points = projected_points_lambda[:2,:]/projected_points_lambda[2,:]
    
    if D is not None:
        projected_points_d = distort_points(projected_points, D, K)
        return projected_points_d

    return projected_points