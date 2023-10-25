import numpy as np


def distort_points(x: np.ndarray,
                   D: np.ndarray,
                   K: np.ndarray) -> np.ndarray:
    """
    Applies lens distortion to 2D points xon the image plane.

    Args:
        x: 2d points (Nx2)
        D: distortion coefficients (4x1)
        K: camera matrix (3x3)
    """
    pass
    # TODO: Your code here
    principle_pt = np.expand_dims(K[:2,2],1)

    disp = (x-principle_pt)**2
    r2 = disp[0]+disp[1]
    distorted_points = (1+(D[0]*r2)+(D[1]*r2**2))*(x-principle_pt) + principle_pt

    return distorted_points