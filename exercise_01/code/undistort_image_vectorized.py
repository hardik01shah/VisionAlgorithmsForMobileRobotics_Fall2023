import numpy as np

from distort_points import distort_points


def undistort_image_vectorized(img: np.ndarray,
                               K: np.ndarray,
                               D: np.ndarray) -> np.ndarray:

    """
    Undistorts an image using the camera matrix and distortion coefficients.

    Args:
        img: distorted image (HxW)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)

    Returns:
        und_img: undistorted image (HxW)
    """
    pass
    # TODO: Your code here
    h, w = img.shape
    u_ind = np.arange(w)
    v_ind = np.arange(h)
    u, v = np.meshgrid(u_ind, v_ind)
    Iu_idx = np.column_stack((
        np.ndarray.flatten(u),
        np.ndarray.flatten(v),
    )).T

    Id_idx = distort_points(Iu_idx, D, K)
    Id_idx_nn = Id_idx.astype(int)

    und_img = np.zeros_like(img)
    Iu_idx_row=Iu_idx[1].reshape(img.shape)
    Iu_idx_col=Iu_idx[0].reshape(img.shape)

    Id_idx_row=Id_idx_nn[1].reshape(img.shape)
    Id_idx_col=Id_idx_nn[0].reshape(img.shape)

    und_img[Iu_idx_row, Iu_idx_col] = img[Id_idx_row, Id_idx_col]

    return und_img


