import numpy as np


def pose_vector_to_transformation_matrix(pose_vec: np.ndarray) -> np.ndarray:
    """
    Converts a 6x1 pose vector into a 4x4 transformation matrix.

    Args:
        pose_vec: 6x1 vector representing the pose as [wx, wy, wz, tx, ty, tz]

    Returns:
        T: 4x4 transformation matrix
    """
    pass
    # TODO: Your code here
    W = pose_vec[:3]
    T = pose_vec[3:]
    theta = np.linalg.norm(W)
    assert theta == np.sqrt(W[0]**2+W[1]**2+W[2]**2)

    print(f"W: {W}")
    print(f"T: {T}")
    print(f"theta: {theta}")

    k = W/theta
    print(f"k: {k}")
    print(f"Norm of k: {np.linalg.norm(k)}")
    np.testing.assert_almost_equal(np.linalg.norm(k), 1.0)

    kx = np.zeros((3,3))
    kx[1,0] = k[2]
    kx[0,1] = -k[2]
    kx[2,0] = -k[1]
    kx[0,2] = k[1]
    kx[2,1] = k[0]
    kx[1,2] = -k[0]
    print(f"kx: {kx}")

    R = np.identity(3) + np.sin(theta)*kx + (1-np.cos(theta))*np.matmul(kx,kx)
    print(f"R: {R}")
    
    TM = np.identity(4)
    TM[:3,:3] = R
    TM[:3,3] = T
    print(f"TM: {TM}")

    return TM