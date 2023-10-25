import numpy as np

def estimatePoseDLT(p, P, K):
    # Estimates the pose of a camera using a set of 2D-3D correspondences
    # and a given camera matrix.
    # 
    # p  [n x 2] array containing the undistorted coordinates of the 2D points
    # P  [n x 3] array containing the 3D point positions
    # K  [3 x 3] camera matrix
    #
    # Returns a [3 x 4] projection matrix of the form 
    #           M_tilde = [R_tilde | alpha * t] 
    # where R is a rotation matrix. M_tilde encodes the transformation 
    # that maps points from the world frame to the camera frame

    pass

    # Convert 2D to normalized coordinates
    # TODO: Your code here
    p_hom = np.column_stack((p, np.ones((p.shape[0],1)))).T
    K_inv = np.linalg.inv(K)
    p_norm = K_inv@p_hom
    p_norm = p_norm/p_norm[2,:]

    # Build measurement matrix Q
    # TODO: Your code here
    P_hom = np.column_stack((P, np.ones((P.shape[0],1))))
    P_hom_x = np.multiply(P_hom, -p_norm[0, :].reshape(-1, 1))
    P_hom_y = np.multiply(P_hom, -p_norm[1, :].reshape(-1, 1))
    P_hom_zeros_x = np.column_stack((P_hom, np.zeros_like(P_hom), P_hom_x))
    zeros_P_hom_y = np.column_stack((np.zeros_like(P_hom), P_hom, P_hom_y))
    Q = np.column_stack((P_hom_zeros_x, zeros_P_hom_y)).reshape(-1, P_hom_zeros_x.shape[1])

    # Solve for Q.M_tilde = 0 subject to the constraint ||M_tilde||=1
    # TODO: Your code here
    _, _, Vh = np.linalg.svd(Q)
    V = Vh.T
    M_tilde = V[:,-1].reshape(3,4)

    # wrong. CHeck if determinant of R is negative
    if(M_tilde[2,3]<0): M_tilde = -1*M_tilde
    
    # Extract [R | t] with the correct scale
    # TODO: Your code here
    R = M_tilde[:3, :3]
    t = M_tilde[:3, 3]

    # Find the closest orthogonal matrix to R
    # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    # TODO: Your code here
    U, _, Vh = np.linalg.svd(R)
    R_tilde = U@Vh

    # Normalization scheme using the Frobenius norm:
    # recover the unknown scale using the fact that R_tilde is a true rotation matrix
    # TODO: Your code here
    alpha = np.linalg.norm(R_tilde)/np.linalg.norm(R)

    # Build M_tilde with the corrected rotation and scale
    # TODO: Your code here
    M_tilde = np.column_stack((R_tilde, alpha*t))
    return M_tilde