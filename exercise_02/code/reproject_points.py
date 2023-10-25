import numpy as np

def reprojectPoints(P, M_tilde, K):
    # Reproject 3D points given a projection matrix
    #
    # P         [n x 3] coordinates of the 3d points in the world frame
    # M_tilde   [3 x 4] projection matrix
    # K         [3 x 3] camera matrix
    #
    # Returns [n x 2] coordinates of the reprojected 2d points

    pass
    # TODO: Your code here
    P_hom = np.column_stack((P, np.ones((P.shape[0],1))))
    reproj_pts_hom = K@M_tilde@(P_hom.T)
    reproj_pts = reproj_pts_hom[:2, :]/reproj_pts_hom[2, :]

    return (reproj_pts.T)
