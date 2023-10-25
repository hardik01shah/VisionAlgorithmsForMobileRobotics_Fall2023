import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

from pose_vector_to_transformation_matrix import \
    pose_vector_to_transformation_matrix
from project_points import project_points
from undistort_image import undistort_image
from undistort_image_vectorized import undistort_image_vectorized


def main():
    pass

    # load camera poses

    # each row i of matrix 'poses' contains the transformations that transforms
    # points expressed in the world frame to
    # points expressed in the camera frame

    # TODO: Your code here
    cam_poses = np.loadtxt('01_camera_projection - exercise/data/poses.txt', dtype=np.longdouble)
    print(f"Camera pose of image 1: {cam_poses[0]}")

    # define 3D corner positions
    # [Nx3] matrix containing the corners of the checkerboard as 3D points
    # (X,Y,Z), expressed in the world coordinate system

    # TODO: Your code here
    x_ind = np.arange(9)
    y_ind = np.arange(6)
    x, y = np.meshgrid(x_ind, y_ind)
    Pw = np.column_stack((np.ndarray.flatten(x),
                          np.ndarray.flatten(y),
                          np.zeros_like(np.ndarray.flatten(x))))*0.04
    # print(f"Pw: {Pw}")

    # load camera intrinsics
    # TODO: Your code here
    cam_matrix = np.loadtxt('01_camera_projection - exercise/data/K.txt', dtype=np.longdouble)
    print(f"Camera Matrix (K): {cam_matrix}")
    cam_distortion = np.loadtxt('01_camera_projection - exercise/data/D.txt', dtype=np.longdouble)
    print(f"Camera Distortion (D): {cam_distortion}")

    # load one image with a given index
    # TODO: Your code here
    # img_undistorted = cv2.imread('01_camera_projection - exercise/data/images_undistorted/img_0001.jpg',cv2.IMREAD_GRAYSCALE)
    img = cv2.imread('01_camera_projection - exercise/data/images/img_0001.jpg',cv2.IMREAD_GRAYSCALE)


    # project the corners on the image
    # compute the 4x4 homogeneous transformation matrix that maps points
    # from the world to the camera coordinate frame

    # TODO: Your code here
    TM = pose_vector_to_transformation_matrix(cam_poses[0])


    # transform 3d points from world to current camera pose
    # TODO: Your code here
    Pw_homogenous = np.column_stack((Pw, np.ones((Pw.shape[0],1))))
    print(f"Pw_homogenous[:5]: {Pw_homogenous[:5].T}")
    Pc_homogenous = np.matmul(TM, Pw_homogenous.T)
    print(f"Pc_homogenous[:,:5]: {Pc_homogenous[:,:5]}")
    Pc = Pc_homogenous[:3,:]/Pc_homogenous[3,:]
    print(f"Pc[:,:5]: {Pc[:,:5]}")

    # plot checkerboard corners
    check_pts = project_points(points_3d=Pc, K=cam_matrix, D=cam_distortion)
    print(f"check_pts[:,:5]: {check_pts[:,:5]}")
    plt.clf()
    plt.close()
    plt.imshow(img, cmap='gray')
    plt.scatter(check_pts[0], check_pts[1], marker='o', color='r')
    plt.show()

    img_undistorted_vectorized = undistort_image_vectorized(img, cam_matrix, cam_distortion)
    plt.clf()
    plt.close()
    plt.imshow(img_undistorted_vectorized, cmap='gray')
    plt.show()
    
    # undistort image with bilinear interpolation
    # """ Remove this comment if you have completed the code until here
    start_t = time.time()
    img_undistorted = undistort_image(img, K=cam_matrix, D=cam_distortion, bilinear_interpolation=True)
    print('Undistortion with bilinear interpolation completed in {}'.format(
        time.time() - start_t))

    # vectorized undistortion without bilinear interpolation
    start_t = time.time()
    img_undistorted_vectorized = undistort_image_vectorized(img, K=cam_matrix, D=cam_distortion)
    print('Vectorized undistortion completed in {}'.format(
        time.time() - start_t))
    
    plt.clf()
    plt.close()
    fig, axs = plt.subplots(2)
    axs[0].imshow(img_undistorted, cmap='gray')
    axs[0].set_axis_off()
    axs[0].set_title('With bilinear interpolation')
    axs[1].imshow(img_undistorted_vectorized, cmap='gray')
    axs[1].set_axis_off()
    axs[1].set_title('Without bilinear interpolation')
    plt.show()
    # """

    # calculate the cube points to then draw the image
    # TODO: Your code here
    cube_origin = (3,1)
    cube_length = 2

    cube_base_idx_0 = cube_origin[1]*9 +cube_origin[0]
    cube_base_idx_1 = (cube_origin[1]+cube_length)*9 +cube_origin[0]
    cube_base_idx_2 = cube_origin[1]*9 +(cube_origin[0]+cube_length)
    cube_base_idx_3 = (cube_origin[1]+cube_length)*9 +(cube_origin[0]+cube_length)
    cube_base = np.row_stack((
        Pw_homogenous[cube_base_idx_0,:],
        Pw_homogenous[cube_base_idx_1,:],
        Pw_homogenous[cube_base_idx_2,:],
        Pw_homogenous[cube_base_idx_3,:],
        ))
    cube_w = np.row_stack((
        cube_base, cube_base
    ))
    cube_w[4:,2] = np.ones(4)*(-1*cube_length*0.04)
    cube_c_homogenous = np.matmul(TM, cube_w.T)
    cube_c = cube_c_homogenous[:3,:]/cube_c_homogenous[3,:]
    cube_pts = project_points(points_3d=cube_c, K=cam_matrix, D=None)
    cube_pts = cube_pts.T

    # Plot the cube
    # """ Remove this comment if you have completed the code until here
    plt.clf()
    plt.close()
    plt.imshow(img_undistorted, cmap='gray')

    lw = 1

    # base layer of the cube
    plt.plot(cube_pts[[1, 3, 7, 5, 1], 0],
             cube_pts[[1, 3, 7, 5, 1], 1],
             'r-',
             linewidth=lw)

    # top layer of the cube
    plt.plot(cube_pts[[0, 2, 6, 4, 0], 0],
             cube_pts[[0, 2, 6, 4, 0], 1],
             'r-',
             linewidth=lw)

    # vertical lines
    plt.plot(cube_pts[[0, 1], 0], cube_pts[[0, 1], 1], 'r-', linewidth=lw)
    plt.plot(cube_pts[[2, 3], 0], cube_pts[[2, 3], 1], 'r-', linewidth=lw)
    plt.plot(cube_pts[[4, 5], 0], cube_pts[[4, 5], 1], 'r-', linewidth=lw)
    plt.plot(cube_pts[[6, 7], 0], cube_pts[[6, 7], 1], 'r-', linewidth=lw)

    plt.show()
    # """


if __name__ == "__main__":
    main()
