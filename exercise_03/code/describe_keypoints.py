import numpy as np


def describeKeypoints(img, keypoints, r):
    """
    Returns a (2r+1)^2xN matrix of image patch vectors based on image img and a 2xN matrix containing the keypoint
    coordinates. r is the patch "radius".
    """
    pass
    # TODO: Your code here
    descriptors = np.zeros(((2*r+1)**2,keypoints.shape[1]))
    img = np.pad(img, ((r,),(r,)))
    for i in range(keypoints.shape[1]):
        cur = (keypoints[:,i]+r).astype(int)
        desc = np.ravel(img[cur[0]-r:cur[0]+r+1, cur[1]-r:cur[1]+r+1], order='F')
        descriptors[:,i] = desc
    # return np.asarray(descriptors).T
    return descriptors
