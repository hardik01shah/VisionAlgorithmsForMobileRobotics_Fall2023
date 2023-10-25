import numpy as np


def selectKeypoints(scores, num, r):
    """
    Selects the num best scores as keypoints and performs non-maximum supression of a (2r + 1)*(2r + 1) box around
    the current maximum.
    """
    pass
    # TODO: Your code here
    scores = np.pad(scores, ((r,),(r,)))
    keypoints = np.zeros((2, num))
    for i in range(num):
        cur_max_ind = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
        keypoints[:,i] = cur_max_ind
        scores[cur_max_ind[0]-r:cur_max_ind[0]+r+1, cur_max_ind[1]-r:cur_max_ind[1]+r+1] = 0
    
    # return np.asarray(keypoints).T
    return keypoints-r