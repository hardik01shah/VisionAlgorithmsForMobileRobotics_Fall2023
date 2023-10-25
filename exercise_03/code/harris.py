import numpy as np
from scipy import signal


def harris(img, patch_size, kappa):
    """ Returns the harris scores for an image given a patch size and a kappa value
        The returned scores are of the same shape as the input image """

    pass
    # TODO: Your code here
    sobel_x = np.array([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.],
    ])
    sobel_y = np.array([
        [-1., -2., -1.],
        [0., 0., 0.],
        [1., 2., 1.],
    ])
    Ix = signal.convolve2d(img, sobel_x, mode='valid')
    Iy = signal.convolve2d(img, sobel_y, mode='valid')

    Ix2 = np.multiply(Ix, Ix)
    Iy2 = np.multiply(Iy, Iy)
    IxIy = np.multiply(Ix, Iy)

    box_filter = np.ones((patch_size, patch_size))
    Ix2 = signal.convolve2d(Ix2, box_filter, mode='valid')
    Iy2 = signal.convolve2d(Iy2, box_filter, mode='valid')
    IxIy = signal.convolve2d(IxIy, box_filter, mode='valid')

    det_M = np.multiply(Ix2,Iy2) - np.multiply(IxIy, IxIy)
    trace2_M = np.square(Ix2 + Iy2)

    R_init = det_M - kappa*trace2_M
    R_init[R_init<0] = 0
    
    pad_value = (patch_size+1)//2
    R = np.pad(R_init, ((pad_value,), (pad_value,)))
    assert R.shape == img.shape

    return R
