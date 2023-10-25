import numpy as np
from scipy import signal


def shi_tomasi(img, patch_size):
    """ Returns the shi-tomasi scores for an image and patch size patch_size
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

    term_1 = Ix2 + Iy2
    term_2 = np.sqrt(4*np.multiply(IxIy, IxIy)+np.square(Ix2-Iy2))
    lamda_1 = 0.5*(term_1+term_2)
    lamda_2 = 0.5*(term_1-term_2)

    R_init = np.minimum(lamda_1, lamda_2)
    R_init[R_init<0] = 0

    pad_value = (patch_size+1)//2
    R = np.pad(R_init, ((pad_value,), (pad_value,)))
    assert R.shape == img.shape

    return R