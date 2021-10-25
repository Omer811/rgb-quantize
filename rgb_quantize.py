"""
rgb quantization using median cut algorithm.
only for power of 2 quants.
"""
import numpy as np
import imageio
from skimage import color
GREYSCALE_DIMS = 2
BINS = 256
RGB = 2
GRAY_SCALE = 1
SCALE_LOW = 0
SCALE_HIGH = 255

def read_image(filename, representation):
    """
    this function reads a file in the given representation.
    :param filename: path to the image file
    :param representation: 1 for grayscale, 2 for rgb
    :return: array of the image in given path
    """
    im_matrix = imageio.imread(filename)
    if representation == GRAY_SCALE:
        im_matrix = color.rgb2gray(im_matrix)

    if im_matrix.dtype != np.float64:
        im_matrix = im_matrix.astype(np.float64)
        im_matrix /= SCALE_HIGH  # normalize values
    return im_matrix

"""
1. sort colors by largest range
2. sort pixels by range from largest to smallest (in place sort, order of
sort from smallest to largest)
3. cut the middle and repeat
"""


def _calc_avg_colors(buckets):
    avg_colors = np.empty([len(buckets),3])
    for i in range(0,len(buckets)):
        for j in range(0,3):
            avg_colors[i][j] = np.average(buckets[i][:,j])
    return avg_colors


def _paint_buckets(buckets, avg_colors):
    for i in range (0,len(buckets)):
        for j in range (0,3):
            buckets[i][:,j] = avg_colors[i,j]
    return buckets


def _process_colored_buckets(colored_buckets, size):
    im = np.empty([0,4])
    for i in range(0,len(colored_buckets)):
        im = np.concatenate((im,colored_buckets[i]))
    return im


def quantize_rgb(im_orig, n_quant):
    pixels = np.empty((im_orig[:,:,0].size,4))#this array will hold the
    # indexes of each pixel, and the rgb values of it
    pixels[:,3] = np.arange(pixels[:,3].size)#number all the pixels
    for i in range(0, 3):
        pixels[:,i] = im_orig[:,:,i].flatten()
    buckets = _generate_buckets(pixels,n_quant)
    avg_colors = _calc_avg_colors(buckets)
    colored_buckets = _paint_buckets(buckets,avg_colors)
    quant_im = _process_colored_buckets(colored_buckets,pixels.size)
    quant_im = quant_im[quant_im[:,3].argsort()]
    return quant_im[:,0:3].reshape(im_orig.shape)

def _generate_buckets(pixels,num_of_buckets, buckets = []):
    if num_of_buckets==1:
        buckets.append(pixels)
        return buckets

    c_range = np.empty([3], dtype="float64")
    for i in range(0, 3):
        c_range[i] = pixels[:, i].max() - pixels[:, i].min()
    sorting_color_order = np.argsort(c_range)  # get the right sorting
    # order
    for i in sorting_color_order:
        order = pixels[:, i].argsort(kind="stable")
        pixels = pixels[order]  # sort pixels
        # while preserving structure
    _generate_buckets(pixels[:pixels[:,0].size//2],
                              num_of_buckets//2)
    _generate_buckets(pixels[pixels[:, 0].size // 2 :],
                              num_of_buckets//2)
    return buckets

