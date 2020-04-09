import os

import numpy as np
from skimage.color import rgb2gray
from imageio import imread
from scipy.signal import convolve2d
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt


####################################################################################################################
## SECTION FROM EX1: ##

def read_image(filename, representation):
    im = imread(filename)
    if (im.dtype == np.uint8 or im.dtype == int or im.np.matrix.max > 1):
        im = im.astype(np.float64) / 255

    if ((representation == 1) and (len(im.shape) >= 3)):  # process image into gray scale

        im = rgb2gray(im)

    return im


####################################################################################################################

## 3.1 Gaussian & Laplacian pyramid construction : ##


def create_filter_vec(filter_size):
    """
    :param filter_size: number of convolutions needed
    :return: the ready filter vec
    """
    if (filter_size == 1):
        return [[1]]
    filter_vec = [[1, 1]]
    convolve_with = [[1, 1]]
    for i in range(1, filter_size - 1):
        filter_vec = convolve2d(convolve_with, filter_vec)
    sum = np.sum(filter_vec)
    normalized = (filter_vec / sum).astype(np.float64)
    return normalized


def reduceImage(im, filter_vec, is_filter_one):
    """
    :param im: image to reduce with filter vector
    :param filter_vec: filter to convolve with
    :return: image after reduce
    """
    # blur rows :
    handle_rows = convolve(im, filter_vec)
    # blur columns :
    if (is_filter_one):
        handle_columns = convolve(handle_rows, [[1]]);
    else:
        handle_columns = convolve(handle_rows, filter_vec.T)
    # sub-sample: select only every 2nd pixel in every 2nd column and 2nd row:
    reduced = handle_columns[::2, ::2]
    if reduced.shape[0] < 16 or reduced.shape[1] < 16:  # minimum dim of the lowest resolution is not smaller than 16.:
        return None;
    return reduced


def build_gaussian_pyramid(im, max_levels, filter_size):
    """

    :param im:  a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter
    :return:  pyr , filter_vec
    """

    filter_vec = create_filter_vec(filter_size)
    gaussian_array = [None] * max_levels
    gaussian_array[0] = im
    for i in range(1, max_levels):
        gaussian_array[i] = reduceImage(gaussian_array[i - 1], filter_vec, filter_size == 1)
        if gaussian_array[i] is None:  # means the size of img is too small already
            break
    trimmed = [i for i in gaussian_array if (i is not None)]
    return trimmed, filter_vec


def expand(im, filter_vec):
    # zero padding:
    padded_im = np.zeros(tuple(np.multiply(im.shape, 2)), dtype=np.float64)
    padded_im[::2, ::2] = im
    # blur:
    filter_vec = filter_vec * 2
    handle_rows = convolve(padded_im, filter_vec)
    handle_columns = convolve(handle_rows, filter_vec.T)
    return handle_columns


def build_laplacian_pyramid(im, max_levels, filter_size):
    laplacian_array = [None] * max_levels
    gausian_pyramid, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    for i in range(len(gausian_pyramid) - 1):
        laplacian_array[i] = (
                gausian_pyramid[i] - expand(gausian_pyramid[i + 1], filter_vec))  # maintain constant brightnes
    laplacian_array[max_levels - 1] = (gausian_pyramid[-1])
    trimmed = [i for i in laplacian_array if (i is not None)]  # trimming down the None
    return trimmed, filter_vec


####################################################################################################################

# 3.2

def laplacian_to_image(lpyr, filter_vec, coeff):
    for i in range(len(coeff)):
        lpyr[i] *= coeff[i]
    new_im = lpyr[-1]  # we take the last laplacian and start to expend from this point
    for i in range(len(lpyr) - 1, 0, -1):
        new_im = (lpyr[i - 1] + expand(new_im, filter_vec))
    return new_im


####################################################################################################################

# 3.3

def stretch_values(im, maxOrig, minOrig):
    return (im - minOrig) / (maxOrig - minOrig)


def render_pyramid(pyr, levels):
    if (levels > len(pyr)):  # in case not legal level entered
        levels = len(pyr)
    maxOrig = np.max(pyr[0])
    minOrig = np.min(pyr[0])
    res = stretch_values(pyr[0], maxOrig, minOrig)
    for i in range(1, levels):
        res = np.hstack((res, np.pad(stretch_values(pyr[i], maxOrig, minOrig),
                                     ((0, pyr[0].shape[0] - pyr[i].shape[0]), (0, 0)), 'constant')))
    return res


def display_pyramid(pyr, levels):
    res = render_pyramid(pyr, levels)
    plt.imshow(res, cmap='gray')
    plt.show()


####################################################################################################################

# 4.0

def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    # 1. Construct Laplacian pyramids L1 and L2 for the input images im1 and im2, respectively:
    lap1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)[0]
    lap2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]

    # 2. Construct a Gaussian pyramid Gm for the provided mask (convert it first to np.float64):
    mask_gausian, filter_vector = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)

    # 3. Construct the Laplacian pyramid Lout of the blended image for each level k by:
    l_out = []
    for k in range(len(lap1)):
        l_out.append((mask_gausian[k] * lap1[k]) + ((1 - mask_gausian[k]) * lap2[k]))

    # 4. Reconstruct the resulting blended image from the Laplacian pyramid Lout (using ones for coeffi- cients).
    res = laplacian_to_image(l_out, filter_vector, [1] * len(lap1))
    res = np.clip(res, 0, 1)
    return res


####################################################################################################################

# 4.1
def show_blending_process(im_list):
    plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        if (i == 2):
            plt.imshow(im_list[i], cmap='gray')
        else:
            plt.imshow(im_list[i])
    plt.savefig("wind_surfer")
    plt.show()


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def blending_example1():
    moon = read_image(relpath("moon.jpg"), 2)
    ice_cream = read_image(relpath("icecream.jpg"), 2)
    im_mask = read_image(relpath("mask1.jpg"), 1)
    mask = im_mask > 0.5
    mask = mask.astype(np.bool)
    im_blend = np.zeros(moon.shape)
    im_blend[:, :, 0] = pyramid_blending(ice_cream[:, :, 0], moon[:, :, 0], mask, 3, 5, 5)
    im_blend[:, :, 1] = pyramid_blending(ice_cream[:, :, 1], moon[:, :, 1], mask, 3, 5, 5)
    im_blend[:, :, 2] = pyramid_blending(ice_cream[:, :, 2], moon[:, :, 2], mask, 3, 5, 5)
    show_blending_process([moon, ice_cream, im_mask, im_blend])
    return ice_cream, moon, mask, im_blend


def blending_example2():
    windsurfer = read_image(relpath("wind.jpg"), 2)
    pool = read_image(relpath("pool.jpg"), 2)
    im_mask = read_image(relpath("mask2.jpg"), 1)
    mask = im_mask > 0.5
    mask = mask.astype(np.bool)
    im_blend = np.zeros(pool.shape)
    im_blend[:, :, 0] = pyramid_blending(pool[:, :, 0], windsurfer[:, :, 0], mask, 3, 5, 5)
    im_blend[:, :, 1] = pyramid_blending(pool[:, :, 1], windsurfer[:, :, 1], mask, 3, 5, 5)
    im_blend[:, :, 2] = pyramid_blending(pool[:, :, 2], windsurfer[:, :, 2], mask, 3, 5, 5)
    show_blending_process([windsurfer, pool, im_mask, im_blend])
    return pool, windsurfer, mask, im_blend


