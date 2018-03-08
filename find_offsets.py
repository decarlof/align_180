#!/bin/env python3
# -*- coding: utf-8 -*-

import logging
log = logging.getLogger(__name__)

import sys
import os
import argparse
import math

from scipy.ndimage import rotate, shift
from scipy.optimize import least_squares
from skimage.filters import threshold_li
from skimage.io import imread
from skimage.feature import register_translation
from skimage.transform import AffineTransform, warp, FundamentalMatrixTransform
import numpy as np
import tqdm

import matplotlib.pyplot as plt

__authors__ = "Mark Wolfman"
__copyright__ = "Copyright (c) 2017, Argonne National Laboratory"
__version__ = "0.0.1"
__all__ = ['alignment_pass',
           'transform_image',
           'image_corrections']

def flip(m, axis):
    if not hasattr(m, 'ndim'):
        m = asarray(m)
    indexer = [slice(None)] * m.ndim
    try:
        indexer[axis] = slice(None, None, -1)
    except IndexError:
        raise ValueError("axis=%i is invalid for the %i-dimensional input array"
                         % (axis, m.ndim))
    return m[tuple(indexer)]


def alignment_pass(img, img_180):
    upsample = 2
    # Register the translation correction
    trans = register_translation(img, img_180, upsample_factor=upsample)
    trans = trans[0]
    # Register the rotation correction
    lp_center = (int(img.shape[0] / 2), int(img.shape[1] / 2))
    img_lp = logpolar_fancy(img, *lp_center, crop=True)
    img_180_lp = logpolar_fancy(img_180, *lp_center, crop=True)
    result = register_translation(img_lp, img_180_lp, upsample_factor=upsample*10)
    scale_rot = result[0]
    angle = np.degrees(scale_rot[1] / img_lp.shape[1] * 2 * np.pi)
    return angle, trans


def _transformation_matrix(r=0, tx=0, ty=0, sx=1, sy=1):
    # Prepare the matrix
    ihat = [ sx * np.cos(r), sx * np.sin(r), 0]
    jhat = [-sy * np.sin(r), sy * np.cos(r), 0]
    khat = [ tx,             ty,             1]
    # Make the eigenvectors into column vectored matrix
    new_transform = np.array([ihat, jhat, khat]).swapaxes(0, 1)
    return new_transform


def transform_image(img, rotation=0, translation=(0, 0), crop=False):
    """Take a set of transformations and apply them to the image.
    
    Rotations occur around the center of the image, rather than the
    (0, 0).
    
    Parameters
    ----------
    translation : 2-tuple, optional
      Translation parameters in (vert, horiz) order.
    rotation : float, optional
      Rotation in degrees.
    scale : 2-tuple, optional
      Scaling parameters in (vert, horiz) order.
    crop : bool, optional
      If true (default), clip the dimensions of the image to avoid
      zero values.
    
    Returns
    -------
    out : np.ndarray
      Similar to input array but transformed. Dimensions will be
      different if ``crop=True``.
    crops : 4-tuple
      The dimensions used for cropping in order (v_min, v_max, h_min,
      h_max)
    
    """
    rot_center = (img.shape[1] / 2, img.shape[0] / 2)
    xy_trans = (translation[1], translation[0])
    M0 = _transformation_matrix(tx=-rot_center[0], ty=-rot_center[1])
    M1 = _transformation_matrix(r=np.radians(rotation), tx=xy_trans[0], ty=xy_trans[1])
    M2 = _transformation_matrix(tx=rot_center[0], ty=rot_center[1])
    # python 3.6
    # M = M2 @ M1 @ M0
    MT = np.dot(M1, M0)
    M = np.dot(M2, MT)
    tr = FundamentalMatrixTransform(M)
    out = warp(img, tr, preserve_range=True)
    # Calculate new boundaries if needed
    # Adjust for rotation
    h_min = 0.5 * img.shape[0] * np.tan(np.radians(rotation))
    v_min = 0.5 * img.shape[1] * np.tan(np.radians(rotation))
    # Adjust for translation
    v_max = min(img.shape[0], img.shape[0] - v_min - translation[0])
    h_max = min(img.shape[1], img.shape[1] - h_min - translation[1])
    v_min = max(0, v_min - translation[0])
    h_min = max(0, h_min - translation[1])
    crops = (int(v_min), int(v_max), int(h_min), int(h_max))
    # Apply the cropping
    if crop:
        out = out[crops[0]:crops[1],
                  crops[2]:crops[3]]
    return out, crops


def image_corrections(img_name_0, img_name_180, passes=15, crop=False, view=False):
    """Report translation, rotation alignment for two images.
    
    The second image is assumed to be the same object as the first
    image but rotated 180° in 3D space along an axis parallel to the
    image plane.
    
    Parameters
    ----------
    img_name_0 : str
      Filename for the original image.
    img_name_180 : str
      Filename for the rotated image.
    passes : int, optional
      How many iterations of alignment to do.
    crop : bool, optional
      Whether to internally crop the images after translation.
    
    Returns
    -------
    cume_angle : float
      Cumulative angle over all passes (in degrees)
    cume_trans : tuple(float, float)
      Cumulative translation (vert, horiz) over all passes (in
      pixels).
    
    """
    img_0 = imread(img_name_0)
    img_180 = imread(img_name_180)
    # python 3.6
    img = flip(img_0, 1)
    cume_angle = 0
    cume_trans = np.array([0, 0], dtype=float)
    for pass_ in tqdm.tqdm(range(passes)):
        # Prepare the inter-translated images
        working_img, crops = transform_image(img,
                                             translation=cume_trans,
                                             rotation=cume_angle,
                                             crop=crop)
        # Calculate a new transformation
        if crop:
            working_img180 = img_180[crops[0]:crops[1], crops[2]:crops[3]]
        else:
            working_img180 = img_180
        angle, trans = alignment_pass(working_img, working_img180)
        log.debug("Pass {}: {:.4f}, {}".format(pass_, angle, trans))
        # Save the cumulative transformations
        cume_angle += angle
        cume_trans += np.array(trans)
    # Convert translations to (x, y)
    cume_trans = (-cume_trans[1], cume_trans[0])
    if view:
        img_rot = rotate(img, cume_angle, mode='wrap')
        img_shift = shift(img_rot, (cume_trans[1], cume_trans[0]), mode='wrap')
        img_crop = crop_center(img_shift, img_0.shape[0], img_0.shape[1])
        img_diff = img_crop-img_180

        images = (img_0, img_180, img_crop, img_diff)
        title = ('0 deg', '180 deg', '180 deg flip/correct', 'Difference')
        show_images(images, titles=title, cols=2)

    return cume_angle, cume_trans


_transforms = {}

def _get_transform(i_0, j_0, i_n, j_n, p_n, t_n, p_s, t_s):
    transform = _transforms.get((i_0, j_0, i_n, j_n, p_n, t_n))
    if transform == None:
        i_k = []
        j_k = []
        p_k = []
        t_k = []
        for p in range(0, p_n):
            p_exp = np.exp(p * p_s)
            for t in range(0, t_n):
                t_rad = t * t_s
                i = int(i_0 + p_exp * np.sin(t_rad))
                j = int(j_0 + p_exp * np.cos(t_rad))
                if 0 <= i < i_n and 0 <= j < j_n:
                    i_k.append(i)
                    j_k.append(j)
                    p_k.append(p)
                    t_k.append(t)
        transform = ((np.array(p_k), np.array(t_k)), (np.array(i_k), np.array(j_k)))
        _transforms[i_0, j_0, i_n, j_n, p_n, t_n] = transform
    return transform


def logpolar_fancy(image, i_0=None, j_0=None, p_n=None, t_n=None, crop=False):

    """Convert an image from cartesian to log-polar coordinates.
    
    Parameters
    ----------
    image : np.ndarray
      2-dimensional array with image data to transform.
    i_0, j_0 : int, optional
      Coords of ``image`` to use as center of rotation. i_0 is dim 0
      (vertical) and j_0 is dim 1 (horizontal). If omitted, center of
      image will be used.
    p_n : int, optional
      Number of output pixels for radii.
    t_n : int, optional
      Number of output pixels for angles.
    crop : bool, optional
      If true, only radii that are fully contained within the image
      are considered, cropping out the corners. Otherwise, the whole
      image will be used and non-existance coordinates will be zero.
    
    """
    # Default for image center
    if i_0 is None:
        i_0 = int(image.shape[0] / 2)
    if j_0 is None:
        j_0 = int(image.shape[1] / 2)
    log.debug("Using image center ({}v, {}h)".format(i_0, j_0))
    
    (i_n, j_n) = image.shape[:2]
    
    # Calculate furthest distances
    i_c = max(i_0, i_n - i_0)
    j_c = max(j_0, j_n - j_0)
    if crop:
        d_c = min(i_c, j_c)
    else:
        d_c = (i_c ** 2 + j_c ** 2) ** 0.5
    
    if p_n == None:
        p_n = int(np.ceil(d_c))
    
    if t_n == None:
        t_n = j_n
    
    p_s = np.log(d_c) / p_n
    t_s = 2.0 * np.pi / t_n
    
    (pt, ij) = _get_transform(i_0, j_0, i_n, j_n, p_n, t_n, p_s, t_s)
    
    transformed = np.zeros((p_n, t_n) + image.shape[2:], dtype=image.dtype)
    
    transformed[pt] = image[ij]
    return transformed

def crop_center(img,cropx,cropy):
    x,y = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return img[startx:startx+cropx, starty:starty+cropy]

def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


if __name__ == '__main__':
    # Prepare arguments
    parser = argparse.ArgumentParser(
        description='Compare two images and get rotation/translation offsets.')
    parser.add_argument('original_image', help='The original image file')
    parser.add_argument('flipped_image',
                        help='Image of the specimen after 180 stage rotation.')
    parser.add_argument('--passes', '-p',
                        help='How many iterations to run.',
                        default=15, type=int)
    parser.add_argument("--show", default=False, action="store_true" , 
                        help="Flag to show images.")
    parser.add_argument('-d', '--debug', action='store_true',
                        help="Show detailed logging and disable threading.")

    args = parser.parse_args()
    # Setup logging
    if args.debug:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.WARNING
    logging.basicConfig(level=loglevel)    
    # Perform the correction calculation
    rot, trans = image_corrections(args.original_image, args.flipped_image,
                                   passes=args.passes, view=args.show)
    # Display the result
    # python 3.6
    print("DR: {:.2f}°, DX: {:.2f} px, DY: {:.2f} px"
          .format(rot, trans[0], trans[1]))
