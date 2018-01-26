from scipy.optimize import least_squares
from skimage.filters import threshold_li
import cv2
import numpy as np
import sys


def residual_func(rot_angle, img, img_180):
    rows, cols, z = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), rot_angle, 1)

    img_rot = cv2.warpAffine(img_180, M, (cols, rows))
    img_dif = img - img_rot
    img_dif = np.abs(img_dif)
    sum_all = np.sum(img_dif) * 0.000001
    #sum_all = np.sum(img_dif) * 0.00000001
    print rot_angle, sum_all
    return sum_all


def find_optimal_180_angle(img_name_0, img_name_180):
    img = cv2.imread(img_name_0)
    img_180 = cv2.imread(img_name_180)
    img = cv2.flip(img, 1)
    threshold = threshold_li(img)
    threshold_180 = threshold_li(img_180)
    binary = img > threshold
    binary = 1.0 * binary
    binary_180 = img_180 > threshold_180
    binary_180 = 1.0 * binary_180
    rot = least_squares(residual_func, 0.0, bounds=(0.0, 360.0), method='dogbox', diff_step=1.0, args=(binary, binary_180))
    #rot = least_squares(residual_func, 0.0, bounds=(0.0, 360.0), method='dogbox', diff_step=90.0, args=(img, img_180))
    return rot.success, rot.x


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'ipython find_opt_angle rotation_0.tif rotation_180.tif'
        exit(-1)
    #image_name_0 = 'faked_rot_0_tilted_5.3deg.tif'
    #image_name_180 = 'faked_rot_180.tif'
    image_name_0 = sys.argv[1]
    image_name_180 = sys.argv[2]
    print find_optimal_180_angle(image_name_0, image_name_180)