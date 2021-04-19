import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import resize
import matplotlib.pyplot as plt
import cv2 

def apply_filters(image, filters, keypoints):
    filtered_image = image
    # filter each portion of the face separately 
    filtered_image = apply_eyes_filter(filtered_image, filters[0], np.take(keypoints, [2,3,0], axis=0), np.take(keypoints, [4,5,1], axis=0))
    filtered_image = apply_mouth_filter(filtered_image, filters[1], np.take(keypoints, [11,12,13,14], axis=0))
    filtered_image = apply_nose_filter(filtered_image, filters[2], keypoints[10])
    return filtered_image
    
def apply_eyes_filter(image, filter_image, left_eye_coords, right_eye_coords):
    # eye coords are as follow: [(left_corner_y, left_corner_x), (right_corner_y, right_corner_x), (center_y, center_x)]
    return image

def apply_mouth_filter(image, filter_image, left_mouth_coords):
    # mouth coords are as follow: [(left_corner_y, left_corner_x), (right_corner_y, right_corner_x), (upper_lip_center_y, upper_lip_ceter_x),(lower_lip_center_y, lower_lip_ceter_x)]
    return image

def apply_nose_filter(image, filter_image, nose_coords):
    # nose coords are as follow: [(center_y, center_x)]
    filter_image = resize(filter_image, (40, 40, 3)) * 255
    filtered_image = image
    for y in range(0, filter_image.shape[0]):
        for x in range(0, filter_image.shape[1]):
            b,g,r = filter_image[y][x][0],filter_image[y][x][1],filter_image[y][x][2]
            if not approximately_white(b,g,r):
                filtered_image[int(nose_coords[0] - filter_image.shape[0]/2 + y), int(nose_coords[1] - filter_image.shape[1]/2 + x), :] = filter_image[y, x, :]
    return filtered_image

def approximately_white(b,g,r):
    return b > 200 and g > 200 and r > 200


    
