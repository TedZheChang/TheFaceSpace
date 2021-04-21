import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import resize
import matplotlib.pyplot as plt
import cv2 
from skimage.transform import resize


def apply_filters(image, eye_filter, nose_filter, mouth_filter, keypoints):
    filtered_image = image
    # filter each portion of the face separately 
    if eye_filter is not None:
        filtered_image = apply_eyes_filter(filtered_image, eye_filter, np.take(keypoints, [2,3,0], axis=0), np.take(keypoints, [4,5,1], axis=0))
    if mouth_filter is not None:
        filtered_image = apply_mouth_filter(filtered_image, mouth_filter, np.take(keypoints, [11,12,13,14], axis=0))
    if nose_filter is not None:
        filtered_image = apply_nose_filter(filtered_image, nose_filter, keypoints[10])
    return filtered_image

# Applies the given filter image to the point on the image
def apply_filter_to_point(image, filter_image, point, scale):
    filter_image = resize(filter_image, (int(40*scale), int(40*scale), 3)) * 255
    filtered_image = image
    for y in range(0, filter_image.shape[0]):
        for x in range(0, filter_image.shape[1]):
            b, g, r = filter_image[y][x][0], filter_image[y][x][1], filter_image[y][x][2]
            if not approximately_white(b, g, r):
                filtered_image[int(point[0] - filter_image.shape[0] / 2 + y),
                int(point[1] - filter_image.shape[1] / 2 + x), :] = filter_image[y, x, :]
    return filtered_image

def apply_eyes_filter(image, filter_image, left_eye_coords, right_eye_coords):
    # eye coords are as follow: [(left_corner_y, left_corner_x), (right_corner_y, right_corner_x), (center_y, center_x)]
    if len(filter_image.shape) > 2 and filter_image.shape[2] == 4:
        filter_image = cv2.cvtColor(filter_image, cv2.COLOR_BGRA2BGR)
    filter_image = resize(filter_image, (40, 80, 3)) * 255
    filtered_image = image
    left_eye_center_x, left_eye_center_y = left_eye_coords[2][1], left_eye_coords[2][0]
    right_eye_center_x, right_eye_center_y = right_eye_coords[2][1], right_eye_coords[2][0]
    eye_x = int((left_eye_center_x + right_eye_center_x)/2)
    eye_y = int((left_eye_center_y + right_eye_center_y)/2)
    for y in range(0, filter_image.shape[0]):
        for x in range(0, filter_image.shape[1]):
            r,b,g = filter_image[y][x][0],filter_image[y][x][1],filter_image[y][x][2]
            if not approximately_white(b,g,r):
                x_loc = int(eye_x - filter_image.shape[1]/2 + x)
                y_loc = int(eye_y - filter_image.shape[0]/2 + y)
                filtered_image[y_loc, x_loc, :] = filter_image[y, x, :]
    return filtered_image

def apply_mouth_filter(image, filter_image, mouth_coords):
    # mouth coords are as follow: [(left_corner_y, left_corner_x), (right_corner_y, right_corner_x), (upper_lip_center_y, upper_lip_ceter_x),(lower_lip_center_y, lower_lip_ceter_x)]
    # check for 4-channel image & resize to appropriate size
    if len(filter_image.shape) > 2 and filter_image.shape[2] == 4:
        filter_image = cv2.cvtColor(filter_image, cv2.COLOR_BGRA2BGR)
    filter_image = resize(filter_image, (15, 30, 3)) * 255
    filtered_image = image
    mouth_center_x = int((mouth_coords[0][1] + mouth_coords[1][1])/2)
    mouth_center_y = int((mouth_coords[2][0] + mouth_coords[3][0])/2)
    for y in range(0, filter_image.shape[0]):
        for x in range(0, filter_image.shape[1]):
            r,b,g = filter_image[y][x][0],filter_image[y][x][1],filter_image[y][x][2]
            if not approximately_black(b,g,r):
                x_loc = int(mouth_center_x - filter_image.shape[1]/2 + x)
                y_loc = int(mouth_center_y - filter_image.shape[0]/2 + y)
                filtered_image[y_loc, x_loc, :] = filter_image[y, x, :]
    return filtered_image


def apply_nose_filter(image, filter_image, nose_coords):
    # nose coords are as follow: [(center_y, center_x)]
    if len(filter_image.shape) > 2 and filter_image.shape[2] == 4:  
        filter_image = cv2.cvtColor(filter_image, cv2.COLOR_BGRA2BGR)
    filter_image = resize(filter_image, (30, 40, 3)) * 255
    
    filtered_image = image
    for y in range(0, filter_image.shape[0]):
        for x in range(0, filter_image.shape[1]):
            b,g,r = filter_image[y][x][0],filter_image[y][x][1],filter_image[y][x][2]
            if not approximately_white(b,g,r):
                filtered_image[int(nose_coords[0] - filter_image.shape[0]/2 + y), int(nose_coords[1] - filter_image.shape[1]/2 + x), :] = filter_image[y, x, :]
    return filtered_image


def approximately_white(b, g, r):
    return b > 200 and g > 200 and r > 200

def approximately_black(b,g,r):
    return b < 1 and g < 1 and r < 1
