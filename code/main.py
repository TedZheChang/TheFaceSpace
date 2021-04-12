import cv2
from models import *
from preprocessing import *
from filters import *
import numpy as np
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
from skimage import io, img_as_float32

def parse_args():
    parser = argparse.ArgumentParser(
        description="Facial Filters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--train_keypoints',
        action='store_true',
        help='''Decide whether to train model before running'''
    )

    parser.add_argument(
        '--keypoint_data',
        required= False,
        default= '../data/training.csv',
        help='''Path to data'''
    )

    parser.add_argument(
        '--filter',
        required= False,
        default= '../data/red_circle.png',
        help='''Filter image file path'''
    )

    return parser.parse_args()

def paint_cross(x, y, img):
    x = x*48 + 48
    y = y*48 + 48
    img[int(y)][int(x)] = 1
    pass

if __name__ == "__main__":
    ARGS = parse_args()
    X_train, y_train, X_test, y_test = load_data_facial_keypoints('../data/training.csv')
    if ARGS.filter is not None:
        img_filter = img_as_float32(io.imread(ARGS.filter))
    else:
        img_filter = img_as_float32(io.imread('../data/red_circle.png'))

    if ARGS.train_keypoints:
        train_facial_keypoints(X_train, y_train)

    model = tf.keras.models.load_model('facial_keypoints_model.h5')
    keypoints = model.predict(X_train[0:5])
    testimgs = np.squeeze(X_train[0:5])

    for i in range(5):
        for j in range(0, 30, 2):
            paint_cross(keypoints[i][j], keypoints[i][j+1], testimgs[i])
    for i in range(5):
        plt.imshow(testimgs[i])
        plt.show()
        apply_red_circle(X_train[i], img_filter, model)
    c = cv2.VideoCapture(0)

    while True:
        (_, frame) = c.read()
        frame = cv2.flip(frame,1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow("test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break