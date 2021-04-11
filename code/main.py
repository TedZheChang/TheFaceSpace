import cv2
from models import *
from preprocessing import *
import numpy as np
import tensorflow as tf
import argparse

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

    return parser.parse_args()
    
if __name__ == "__main__":
    ARGS = parse_args()

    if ARGS.train_keypoints:
        X_train, y_train, X_test, y_test = load_data_facial_keypoints('../data/training.csv')
        train_facial_keypoints(X_train, y_train)

    model = tf.keras.models.load_model('facial_keypoints_model.h5')

    c = cv2.VideoCapture(0)
    
    while True:
        (_, frame) = c.read()
        frame = cv2.flip(frame,1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow("test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break