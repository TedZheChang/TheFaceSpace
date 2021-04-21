import cv2
from models import *
from preprocessing import *
from filters import *
import numpy as np
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt

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
        '--train_expression',
        action='store_true',
        help='''Decide whether to train model before running'''
    )

    parser.add_argument(
        '--filter',
        required= False,
        default= None,
        help='''Filter image file path'''
    )

    parser.add_argument(
        '--cv2',
        required=False,
        default=True,
        help='''Whether to use cv2 library'''
    )

    return parser.parse_args()

def reverse_norm(x, y):
    x = x*48 + 48
    y = y*48 + 48
    return x, y

if __name__ == "__main__":
    # parse arguments from cli for selected options
    ARGS = parse_args()
    # load cv2 face classifier to help find faces to put filters on
    face_classifier = cv2.CascadeClassifier('../data/haar_cascade.xml')

    # read which filter to use, default to red nose filter
    if ARGS.filter is not None:
        filters = img_as_float32(io.imread(ARGS.filter))
    else:
        filters = [cv2.imread('../data/googly-eye.png', cv2.IMREAD_UNCHANGED),
                   None,
                   cv2.imread('../data/clown-nose.png', cv2.IMREAD_UNCHANGED)]

    # only train the model if specificed
    if ARGS.train_keypoints:
        # load in training data
        X_train, y_train = load_data_facial_keypoints('../data/facial_keypoints_data.csv')
        # train the model
        train_facial_keypoints(X_train, y_train)

    if ARGS.train_expression:
        # load in training data
        X_train,y_train = load_data_facial_expressions('../data/facial_expression_data.csv')
        # train the model
        train_facial_expressions(X_train, y_train)

    # load in the models
    keypoints_model = tf.keras.models.load_model('facial_keypoints_model.h5')
    expressions_model = tf.keras.models.load_model('facial_expressions_model.h5')

# always true for some reason
    if ARGS.cv2:
        # use cv2 to capture current video feed
        c = cv2.VideoCapture(0)

        while True:
            # get the frame and flip for aesthetic purposes
            (_, frame) = c.read()
            frame = cv2.flip(frame,1)
            # grayscale version because that's what the model is trained on
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray)

            for f in faces:
                # get the x,y coordinates and height and width of detected face
                x, y, w, h = f[0], f[1], f[2], f[3]
                # retrieve face from the fram
                face = gray[y:y+h, x:x+h]
                # normalize and resize
                face = np.reshape(cv2.resize(face/255, (96,96)), (1,96,96,1))
                # predict keypoints and rescale
                p = keypoints_model.predict(face) * 48 + 48
                # map keypoints to coordinates
                keypoints = []
                for i in range(0,30,2):
                    keypoints.append((p[0][i+1],p[0][i]))
                # apply filters
                colored_face = cv2.resize(frame[y:y+h, x:x+h], (96,96))
                filtered_face = apply_filters(colored_face, filters, keypoints)
                frame[y:y+h, x:x+h] = cv2.resize(filtered_face, (h,w))

            cv2.imshow("Face Space", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        print("1")
        X, Y = load_raw_keypoints('../data/facial_keypoints_data.csv')
        print("2")
        spot = np.zeros((1, 1, 3))
        for k in range(len(X)):
            frame = X[k]
            gray = frame
            f = [0, 0, 96, 96]
            x, y, w, h = f[0], f[1], f[2], f[3]
            face = gray[y:y + h, x:x + h]
            face = np.reshape(face, (1, 96, 96, 1))
            p = keypoints_model.predict(face/255) * 48 + 48
            keypoints = []
            for i in range(0, 30, 2):
                keypoints.append((p[0][i + 1], p[0][i]))
            # apply filters
            colored_face = cv2.resize(frame[y:y + h, x:x + h], (96, 96))
            colored_face = np.squeeze(colored_face)
            c_face = np.zeros((len(colored_face), len(colored_face[0]), 3))
            for i in range(len(colored_face)):
                for j in range(len(colored_face[0])):
                    c_face[i][j][0] = colored_face[i][j]
                    c_face[i][j][1] = colored_face[i][j]
                    c_face[i][j][2] = colored_face[i][j]
            # filters = [spot, spot, spot]
            # black = np.zeros(3)
            # for point in keypoints:
            #     c_face[int(point[0])][int(point[1])] = black
            filtered_face = apply_filters(c_face, filters, keypoints)
            plt.imshow(filtered_face[:, :, 0])
            plt.show()




