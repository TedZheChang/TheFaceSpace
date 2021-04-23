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
from skimage import io, img_as_float32

def parse_args():
    parser = argparse.ArgumentParser(
        description="Facial Filters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--train_keypoints',
        default=True,
        required=False,
        help='''Decide whether to train model before running'''
    )

    parser.add_argument(
        '--train_expression',
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
        '--cv2',
        required=False,
        default='True',
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
    
    # only train the model if specificed
    if ARGS.train_keypoints:
        # load in training data
        X_train, y_train = load_data_facial_keypoints('../data/facial_keypoints_data.csv')
        # train the model
        train_facial_keypoints(X_train, y_train)
    keypoints_model = tf.keras.models.load_model('facial_keypoints_model.h5')
   
    if ARGS.train_expression:
        # load in training data
        X_train,y_train = load_data_facial_expressions('../data/facial_expression_data.csv')
        # train the model
        train_facial_expressions(X_train, y_train)
    
    expressions_model = tf.keras.models.load_model('expressions_model.h5')

# always true for some reason
    if ARGS.cv2 == 'True':
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
                # apply filters based on expressions 
                expressions_prob = expressions_model.predict(face)
                predictec_exp = np.argmax(expressions_prob)
                
                if predictec_exp==0: #angry
                    eye_filter = cv2.imread('../data/angry_eyes.jpeg', -1)
                    nose_filter = cv2.imread('../data/clown-nose.png', -1)
                    mouth_filter = None
                elif predictec_exp==1: #disgust
                    eye_filter = None
                    nose_filter = None
                    mouth_filter = cv2.imread('../data/disgusted_mouth.jepg', -1)
                elif predictec_exp==2: #fear
                    eye_filter = cv2.imread('../data/cute_eyes.jepg', -1)
                    nose_filter = None
                    mouth_filter = None
                elif predictec_exp==3: #happy
                    eye_filter = cv2.imread('../data/sunglasses.jpg', -1)
                    nose_filter = None
                    mouth_filter = None
                elif predictec_exp==4: #sad
                    eye_filter = cv2.imread('../data/cute_eyes.png', -1)
                    nose_filter = None
                    mouth_filter = None
                elif predictec_exp==5: #surprise
                    eye_filter = None
                    nose_filter = None
                    mouth_filter = cv2.imread('../data/surprised_mouth.png', -1)
                else: #neutral
                    eye_filter = None 
                    nose_filter = cv2.imread('../data/pig_nose.png', -1)
                    mouth_filter = None
                colored_face = cv2.resize(frame[y:y+h, x:x+h], (96,96))
                filtered_face = apply_filters(colored_face, eye_filter, nose_filter, mouth_filter, keypoints)
                frame[y:y+h, x:x+h] = cv2.resize(filtered_face, (h,w))

            cv2.imshow("Face Space", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        X, Y = load_raw_keypoints('../data/facial_keypoints_data.csv')
        spot = np.zeros((1, 1, 3))
        for k in range(len(X)):
            frame = X[k]
            gray = frame
            f = [0, 0, 96, 96]
            x, y, w, h = f[0], f[1], f[2], f[3]
            # retrieve face from the fram
            face = gray[y:y+h, x:x+w]
            # normalize and resize
            face = np.reshape(cv2.resize(face/255, (96,96)), (1,96,96,1))
            # predict keypoints and rescale
            p = keypoints_model.predict(face) * 48 + 48
            # map keypoints to coordinates
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
            filtered_face = apply_filters(c_face, eye_filter, nose_filter, mouth_filter, keypoints)
            plt.imshow(filtered_face[:, :, 0])
            plt.show()




