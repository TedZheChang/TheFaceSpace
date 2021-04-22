import pandas
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf


def load_data_facial_keypoints(path, testing=False):
    # load data into dataframe
    df = pandas.read_csv(path)

    # turn images in numpy arrays
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    # drop all missing values
    df = df.dropna()

    # get all the images and their keypoint labels
    X = np.vstack(df['Image'].values).astype(np.float32).reshape((-1, 96, 96, 1))
    y = df[df.columns[:-1]].values.astype(np.float32)

    # normalize the datapoints
    X = X / 255.
    y = (y - 48) / 48

    if testing:
        # split the data into 80% training 20% testing
        train_size = int(len(X)*.8)
        X_train,y_train = X[:train_size], y[:train_size]
        X_test,y_test = X[train_size:], y[:train_size:]
        return X_train, y_train, X_test, y_test
    else:
        return X,y

def load_data_facial_expressions(path):
    # load data into pandas
    df = pandas.read_csv(path)

    # turn pixels in np arrays
    df['pixels'] = df['pixels'].apply(lambda im: np.fromstring(im, sep=' '))

    # drop undefined rows
    df = df.dropna()

    # resize pixels to be more consistent with facial landmark model
    df['pixels'] = df['pixels'].apply(lambda x: cv2.resize(x, (96,96)))

    # normalize pixels and stack
    X = np.vstack(df['pixels'].values).astype(np.float32).reshape((-1, 96, 96, 1))/255
    y = np.array(df['emotion'])
    print('x shape = ', X.shape)
    print('y shape = ', y.shape)
    return X[0:2000,:,:,:],y[0:2000]
    
def load_raw_keypoints(path):
    # load data into dataframe
    df = pandas.read_csv(path)

    # turn images in numpy arrays
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    # drop all missing values
    df = df.dropna()

    # get all the images and their keypoint labels
    X = np.vstack(df['Image'].values).astype(np.float32).reshape((-1, 96, 96, 1))
    y = df[df.columns[:-1]].values.astype(np.float32)

    return X, y

# load_data_facial_expressions('../data/facial_expression_data.csv')

