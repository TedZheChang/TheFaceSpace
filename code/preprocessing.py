import pandas
import numpy as np
import matplotlib.pyplot as plt


def load_data_facial_keypoints(path):
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

    # split the data into 80% training 20% testing
    train_size = int(len(X)*.8)
    X_train,y_train = X, y
    X_test,y_test = X[train_size:], y[:train_size:]

    return X_train, y_train, X_test, y_test

