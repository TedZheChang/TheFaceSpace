import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense


def facial_keypoints_model():
    model = Sequential()

    # first conv layer
    model.add(Conv2D(32, (5, 5), input_shape=(96,96,1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    # second conv layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(.2))

    # third conv layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(.2))

    # fourth conv layer
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(.3))

    # flatten
    model.add(Flatten())

    # mlp
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    
    # there are 30 facial keypoints
    model.add(Dense(30))

    return model

def train_facial_keypoints(X,y):
    model = facial_keypoints_model()
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=["accuracy"])
    model.fit(X, y, epochs=2, batch_size=100, verbose=1, validation_split=0.2)
    model.save("facial_keypoints_model.h5")
