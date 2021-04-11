import pandas
import numpy as np

def load_data_facial_keypoints(path, testing=False):
    df = pandas.read_csv(path)
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    df.dropna()

    X = np.vstack(df['Image'].values) / 255.
    X = X.astype(np.float32)
    X = X.reshape(-1, 96, 96, 1)

    if not testing:
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

load_data_facial_keypoints('../data/training.csv')