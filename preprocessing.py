import os
import numpy as np
import time
from tensorflow.keras.utils import to_categorical

from plotting_saving import plotting_images

DATABASE_DIRECTORY = f"dataset"

if DATABASE_DIRECTORY not in os.listdir():
    os.mkdir(DATABASE_DIRECTORY)


def preprocess(filename: str):

    filepath_mnist = os.path.join(DATABASE_DIRECTORY, filename)

    start = time.perf_counter()

    with np.load(filepath_mnist) as data:
        x_train = data['x_train']
        y_train = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']

    end = time.perf_counter()

    print(f"Loading from MNIST time {end - start}")

    num_cls = len(np.unique(y_train))
    print('Number of classes: ' + str(num_cls))

    # Reshape and standardize
    x_train = np.expand_dims(x_train / 255, axis=3)
    x_test = np.expand_dims(x_test / 255, axis=3)

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_cls)

    print(f"Shape of x_train: {str(x_train.shape)}")
    print(f"Shape of y_train: {str(y_train.shape)}")
    print(f"Shape of x_test: {str(x_test.shape)}")
    print(f"Shape of y_test`: {str(y_test.shape)}")

    return x_train, y_train, x_test, y_test
