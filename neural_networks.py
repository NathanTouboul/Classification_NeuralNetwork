import os
import numpy as np
import pickle

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adadelta, Adam, SGD
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, Dropout, Flatten, AveragePooling2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import load_model


"""
Parameters to test 
    - Type of layers: Conv2D, MaxPooling2D...
    - Size of kernel for convolution
    - number of hidden layers 
    - optimizer

"""


def generate_neural_network(x_train, y_train, model_name):

    p_weight = f"./weights/weights_{model_name}.hdf5"

    if not os.path.exists('./weights'):
        os.mkdir('./weights')

    # EXPLORE VALUES AND FIND A GOOD SET
    batch_size = 100  # batch size
    val_split = 0.5  # percentage of samples used for validation (e.g. 0.5)
    ep = 10  # number of epochs

    num_cls = len(np.unique(y_train, axis=0))

    # Input shape of the neural network
    input_shape = x_train.shape[1:4]    # (28,28,1)

    # Creating Sequential Model
    model = Sequential()

    if model_name == "CNN_initial":

        # Input Convolutional 2D layer
        model.add(Conv2D(1, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        # Transforms matrix feature map to vector for dense layer (fully connected)
        model.add(Flatten())

        # Adding output layer
        model.add(Dense(num_cls, activation='softmax'))

    elif model_name == "CNN_dense":

        # Adding dense layer for additional parameters
        # Input Convolutional 2D layer
        n_filters = 5
        model.add(Conv2D(n_filters, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

        # Transforms matrix feature map to vector for dense layer (fully connected)
        model.add(Flatten())

        for _ in range(3):
            dimension = n_filters * 26 ** 2
            model.add(Dense(dimension, activation='relu'))
            model.add(Dropout(0.2))

        # Adding output layer
        model.add(Dense(num_cls, activation='softmax'))

    elif model_name == "CNN_pooling":

        # Using max pooling to reduce resolution of the output of the convolution to reduce the number of parameters and
        # so cost of computation, also it is a way to extract particular features such as edges, curves, circles..

        n_filters = 100
        model.add(Conv2D(n_filters, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2), padding="valid"))
        model.add(Conv2D(n_filters, kernel_size=(3, 3), activation='relu', padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2), padding="valid"))
        model.add(Conv2DTranspose(n_filters, kernel_size=(3, 3), activation='relu', padding="same"))

        # Transforms matrix feature map to vector for dense layer (fully connected)
        model.add(Flatten())

        # Adding output layer
        model.add(Dense(num_cls, activation='softmax'))

    # Compiling
    model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

    # Summary of the model
    model.summary()
    
    checkpoints = ModelCheckpoint(filepath=p_weight, verbose=1, save_best_only=True)
    callbacks_list = [checkpoints]

    print(f"Shape of x_train: {str(x_train.shape)}")
    print(f"Shape of y_train: {str(y_train.shape)}")

    history = model.fit(x_train, y_train, epochs=ep, batch_size=batch_size, verbose=1, shuffle=True,
                        validation_split=val_split, callbacks=callbacks_list)
    
    print('CNN weights saved in ' + p_weight)

    return history


def predicting_testing(x_test, model_name):

    print('Shape of testing dataset: ' + str(x_test.shape) + '\n')

    # Define model parameters
    p_weight = './weights/weights_' + model_name + '.hdf5'

    model = load_model(p_weight)
    y_prediction = model.predict(x_test)

    y = np.amax(y_prediction, axis=1)

    # predictions_accuracies = np.sum() / len(y_test)

    # print('Accuracy in test set is: '+ vstr(Acc_pred))
