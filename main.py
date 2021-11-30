from preprocessing import preprocess
from neural_networks import generate_neural_network, predicting_testing
from plotting_saving import plotting_loss

import numpy as np

DATABASE_MNIST = f"MNIST_CorrNoise.npz"


def main():

    # Preprocessing step
    x_train, y_train, x_test, y_test = preprocess(DATABASE_MNIST)

    model_name = "CNN_initial"   # Neural Network 1: CNN with 1 layer
    model_name = "CNN_dense"  # Neural Network 2: CNN adding dense layers
    model_name = "CNN_pooling"  # Neural_network 3: CNN full conv 2D and max pooling

    history = generate_neural_network(x_train, y_train, model_name)

    # Plotting Evolution of the loss and accuracy
    title_figure = f"Model {model_name} Training of the neural network.png"
    plotting_loss(history, title_figure)

    # Predicting on testing data
    #predicting_testing(x_test, model_name)


if __name__ == '__main__':
    main()
