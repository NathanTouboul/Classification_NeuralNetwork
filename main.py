from preprocessing import preprocess
from neural_networks import generate_neural_network, predicting_testing
from plotting_saving import plotting_images, plotting_loss, plotting_comparison

import numpy as np

DATABASE_MNIST = f"MNIST_CorrNoise.npz"


def main():

    # Preprocessing step
    x_train, y_train, x_test, y_test = preprocess(DATABASE_MNIST)

    # Plotting Examples
    plotting_images(x_train, y_train, title=f"Noisy Handwritten digits")

    # Choice of Neural Network to study
    model_name = "CNN_initial"   # Neural Network 1: CNN with 1 layer
    model_name = "CNN_add_dense"  # Neural Network 2: CNN adding dense layers
    model_name = "full_dense"  # Neural Network 3: CNN adding dense layers
    model_name = "CNN_pooling"  # Neural_network 4: CNN full conv 2D and max pooling

    history = generate_neural_network(x_train, y_train, model_name)

    # Plotting Evolution of the loss and accuracy
    title_figure = f"Training model {model_name}"
    plotting_loss(history, title_figure)

    # Predicting on testing data
    predicting_testing(x_test, y_test, model_name)

    # Impact of number of filters
    filters = [5, 10, 15]
    histories = []

    for n in filters:
        histories.append(generate_neural_network(x_train, y_train, model_name, n_filters=n, patience=1))

    title_comparison = f"Comparison for the number of filters"
    plotting_comparison(histories, filters, title=title_comparison)

    # Impact of optimizer
    optimizers = ["Adadelta", "Adam", "SGD"]
    histories = []

    for optimizer in optimizers:
        histories.append(generate_neural_network(x_train, y_train, model_name, optimizer=optimizer, patience=1))

    title_comparison = f"Comparison for the choice of optimizer"
    plotting_comparison(histories, optimizers, title=title_comparison)

    # Impact of batch size
    batch_sizes = [10, 100, 1000]
    histories = []

    for batch_size in batch_sizes:
        histories.append(generate_neural_network(x_train, y_train, model_name, batch_size=batch_size, patience=1))

    title_comparison = f"Comparison for the size of the batch"
    plotting_comparison(histories, batch_sizes, title=title_comparison)


if __name__ == '__main__':
    main()

