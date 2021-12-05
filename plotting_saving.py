import matplotlib.pyplot as plt
import numpy as np
import os

BLOCK = False
FIGURES_DIRECTORY = f"figures"

if FIGURES_DIRECTORY not in os.listdir():
    os.mkdir(FIGURES_DIRECTORY)


def plotting_images(images: np.ndarray, labels: np.ndarray, title=f"Title") -> None:

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 5))
    fig.suptitle(title)

    numbers = [number for number in np.arange(10)]

    for i, image in enumerate(images):

        image_label = np.argmax(labels[i])

        if image_label not in numbers:
            continue

        else:
            numbers.remove(image_label)

            if 0 <= image_label <= 4:
                row = 0
                col = image_label
            else:
                row = 1
                col = image_label - 5

            axes[row][col].imshow(np.squeeze(images[i, :, :]))
            axes[row][col].set_title(f"Class {image_label}")

        if not numbers:
            break

    plt.show(block=BLOCK)

    filepath_figure = os.path.join(FIGURES_DIRECTORY, title)
    plt.savefig(filepath_figure)
    plt.close()


def plotting_loss(history, title=f"Training and Validation"):

    # Plot loss vs epochs
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.suptitle(title)

    axes[0].grid()
    axes[0].plot(history.history['loss'])
    axes[0].plot(history.history['val_loss'])
    axes[0].set_title('Model loss Evolution')
    axes[0].set_ylabel('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(['Training', 'Validation'], loc='upper right')

    # Plot accuracy vs epochs
    axes[1].grid()
    axes[1].plot(history.history['accuracy'])
    axes[1].plot(history.history['val_accuracy'])
    axes[1].set_title('Model accuracy')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(['Training', 'Validation'], loc='upper left')

    plt.show(block=BLOCK)

    filepath_figure = os.path.join(FIGURES_DIRECTORY, title)
    plt.savefig(filepath_figure)
    plt.close()


def plotting_comparison(histories: list, legends: list, title=f"Comparison of Accuracies") -> None:

    """
    :param histories: list of each history of neural networks
    :param legends: list of string parameters
    :param title:
    :return:
    """

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
    fig.suptitle(title)

    for history in histories:

        axes[0].grid()
        axes[0].plot(history.history['accuracy'])
        axes[0].set_ylabel('Loss')
        axes[0].set_xlabel('Epoch')

        axes[1].grid()
        axes[1].plot(history.history['val_accuracy'])
        axes[1].set_ylabel('Accuracy')
        axes[1].set_xlabel('Epoch')

    axes[0].set_title('Model Accuracies training')
    axes[0].legend(legends, loc='upper right')

    axes[1].set_title('Model accuracies validation')
    axes[1].legend(legends, loc='upper left')

    plt.show(block=BLOCK)

    filepath_figure = os.path.join(FIGURES_DIRECTORY, title)
    plt.savefig(filepath_figure)
    plt.close()
