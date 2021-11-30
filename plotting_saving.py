import matplotlib.pyplot as plt
import numpy as np
import os

BLOCK = False
FIGURES_DIRECTORY = f"figures"

if FIGURES_DIRECTORY not in os.listdir():
    os.mkdir(FIGURES_DIRECTORY)


def plotting_image(images: np.ndarray, num_image: int, title=f"Title") -> None:

    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.imshow(np.squeeze(images[num_image, :, :]))
    ax.set_title(title)

    plt.show(block=BLOCK)

    filepath_figure = os.path.join(FIGURES_DIRECTORY, title)
    plt.savefig(filepath_figure)
    plt.close()


def plotting_loss(history, title=f"Training and Validation"):

    # Plot loss vs epochs
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.suptitle(title)

    axes[0].plot(history.history['loss'])
    axes[0].plot(history.history['val_loss'])
    axes[0].set_title('Model loss Evolution')
    axes[0].set_ylabel('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(['Training', 'Validation'], loc='upper right')

    # Plot accuracy vs epochs
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
