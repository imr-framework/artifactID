import pickle
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def main(path_pickle: str):
    """
    Plot training accuracy, validation accuracy and loss versus epochs.

    Parameters
    ==========
    path_pickle : str
        Path to training history pickle.
    """
    with open(path_pickle, "rb") as pkl:  # Read training history pickle
        history = pickle.load(pkl)

    # Plot loss vs epochs
    fig, axes = plt.subplots(1, 2, sharex=True)
    loss = history["loss"]
    val_loss = history["val_loss"]
    axes[0].plot(range(1, len(loss) + 1), loss, ".-", label="Train loss")
    axes[0].plot(range(1, len(val_loss) + 1), val_loss, ".-", label="Validation loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].title.set_text("Training/validation loss vs epochs")
    axes[0].legend()
    axes[0].grid()

    # Plot training and validation accuracies vs epochs
    accuracy = history["accuracy"]
    val_accuracy = history["val_accuracy"]
    axes[1].plot(range(1, len(accuracy) + 1), accuracy, ".-", label="Train accuracy")
    axes[1].plot(range(1, len(val_accuracy) + 1), val_accuracy, ".-", label="Val accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].title.set_text("Training/validation accuracy vs epochs")
    axes[1].legend()
    axes[1].grid()

    print(
        f"Min loss(epoch)/ val_loss (epoch): {np.min(loss)}({np.argmin(loss) + 1})/ {np.min(val_loss)}({np.argmin(val_loss) + 1})"
    )
    print(
        f"Max accuracy(epoch)/ val_accuracy (epoch): {np.max(accuracy)}({np.argmax(accuracy) + 1})/ {np.max(val_accuracy)}({np.argmax(val_accuracy) + 1})"
    )

    plt.tight_layout(w_pad=0.05)
    plt.show()


if __name__ == "__main__":
    path_history = Path(r"models/20220902_1515_gamma1_finalpick/history")
    # path_history = Path(r"models/20240310_1541/history")
    main(path_pickle=str(path_history))
