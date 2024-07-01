import pickle
from pathlib import Path

from matplotlib import pyplot as plt


def main(path_sim_history: Path, path_nosim_history: Path):
    """
    Plot training accuracy, validation accuracy and loss versus epochs.

    Parameters
    ==========
    path_pickle : str
        Path to training history pickle.
    """
    with open(path_sim_history, "rb") as pkl:  # Read training history pickle
        sim_history = pickle.load(pkl)
    with open(path_nosim_history, "rb") as pkl:  # Read training history pickle
        nosim_history = pickle.load(pkl)

    # Plot loss vs epochs
    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()

    loss = sim_history["loss"]
    val_loss = sim_history["val_loss"]
    axes[0].plot(range(1, len(loss) + 1), loss, alpha=0.5, color="blue", label="Train loss")
    axes[0].plot(range(1, len(val_loss) + 1), val_loss, "--", alpha=0.75, color="blue", label="Validation loss")
    axes[0].set_ylabel("Loss")
    axes[0].title.set_text("Loss vs epochs")
    axes[0].legend()
    axes[0].grid()

    loss = nosim_history["loss"]
    val_loss = nosim_history["val_loss"]
    axes[2].plot(range(1, len(loss) + 1), loss, alpha=0.5, color="blue", label="Train loss")
    axes[2].plot(range(1, len(val_loss) + 1), val_loss, "--", alpha=0.75, color="blue",
                 label="Validation loss")
    axes[2].set_xlabel("Epochs")
    axes[2].set_ylabel("Loss")
    axes[2].grid()

    # =========
    accuracy = sim_history["accuracy"]
    val_accuracy = sim_history["val_accuracy"]
    axes[1].plot(range(1, len(accuracy) + 1), accuracy, alpha=0.75, color="red", label="Train accuracy")
    axes[1].plot(range(1, len(val_accuracy) + 1), val_accuracy, "--", alpha=1.0, color="red", label="Val accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[1].title.set_text("Accuracy vs epochs")
    axes[1].legend()
    axes[1].grid()

    accuracy = sim_history["accuracy"]
    val_accuracy = sim_history["val_accuracy"]
    axes[3].plot(range(1, len(accuracy) + 1), accuracy, alpha=0.75, color="red", label="Train accuracy")
    axes[3].plot(range(1, len(val_accuracy) + 1), val_accuracy, "--", alpha=1.0, color="red", label="Val accuracy")
    axes[3].set_xlabel("Epochs")
    axes[3].set_ylabel("Accuracy")
    axes[3].grid()

    plt.tight_layout(w_pad=0.05)
    plt.show()


if __name__ == "__main__":
    # path_history = Path(r"models/20220902_1515_gamma1_finalpick/history")
    path_sim_history = Path(r"C:\Users\imrfr\Downloads\gr_sim")
    path_nosim_history = Path(r"C:\Users\imrfr\Downloads\gr_nosim")
    main(path_sim_history=path_sim_history, path_nosim_history=path_nosim_history)
