import pickle
from pathlib import Path

from matplotlib import pyplot as plt


def main(path_pickle: str):
    """
    Plot training accuracy, validation accuracy and loss versus epochs.

    Parameters
    ==========
    path_pickle : str
        Path to training history pickle.
    """
    with open(path_pickle, 'rb') as pkl:  # Read training history pickle
        history = pickle.load(pkl)

    # Plot
    plt.figure()
    loss = history['loss']
    plt.plot(range(1, len(loss) + 1), loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(range(len(loss)))
    plt.title('Loss vs epochs')
    plt.grid()

    plt.figure()
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    plt.plot(range(1, len(acc) + 1), acc, label='Train accuracy')
    plt.plot(range(1, len(val_acc) + 1), val_acc, label='Val accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xticks(range(len(acc)))
    plt.title('Training/validation accuracy vs epochs')

    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    path_history = Path('output/20211101_1905_motion_sagittal') / 'history'
    main(path_pickle=str(path_history))
