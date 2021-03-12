import configparser
import pickle

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
    plt.title('Loss vs epochs')
    plt.grid()

    plt.figure()
    acc = history['accuracy']
    plt.plot(range(1, len(acc) + 1), acc, label='Train')
    val_acc = history['val_accuracy']
    plt.plot(range(1, len(val_acc) + 1), val_acc, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training/validation accuracy vs epochs')

    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # Read settings.ini configuration file
    path_settings = 'settings.ini'
    config = configparser.ConfigParser()
    config.read(path_settings)

    config_plot = config['PLOT HISTORY']
    path_history = config_plot['path_history']
    main(path_pickle=path_history)
