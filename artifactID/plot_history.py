import configparser
import pickle

from matplotlib import pyplot as plt


def main(path_history: str):
    """
    Plot training accuracy, validation accuracy and loss versus epochs in separate windows.

    Parameters
    ==========
    path_history : str
        Path to pickle containing training history.
    """
    with open(path_history, 'rb') as pkl:  # Read history pickle
        history = pickle.load(pkl)

    # Plot
    plt.figure()
    loss = history['loss']
    plt.plot(range(1, len(loss) + 1), loss)
    plt.plot(range(1, len(loss) + 1), loss, '.')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs epochs')

    plt.figure()
    loss = history['accuracy']
    plt.plot(range(1, len(loss) + 1), loss)
    plt.plot(range(1, len(loss) + 1), loss, '.')
    plt.xlabel('Epochs')
    plt.ylabel('Training accuracy')
    plt.title('Training accuracy vs epochs')

    # plt.figure()
    # loss = history['val_loss']
    # plt.plot(range(1, len(loss) + 1), loss)
    # plt.plot(range(1, len(loss) + 1), loss, '.')
    # plt.xlabel('Epochs')
    # plt.ylabel('Validation accuracy')
    # plt.title('Validation accuracy vs epochs')

    plt.show()


if __name__ == '__main__':
    # Read settings.ini configuration file
    path_settings = 'settings.ini'
    config = configparser.ConfigParser()
    config.read(path_settings)

    config_plot = config['PLOT HISTORY']
    path_history = config_plot['path_history']
    main(path_history=path_history)
