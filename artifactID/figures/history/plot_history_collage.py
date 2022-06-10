import pickle

import numpy as np
from matplotlib import pyplot as plt


def main():
    """
    Plot training accuracy, validation accuracy and loss versus epochs.

    Parameters
    ==========
    path_pickle : str
        Path to training history pickle.
    """
    path_wa_xy = r"C:\Users\sravan953\Downloads\history_inplane.npy"
    path_wa_z = r"C:\Users\sravan953\Downloads\history_throughplane.npy"
    path_gibbs_sim = r"C:\Users\sravan953\Documents\CU\Projects\imr-framework\ArtifactID\Code\artifactID-MICCAI_2021\artifactID\output\20210301_2022_Gibbs_sim\history"
    path_gibbs_nosim = r"C:\Users\sravan953\Documents\CU\Projects\imr-framework\ArtifactID\Code\artifactID-MICCAI_2021\artifactID\output\20210301_2029_Gibbs_nosim\history"

    with open(path_gibbs_sim, 'rb') as pkl:
        h_gibbs_sim = pickle.load(pkl)
    with open(path_gibbs_nosim, 'rb') as pkl:
        h_gibbs_nosim = pickle.load(pkl)
    h_wa_xy = np.load(path_wa_xy, allow_pickle=True).tolist()
    h_wa_z = np.load(path_wa_z, allow_pickle=True).tolist()

    # Plot
    plt.figure()
    plt.subplot(221)
    loss_wa_xy = h_wa_xy['loss']
    loss_wa_z = h_wa_z['loss']
    plt.plot(range(1, len(loss_wa_xy) + 1), loss_wa_xy, 'b', label='In-plane wrap-around')
    plt.plot(range(1, len(loss_wa_z) + 1), loss_wa_z, 'skyblue', label='Through-plane wrap-around')
    plt.xticks(range(1, len(loss_wa_xy) + 1))
    plt.ylabel('Loss')
    plt.title('Loss vs epochs')
    plt.grid()
    plt.legend()

    plt.subplot(222)
    acc_wa_xy = h_wa_xy['accuracy']
    acc2_wa_z = h_wa_z['accuracy']
    plt.plot(range(1, len(acc_wa_xy) + 1), acc_wa_xy, 'b', label='In-plane wrap-around')
    plt.plot(range(1, len(acc2_wa_z) + 1), acc2_wa_z, 'skyblue', label='Through-plane wrap-around')
    plt.xticks(range(1, len(loss_wa_xy) + 1))
    plt.ylabel('Accuracy')
    plt.title('Training accuracy vs epochs')
    plt.grid()
    plt.legend()

    # Gibbs
    plt.subplot(223)
    loss_gibbs_sim = h_gibbs_sim['loss']
    loss_gibbs_nosim = h_gibbs_nosim['loss']
    plt.plot(range(1, len(loss_gibbs_sim) + 1), loss_gibbs_sim, 'mediumseagreen', label='Low-field + sim')
    plt.plot(range(1, len(loss_gibbs_nosim) + 1), loss_gibbs_nosim, 'darkseagreen', label='Low-field only')
    plt.xticks([1, 25, 50, 75, 100])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()

    plt.subplot(224)
    acc_gibbs_sim = h_gibbs_sim['accuracy']
    acc_gibbs_nosim = h_gibbs_nosim['accuracy']
    plt.plot(range(1, len(acc_gibbs_sim) + 1), acc_gibbs_sim, 'mediumseagreen', label='Low-field + sim')
    plt.plot(range(1, len(acc_gibbs_nosim) + 1), acc_gibbs_nosim, 'darkseagreen', label='Low-field only')
    plt.xticks([1, 25, 50, 75, 100])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
