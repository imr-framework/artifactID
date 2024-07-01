from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from common_utils import data_loader


def main(path_noartifact: Path, path_gibbs: Path):
    line = 90

    noartifact = data_loader.load_data(
        path_noartifact, data_format="npy", normalize=True
    )

    gibbs = np.fft.fftshift(np.fft.fft2(noartifact))
    gibbs[:, :64:2] = 0
    gibbs[:, 192::2] = 0
    gibbs = np.fft.ifft2(np.fft.ifftshift(gibbs))
    gibbs = np.abs(gibbs)
    res = gibbs - noartifact
    gibbs += res * 0

    #
    lip_noartifact = noartifact[line]
    lip_noartifact1 = np.exp(lip_noartifact[25:50])
    lip_gibbs = gibbs[line]
    lip_gibbs1 = np.exp(lip_gibbs[25:50])
    #

    autocorr_noartifact = np.correlate(lip_noartifact1, lip_noartifact1, mode="full")
    autocorr_gibbs = np.correlate(lip_gibbs1, lip_gibbs1, mode="full")

    plt.subplot(121)
    plt.imshow(noartifact, cmap="gray")
    plt.title("No artifact")
    plt.subplot(122)
    plt.imshow(gibbs, cmap="gray")
    plt.title("Gibbs Ringing")
    plt.axhline(line)

    plt.figure()
    plt.subplot(121)
    plt.plot(lip_noartifact1, 'b', label="No artifact")
    plt.plot(lip_gibbs1, 'r', label="Gibbs ringing")
    plt.grid()
    plt.subplot(122)
    plt.plot(autocorr_noartifact, 'b', label="No artifact")
    plt.plot(autocorr_gibbs, 'r', label="Gibbs ringing")
    plt.grid()
    plt.legend()
    plt.title("STFT")

    plt.show()


if __name__ == "__main__":
    img = 45
    path_noartifact = Path(
        rf"E:\Data\Datagen\ArtifactID\IXI-Gibbs_sag\noartifact\1\{img}.npy"
    )
    path_gibbs = Path(rf"E:\Data\Datagen\ArtifactID\IXI-Gibbs_sag\gibbs\1\{img}.npy")
    main(path_noartifact, path_gibbs)
