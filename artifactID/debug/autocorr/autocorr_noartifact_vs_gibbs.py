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

    plt.subplot(121)
    plt.imshow(noartifact, cmap="gray")
    plt.title("No artifact")
    plt.subplot(122)
    plt.imshow(gibbs, cmap="gray")
    plt.title("Gibbs Ringing")
    plt.axhline(line)

    plt.figure()
    plt.subplot(211)
    plt.plot(noartifact[line], "b", label="No artifact")
    plt.plot(gibbs[line], "r", label="Gibbs Ringing")
    plt.grid()
    plt.legend()
    plt.title("LIP")

    plt.subplot(212)
    corr_noartifact = np.correlate(noartifact[line], noartifact[line], mode="full")
    # corr_noartifact = corr_noartifact[corr_noartifact.size // 2:]
    plt.plot(corr_noartifact, "b", label="No artifact")
    corr_gibbs = np.correlate(gibbs[line], gibbs[line], mode="full")
    # corr_gibbs = corr_gibbs[corr_gibbs.size // 2:]
    plt.plot(corr_gibbs, "r", label="Gibbs Ringing")
    plt.grid()
    plt.legend()
    plt.title("Autocorrelation")

    plt.show()


if __name__ == "__main__":
    img = 45
    path_noartifact = Path(
        rf"E:\Data\Datagen\ArtifactID\IXI-Gibbs_sag\noartifact\1\{img}.npy"
    )
    path_gibbs = Path(rf"E:\Data\Datagen\ArtifactID\IXI-Gibbs_sag\gibbs\1\{img}.npy")
    main(path_noartifact, path_gibbs)
