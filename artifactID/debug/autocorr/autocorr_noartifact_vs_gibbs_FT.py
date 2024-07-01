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
    gibbs += res * 1

    #
    # noartifact = preprocessor.mask_subject(noartifact)
    # gibbs = preprocessor.mask_subject(gibbs)
    #

    #
    # noartifact *= 100
    # gibbs *= 100
    #

    #
    lip_noartifact = noartifact[line]#[:-1] - noartifact[line][1:]
    lip_noartifact1 = lip_noartifact[:64]
    lip_noartifact2 = lip_noartifact[64:128]
    lip_noartifact3 = lip_noartifact[128:192]
    lip_noartifact4 = lip_noartifact[192:256]
    lip_gibbs = gibbs[line]#[:-1] - gibbs[line][1:]
    lip_gibbs1 = lip_gibbs[:64]
    lip_gibbs2 = lip_gibbs[64:128]
    lip_gibbs3 = lip_gibbs[128:192]
    lip_gibbs4 = lip_gibbs[192:256]
    #

    plt.subplot(121)
    plt.imshow(noartifact, cmap="gray")
    plt.title("No artifact")
    plt.subplot(122)
    plt.imshow(gibbs, cmap="gray")
    plt.title("Gibbs Ringing")
    plt.axhline(line)

    plt.figure()
    plt.subplot(221)
    plt.plot(lip_noartifact1, 'b', label="No artifact")
    plt.plot(lip_gibbs1, 'r', label="Gibbs ringing")
    plt.grid()
    plt.subplot(222)
    plt.plot(lip_noartifact2, 'b', label="No artifact")
    plt.plot(lip_gibbs1, 'r', label="Gibbs ringing")
    plt.grid()
    plt.subplot(223)
    plt.plot(lip_noartifact3, 'b', label="No artifact")
    plt.plot(lip_gibbs3, 'r', label="Gibbs ringing")
    plt.grid()
    plt.subplot(224)
    plt.plot(lip_noartifact4, 'b', label="No artifact")
    plt.plot(lip_gibbs4, 'r', label="Gibbs ringing")
    plt.grid()
    plt.legend()
    plt.title("LIP")

    plt.figure()
    plt.subplot(221)
    plt.plot(np.abs(np.fft.fftshift(np.fft.fft(lip_noartifact1))), 'b', label="No artifact")
    plt.plot(np.abs(np.fft.fftshift(np.fft.fft(lip_gibbs1))), 'r', label="Gibbs ringing")
    plt.grid()
    plt.subplot(222)
    plt.plot(np.abs(np.fft.fftshift(np.fft.fft(lip_noartifact2))), 'b', label="No artifact")
    plt.plot(np.abs(np.fft.fftshift(np.fft.fft(lip_gibbs2))), 'r', label="Gibbs ringing")
    sum1 = np.sum(np.abs(np.fft.fftshift(np.fft.fft(lip_noartifact2)))[48:])
    sum2 = np.sum(np.abs(np.fft.fftshift(np.fft.fft(lip_gibbs2)))[48:])
    plt.title(f"{sum1} {sum2}")
    plt.grid()
    plt.subplot(223)
    plt.plot(np.abs(np.fft.fftshift(np.fft.fft(lip_noartifact3))), 'b', label="No artifact")
    plt.plot(np.abs(np.fft.fftshift(np.fft.fft(lip_gibbs3))), 'r', label="Gibbs ringing")
    sum1 = np.sum(np.abs(np.fft.fftshift(np.fft.fft(lip_noartifact3)))[48:])
    sum2 = np.sum(np.abs(np.fft.fftshift(np.fft.fft(lip_gibbs3)))[48:])
    plt.title(f"{sum1} {sum2}")
    plt.grid()
    plt.subplot(224)
    plt.plot(np.abs(np.fft.fftshift(np.fft.fft(lip_noartifact4))), 'b', label="No artifact")
    plt.plot(np.abs(np.fft.fftshift(np.fft.fft(lip_gibbs4))), 'r', label="Gibbs ringing")
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
