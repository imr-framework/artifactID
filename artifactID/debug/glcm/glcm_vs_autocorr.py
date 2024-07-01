from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from numpy import correlate
from scipy.signal import correlate2d
from skimage.feature import greycomatrix, greycoprops

from common_utils import data_loader


def get_normalized_glcm(img: np.ndarray, prop: str):
    glcm = greycomatrix(
        image=img,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, np.pi / 4 * 3],
        symmetric=True,
    )

    all_glcm = []
    for i in range(glcm.shape[-1]):
        g = glcm[..., i]
        g = np.log(g.squeeze())
        g[g == -np.inf] = 0
        all_glcm.append(g)

    return all_glcm, greycoprops(glcm, prop=prop)


def main(path_noartifact: Path, path_gibbs: Path):
    noartifact_orig = data_loader.load_data(
        path_noartifact, data_format="npy", normalize=True
    )
    noartifact = noartifact_orig * 255
    noartifact = noartifact.astype(np.uint8)

    gibbs = np.fft.fftshift(np.fft.fft2(noartifact_orig))
    gibbs[:, :64:2] = 0
    gibbs[:, 192::2] = 0
    gibbs = np.fft.ifft2(np.fft.ifftshift(gibbs))
    gibbs = np.abs(gibbs)
    res = gibbs - noartifact_orig
    gibbs += res * 0
    gibbs *= 255
    gibbs = gibbs.astype(np.uint8)
    # =========
    img = gibbs

    line = 90
    autocorr_noartifact = correlate(noartifact[line], noartifact[line], mode="full")
    autocorr_gibbs = correlate(gibbs[line], gibbs[line], mode="full")
    plt.plot(autocorr_noartifact)
    plt.plot(autocorr_gibbs)
    plt.title("2D autocorrelation")
    plt.show()

    plt.imshow(img, cmap="gray")
    plt.axhline(128)
    plt.axvline(128)

    p1 = img[:128, :128]
    p2 = img[:128, 128:]
    p3 = img[128:, :128]
    p4 = img[128:, 128:]

    prop = "correlation"
    for counter, p in enumerate((p1, p2, p3, p4)):
        all_glcm, features = get_normalized_glcm(p, prop=prop)
        fig, ax = plt.subplots(nrows=3, ncols=2)
        ax = ax.flatten()

        ax[0].imshow(p, cmap='gray')
        ax[0].axis('off')
        ax[1].axis('off')

        for i in range(len(ax) - 2):
            subplot = ax[i + 2]
            glcm = all_glcm[i]
            subplot.imshow(glcm, cmap="jet")
            subplot.set_title(f"{features[0, i]:.3f}")
            subplot.axis('off')
        fig.suptitle(counter + 1)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    img = 45
    path_noartifact = Path(
        rf"E:\Data\Datagen\ArtifactID\IXI-Gibbs_sag\noartifact\1\{img}.npy"
    )
    path_gibbs = Path(rf"E:\Data\Datagen\ArtifactID\IXI-Gibbs_sag\gibbs\1\{img}.npy")
    main(path_noartifact, path_gibbs)
