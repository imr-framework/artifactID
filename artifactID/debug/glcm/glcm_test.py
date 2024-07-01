from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix, greycoprops

import sass


def main(path_read_npy1: Path, path_read_npy2: Path, viz_3D=False):
    files1 = list(path_read_npy1.glob("**/*.npy"))
    files2 = list(path_read_npy2.glob("**/*.npy"))

    arr1 = []
    arr2 = []
    for i in range(len(files1)):
        npy1 = np.load(str(files1[i]))
        npy1 = npy1.astype(np.float).squeeze()[80:112, 55:87]

        npy2 = np.load(str(files2[i]))
        npy2 = npy2.astype(np.float).squeeze()[80:112, 55:87]

        npy1 *= 255
        npy1 = npy1.astype(np.uint8)
        npy2 *= 255
        npy2 = npy2.astype(np.uint8)

        d = 1
        p = "ASM"

        glcm1 = greycomatrix(
            image=npy1, distances=[d], angles=[0, np.pi / 4], symmetric=True
        )
        glcm11 = glcm1[..., 0]
        prop11 = greycoprops(glcm1, prop=p)
        glcm11 = np.log(glcm11.squeeze())
        glcm11[glcm11 == -np.inf] = 0

        glcm12 = glcm1[..., 1].squeeze()
        prop12 = greycoprops(glcm1, prop=p)
        glcm12 = np.log(glcm12.squeeze())
        glcm12[glcm12 == -np.inf] = 0

        glcm2 = greycomatrix(
            image=npy2, distances=[d], angles=[0, np.pi / 4], symmetric=True
        )
        glcm21 = glcm2[..., 0].squeeze()
        prop21 = greycoprops(glcm2, prop=p)
        glcm21 = np.log(glcm21.squeeze())
        glcm21[glcm21 == -np.inf] = 0

        glcm22 = glcm2[..., 1].squeeze()
        prop22 = greycoprops(glcm2, prop=p)
        glcm22 = np.log(glcm22.squeeze())
        glcm22[glcm22 == -np.inf] = 0

        if viz_3D:
            arr1.append(npy1)
            arr2.append(npy2)
        else:
            ax = plt.subplot(131)
            plt.imshow(npy1, cmap="gray")
            plt.axis("off")
            plt.subplot(132, sharex=ax, sharey=ax)
            plt.imshow(npy2, cmap="gray")
            plt.axis("off")
            plt.subplot(133, sharex=ax, sharey=ax)
            plt.imshow(npy2 - npy1, cmap="gray")
            plt.axis("off")

            plt.figure()
            ax = plt.subplot(221)
            plt.imshow(glcm11, cmap="jet")
            plt.title(prop11)
            plt.subplot(222, sharex=ax, sharey=ax)
            plt.imshow(glcm21, cmap="jet")
            plt.title(prop12)

            ax = plt.subplot(223)
            plt.imshow(glcm12, cmap="jet")
            plt.title(prop21)
            plt.subplot(224, sharex=ax, sharey=ax)
            plt.imshow(glcm22, cmap="jet")
            plt.title(prop22)
            plt.show()

    if viz_3D:
        arr1 = np.stack(arr1, axis=-1)
        arr2 = np.stack(arr2, axis=-1)
        diff = arr1 - arr2
        sass.scroll(arr1, arr2, diff)


if __name__ == "__main__":
    path_read_npy1 = Path(r"E:\Data\Datagen\ArtifactID\IXI-Gibbs_sag\noartifact")
    path_read_npy2 = Path(r"E:\Data\Datagen\ArtifactID\IXI-Gibbs_sag\gibbs")
    main(path_read_npy1, path_read_npy2, viz_3D=False)
