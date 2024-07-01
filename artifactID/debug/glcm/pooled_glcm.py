from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from tqdm import tqdm


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
    texture_means = []
    for path_data in (path_noartifact, path_gibbs):
        files = list(path_data.glob("**/*.npy"))

        all_textures = []
        for f in tqdm(files):
            npy = np.load(str(f))
            npy = npy * 255
            npy = npy.astype(np.uint8)

            prop = "correlation"
            _, features = get_normalized_glcm(npy, prop=prop)
            all_textures.append(features.squeeze())

        all_textures = np.stack(all_textures)
        texture_means.append(np.mean(all_textures, axis=0))

    angles = [0, 45, 90, 135]
    plt.plot(angles, texture_means[0], 'b', label="No artifact")
    plt.plot(angles, texture_means[1], 'r', label="Gibbs Ringing")
    plt.xticks(ticks=angles, labels=[str(n) for n in angles])
    plt.xlabel("Angle (degrees)")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    path_noartifact = Path(r"E:\Data\Datagen\ArtifactID\IXI-Gibbs_sag\noartifact")
    path_gibbs = path_noartifact.with_name("gibbs")
    main(path_noartifact, path_gibbs)
