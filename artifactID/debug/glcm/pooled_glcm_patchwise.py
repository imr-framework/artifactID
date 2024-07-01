from matplotlib import pyplot as plt
from pathlib import Path

import numpy as np
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


def main(path_data: Path):
    files = list(path_data.glob("**/*.npy"))

    all_textures = []
    for f in tqdm(files):
        npy = np.load(str(f))
        npy = npy * 255
        npy = npy.astype(np.uint8)

        p1 = npy[:128, :128]
        p2 = npy[:128, 128:]
        p3 = npy[128:, :128]
        p4 = npy[128:, 128:]

        prop = "contrast"
        values = []
        for counter, p in enumerate((p1, p2, p3, p4)):
            _, features = get_normalized_glcm(p, prop=prop)
            values.append(features.squeeze())
        all_textures.append(values)

    all_textures = np.stack(all_textures)
    patchwise_means = np.mean(all_textures, axis=0)

    plt.matshow(patchwise_means)
    plt.xlabel("0 -- 45 -- 90 -- 135 (degrees)")
    plt.ylabel("Patch")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    path_data = Path(r"E:\Data\Datagen\ArtifactID\IXI-Gibbs_sag\gibbs")
    main(path_data)
