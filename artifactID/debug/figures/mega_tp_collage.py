from pathlib import Path

import numpy as np
from skimage.io import imread, imsave


def main(path_tp_tn_fp_fn: Path):
    path_read = path_tp_tn_fp_fn / "tp"
    if path_read.exists():
        print(path_read)

        files = list(path_read.glob("*.png"))
        images = []
        for counter in range(len(files)):
            f = files[counter]
            img = imread(str(f))
            images.append(img)
        collage = np.array(images)
        collage = np.split(collage, 4)
        collage = [np.hstack(row) for row in collage]
        collage = np.vstack(collage)

        imsave(arr=collage, fname=str(path_read / "mega_collage.png"))

        # Debug - visualize
        # from matplotlib import pyplot
        #
        # plt.imshow(collage)
        # plt.show()


if __name__ == "__main__":
    path_tp_tn_fp_fn = Path(r"E:\Data\Datagen\ArtifactID\IXI-Gibbs_sag")
    main(path_tp_tn_fp_fn)
