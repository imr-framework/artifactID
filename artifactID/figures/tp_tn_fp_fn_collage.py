from pathlib import Path

import numpy as np
from skimage.io import imread, imsave


def main(path_tp_tn_fp_fn: Path):
    for folder in ["tp", "tn", "fp", "fn"]:
        path_read = path_tp_tn_fp_fn / folder
        if path_read.exists():
            print(path_read)

            files = list(path_read.glob("*.png"))
            images = []
            for f in files:
                img = imread(str(f))
                if folder in ["fp", "tn"]:
                    images.append(np.ones((img.shape[0], img.shape[1], 4)))

                images.append(img)
            collage = np.hstack(images)

            imsave(arr=collage, fname=str(path_read / "collage.png"))

            # Debug - visualize
            # from matplotlib import pyplot
            #
            # plt.imshow(collage)
            # plt.show()


if __name__ == "__main__":
    path_tp_tn_fp_fn = Path(r"D:\Sravan\Data\Datagen\ArtifactID_v1\IXI-T1")
    main(path_tp_tn_fp_fn)
