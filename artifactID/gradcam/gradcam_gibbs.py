from pathlib import Path

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from artifactID.gradcam import gradcam
from artifactID.utils import get_tv_patches, draw_labels_on_subplot, stitch_patches

# =========
# TENSORFLOW INIT
# =========
# Prevent crash
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def get_tv(img: np.ndarray) -> np.ndarray:
    return img[:-1, :-1] - img[1:, 1:]


def main(path_model: str, path_txt: str):
    path_txt = Path(path_txt)
    files = path_txt.read_text().splitlines()[1:]
    path_root = path_txt.parent

    for path_npy in files:
        path_npy = path_root / path_npy.strip()
        if 'gibbs' in path_npy.parent.name:
            npy = np.load(str(path_npy)).squeeze()
            path_npy_clean = path_npy.parent.parent / 'noartifact' / path_npy.name
            npy_clean = np.load(str(path_npy_clean)).squeeze()

            npy_patches = get_tv_patches(npy)
            npy_clean_patches = get_tv_patches(npy_clean)

            # =========
            # GRAD-CAM
            # =========
            # Perform Grad-CAM
            heatmaps_clean, y_preds_clean = gradcam.main(arr_npy=npy_clean_patches, artifact='gibbs',
                                                         path_model=path_model)
            heatmaps, y_preds = gradcam.main(arr_npy=npy_patches, artifact='gibbs',
                                             path_model=path_model)

            # Print output
            y_preds_clean = np.reshape(y_preds_clean, (4, -1))
            y_preds = np.reshape(y_preds, (4, -1))
            print(f'Pred clean: {y_preds_clean}')
            print(f'Pred: {y_preds}')

            # Plotting
            heatmap_clean = stitch_patches(heatmaps_clean)
            heatmap = stitch_patches(heatmaps)

            plt.subplot(221)
            plt.imshow(get_tv(npy_clean).astype(np.float), cmap='gray')
            plt.axis('off')
            plt.title('Clean')

            plt.subplot(222)
            plt.imshow(get_tv(npy).astype(np.float), cmap='gray')
            plt.axis('off')
            plt.title('Gibbs')

            ax223 = plt.subplot(223)
            plt.imshow(get_tv(npy_clean).astype(np.float), cmap='gray')
            plt.imshow(heatmap_clean, cmap='jet', alpha=0.5)
            draw_labels_on_subplot(ax223, y_preds_clean)
            plt.axis('off')

            ax224 = plt.subplot(224)
            plt.imshow(get_tv(npy).astype(np.float), cmap='gray')
            plt.imshow(heatmap, cmap='jet', alpha=0.5)
            draw_labels_on_subplot(ax224, y_preds)
            plt.axis('off')

            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    path_model = r"../output/20211102_0925/model.hdf5"
    path_txt = r"D:\CU Data\Datagen\artifactID_IXI_sagittal_Gibbs\test.txt"
    main(path_model, path_txt)
