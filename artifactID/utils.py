from pathlib import Path

import numpy as np
from matplotlib import patches


def get_tv_patches(img: np.ndarray) -> np.ndarray:
    img = img.squeeze()
    if len(img.shape) > 2:
        raise ValueError()

    patches = []
    for i in range(0, img.shape[0], 64):
        for j in range(0, img.shape[1], 64):
            p = img[i:i + 64, j:j + 64]
            p = p[:-1, :-1] - p[1:, 1:]
            p = np.expand_dims(p, axis=2)
            p = (p - p.mean()) / p.std()  # Normalize
            patches.append(p)
    patches = np.stack(patches)
    return patches


def glob_nifti(path: Path) -> list:
    arr_path = list(path.glob('**/*.nii.gz'))
    arr_path2 = list(path.glob('**/*.nii'))
    return arr_path + arr_path2


def draw_labels_on_subplot(ax, ypred: np.ndarray):
    bbox_size = 3
    for i in range(4):
        for j in range(4):
            x_coord = (j + 1) * 63 - 32 - bbox_size
            y_coord = (i + 1) * 63 - 32
            if ypred[i, j] == 1:  # Artifact
                rect = patches.Rectangle((x_coord, y_coord), bbox_size, bbox_size, edgecolor='red', facecolor='red')
                ax.add_patch(rect)
            else:  # No artifact
                rect = patches.Rectangle((x_coord, y_coord), bbox_size, bbox_size, edgecolor='white', facecolor='white')
                ax.add_patch(rect)


def stitch_patches(patches: np.ndarray) -> np.ndarray:
    n = int(np.sqrt(patches.shape[0]))
    patches = np.reshape(patches, (n, n, 63, 63))
    img = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(patches[i, j])
        row = np.hstack(row)
        img.append(row)
    img = np.vstack(img)
    return img
