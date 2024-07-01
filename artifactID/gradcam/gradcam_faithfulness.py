from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from artifactID.gradcam import gradcam


def main(path_model: str, path_txt: str):
    path_txt = Path(path_txt)
    files = path_txt.read_text().splitlines()[1:]

    arr_relevance = []
    arr_npy = []
    arr_npy_clean = []
    for f in files:
        arr_npy.append(np.load(f))
        path_npy_clean = list(Path(f).parts)
        path_npy_clean[-3] = "noartifact"
        path_npy_clean = Path().joinpath(*path_npy_clean)
        npy_clean = np.load(str(path_npy_clean)).squeeze()
        arr_npy_clean.append(npy_clean)

    # Perform Grad-CAM
    all_heatmaps, all_y_preds = gradcam.main(files=files, path_model=path_model)

    gradcam_threshold = 0.85
    for i in range(len(files)):
        heatmap_thresholded = np.zeros_like(all_heatmaps[i])
        heatmap_thresholded[all_heatmaps[i] > gradcam_threshold] = 1

        cutart = arr_npy[i] - arr_npy_clean[i]
        cutart[cutart != 0] = 1

        # IOU
        intersection = np.count_nonzero(heatmap_thresholded * cutart)
        relevance = intersection / np.count_nonzero(heatmap_thresholded)
        # union = heatmap_thresholded + cutart
        # union[union > 0] = 1
        # union = np.count_nonzero(union)
        # iou = intersection / np.count_nonzero(cutart)
        arr_relevance.append(relevance)

        # Debug - visualize
        # plt.subplot(121)
        # plt.imshow(arr_npy[i], cmap='gray')
        # plt.imshow(heatmap_thresholded, cmap='jet', alpha=0.2)
        # plt.title(f"Pred: {all_y_preds[i]}")
        # plt.axis('off')
        #
        # plt.subplot(122)
        # plt.imshow(cutart, cmap='jet')
        # plt.imshow(heatmap_thresholded * 10, cmap='gray', alpha=0.2)
        # plt.title(f"{intersection / np.count_nonzero(heatmap_thresholded):.3g}")
        # plt.axis('off')
        #
        # plt.tight_layout()
        # plt.show()

    print(f"Threshold: {gradcam_threshold}")
    print(f"Mean relevance: {np.mean(arr_relevance)}, Std: {np.std(arr_relevance)}")


if __name__ == "__main__":
    path_model = r"../models/20220908_1645/best_model.77"
    path_txt = r"D:\Sravan\Data\Datagen\ArtifactID\IXI-T1\mini\tp.txt"
    main(path_model, path_txt)
