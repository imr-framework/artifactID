from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from artifactID.gradcam import gradcam as gradcam


def main(path_model: str, path_txt: str):
    path_txt = Path(path_txt)
    files = path_txt.read_text().splitlines()[1:]
    # inference.txt will have shuffled data unlike train/val/test.txt where noartifact/gibbs are alternating
    files_clean = []
    files_gibbs = []
    for f in files[:100]:
        folder = Path(f).parent.parent.name
        if folder == "noartifact":
            files_clean.append(f)
            # Add corresponding gibbs file
            f = list(Path(f).parts)
            f[-3] = "gibbs"
            files_gibbs.append(str(Path(*f)))

    # Perform Grad-CAM
    all_heatmaps_clean, all_y_pred_clean = gradcam.main(path_model=path_model, files=files_clean)
    all_heatmaps, all_y_preds = gradcam.main(path_model=path_model, files=files_gibbs)

    for i in range(len(files_clean)):
        npy_clean = np.load(files_clean[i])
        y_pred_clean = all_y_pred_clean[i]
        heatmap_clean = all_heatmaps_clean[i]

        npy = np.load(files_gibbs[i])
        y_pred = all_y_preds[i]
        heatmap = all_heatmaps[i]

        # Print output
        print(f"Pred clean: {y_pred_clean}")
        print(f"Pred: {y_pred}")

        # Plotting
        plt.subplot(231)
        plt.imshow(npy_clean, cmap="gray")
        plt.axis("off")
        plt.title("Clean")

        plt.subplot(232)
        plt.imshow(npy, cmap="gray")
        plt.axis("off")
        plt.title("Gibbs")

        plt.subplot(233)
        plt.imshow(npy - npy_clean, cmap="gray")
        plt.axis("off")
        plt.title("Diff")

        plt.subplot(234)
        plt.imshow(npy_clean, cmap="gray")
        plt.imshow(heatmap_clean, cmap="jet", alpha=0.5)
        plt.title(f"Pred - {y_pred_clean}")
        plt.axis("off")

        plt.subplot(235)
        plt.imshow(npy, cmap="gray")
        plt.imshow(heatmap, cmap="jet", alpha=0.5)
        plt.title(f"Pred - {y_pred}")
        plt.axis("off")

        plt.subplot(236)
        residual = npy - npy_clean
        cutart = np.zeros_like(residual)
        cutart[residual != 0] = 1
        plt.imshow(cutart, cmap="gray")
        plt.title(f"CutArt")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    path_model = r"../models/20220902_1515_gamma1_finalpick/best_model.100"
    path_model = r"../models/20240311_2242/model.78"
    path_txt = r"D:\Sravan\Data\Datagen\ArtifactID_v2\IXI-T1\train.txt"
    main(path_model, path_txt)
