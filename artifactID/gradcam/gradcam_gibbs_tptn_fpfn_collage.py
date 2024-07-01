from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from artifactID.gradcam import gradcam as gradcam


def main(path_model: Path, path_data: Path, n: int):
    for folder in ["tp", "tn", "fp", "fn"]:
        path_txt = path_data / f"{folder}.txt"

        if not path_txt.exists():
            continue

        text = path_txt.read_text().splitlines()
        min_n = min(int(text[0]), n)
        if min_n == 0:
            continue
        files = np.random.choice(text[1:], min_n)

        # Load clean version, where applicable
        all_npy_clean = []
        if folder in ["tp", "fn"]:
            for f in files:
                path_npy_noartifact = list(Path(f).parts)
                path_npy_noartifact[-3] = "noartifact"
                path_npy_noartifact = Path().joinpath(*path_npy_noartifact)
                npy_clean = np.load(str(path_npy_noartifact))
                all_npy_clean.append(npy_clean)

        all_npy = []
        all_labels = []
        for f in files:
            npy = np.load(f)
            all_npy.append(npy)
            if "gibbs" in Path(f).parent.parent.name:
                all_labels.append(1)
            else:
                all_labels.append(0)

            # Perform Grad-CAM
        all_heatmaps, all_y_preds = gradcam.main(
            files=files, path_model=str(path_model)
        )

        # Make save destination
        path_save = path_data / folder
        if not path_save.exists():
            path_save.mkdir(parents=False, exist_ok=True)

        # Viz and save
        for i in range(len(files)):
            plt.figure()
            if folder in ["tp", "fn"]:
                plt.subplot(121)
                plt.imshow(all_npy_clean[i] - all_npy[i], cmap="gray")
                plt.axis("off")
                plt.subplot(122)
            plt.imshow(all_npy[i], cmap="gray")
            plt.imshow(all_heatmaps[i], cmap="jet", alpha=0.5)
            plt.axis("off")
            plt.title(f"Label: {all_labels[i]} | Pred: {all_y_preds[i]}")
            filename = path_save / f"{i + 1}.png"
            plt.savefig(str(filename), bbox_inches="tight")
            plt.close()


if __name__ == "__main__":
    all_models = [
        Path(r"../models/20220902_1515_gamma1_finalpick/best_model.100"),
    ]
    all_data = [
        Path(r"D:\Sravan\Data\Datagen\ArtifactID\IXI-T1"),
    ]
    n = 10

    for path_model, path_data in zip(all_models, all_data):
        print(f"Working on:")
        print(path_model)
        print(path_data)
        main(path_model=path_model, path_data=path_data, n=n)
        print()
