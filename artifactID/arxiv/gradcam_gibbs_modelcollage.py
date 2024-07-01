from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

from artifactID.gradcam import gradcam


def main(path_model: Path, path_data: Path, n: int):
    # Load model
    model = load_model(str(path_model))

    for folder in ["tp", "tn", "fp", "fn"]:
        path_txt = path_data / f"{folder}.txt"

        text = path_txt.read_text().splitlines()
        min_n = min(int(text[0]), n)
        files = text[1:]
        for counter, path_npy in enumerate(np.random.choice(files, min_n)):
            # Determine label
            if "gibbs" in Path(path_npy).parent.parent.name:
                label = 1
            else:
                label = 0

            # Load numpy
            npy = np.load(path_npy).squeeze()

            # Load clean version, where applicable
            if folder in ["tp", "fn"]:
                path_npy_noartifact = list(Path(path_npy).parts)
                path_npy_noartifact[-3] = "noartifact"
                path_npy_noartifact = Path().joinpath(*path_npy_noartifact)
                npy_clean = np.load(str(path_npy_noartifact))

            # Perform Grad-CAM
            heatmap, y_pred = gradcam.main(npy=npy, model=model)

            # Print output
            print(f"Label: {label}")
            print(f"Pred: {y_pred}")

            # Make save destination
            path_save = path_data / folder
            if not path_save.exists():
                path_save.mkdir(parents=False, exist_ok=True)

            # Viz and save
            plt.figure()
            if folder in ["tp", "fn"]:
                plt.subplot(121)
                plt.imshow(npy_clean - npy, cmap="gray")
                plt.axis("off")
                plt.subplot(122)
            plt.imshow(npy, cmap="gray")
            plt.imshow(heatmap, cmap="jet", alpha=0.5)
            plt.axis("off")
            plt.title(f"Label: {label} | Pred: {y_pred}")
            filename = path_save / f"{counter}.png"
            plt.savefig(str(filename), bbox_inches="tight")
            plt.close()


if __name__ == "__main__":
    all_models = [
        # Path(r"../models/20220713_1009_fullvol/model.47"),
        # Path(r"../models/20220713_1042_fullvol2/model.41"),
        # Path(r"../models/20220719_1046_bet/model.96"),
        # Path(r"../models/20220719_1219_flirt_bet/model.81"),
        Path(r"../models/20220720_1745_IXI/model.98"),
    ]
    all_data = [
        # Path(r"E:\Data\Datagen\ArtifactID\HCP-Gibbs_sag"),
        # Path(r"E:\Data\Datagen\ArtifactID\HCP-Gibbs_sag-BET"),
        # Path(r"E:\Data\Datagen\ArtifactID\HCP-Gibbs_sag-FLIRT-BET"),
        Path(r"E:\Data\Datagen\ArtifactID\IXI-Gibbs_sag"),
    ]
    n = 20

    for path_model, path_data in zip(all_models, all_data):
        print(f"Working on:")
        print(path_model)
        print(path_data)
        main(path_model=path_model, path_data=path_data, n=n)
        print()
