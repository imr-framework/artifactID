from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm

from artifactID.gradcam import gradcam
from common_utils import preprocessor, data_loader


def main(path_read: Path, path_model: Path, save_heatmaps: bool = False):
    # Iterate over all DICOM folders
    dicom_folders = list(path_read.glob('*'))
    results = {"folder": [], "num_gibbs": []}
    for folder in tqdm(dicom_folders):
        files = list(folder.glob("*"))
        is_dicom_dir = all([f.suffix != ".png" for f in files])
        if folder.is_dir() and is_dicom_dir:
            # Read DICOM folder
            vol = data_loader.load_dicom_folder(folder)
            if vol.shape[0] != 256 or vol.shape[1] != 256:
                vol = preprocessor.resize(vol, 256)

            # Subject specific data preproc
            # S1 - rot -10
            # S2 - shift down by 20
            # S3 - rot -12
            # S4 - rot -15

            # vol = transform.rotate(vol, -10)  # S1, S3, S4
            # vol = vol[:, :-20]
            # vol = np.pad(vol, ((0, 0), (20, 0), (0, 0)))

            vol = preprocessor.normalize_volume(vol)
            vol = np.moveaxis(vol, -1, 0)

            # Inference
            print(f"Performing inference...")
            dataset_inf = tf.data.Dataset.from_tensor_slices(vol)
            dataset_inf = dataset_inf.batch(64)
            all_heatmaps, all_y_preds = gradcam.main(path_model=str(path_model), dataset_test=dataset_inf)

            num_gibbs = np.count_nonzero(all_y_preds)
            print(f"{num_gibbs} slices have Gibbs Ringing in {folder.name}")
            results["folder"].append(folder.name)
            results["num_gibbs"].append(num_gibbs)

            # Save heatmaps to disk
            if save_heatmaps:
                print("Saving output to disk...")
                path_save = folder.parent / (folder.stem + "_output")
                path_save.mkdir(parents=False, exist_ok=True)
                for i in range(vol.shape[0]):
                    npy = vol[i]
                    heatmap = all_heatmaps[i]
                    heatmap *= 255.0
                    heatmap = heatmap.astype(np.uint8)
                    plt.imshow(npy, cmap='gray')
                    plt.imshow(heatmap, cmap='jet', alpha=0.25)
                    plt.axis('off')
                    decision = 'Gibbs ringing' if all_y_preds[i] else 'No Gibbs ringing'
                    plt.title(decision)
                    plt.savefig(str(path_save / f"{i + 1}.png"))
                    plt.close()
                    # plt.show()
    df = pd.DataFrame.from_dict(results)
    df.to_csv(path_read / "results.csv")


if __name__ == '__main__':
    for i in range(1, 6):
        path_read = Path(fr"D:\Sravan\Data\Source\202312-GE_prospective_5subjects\sub-0{i}")
        path_model = Path(r"models/20240306_2050/model.350")

        main(path_read, path_model, save_heatmaps=True)
