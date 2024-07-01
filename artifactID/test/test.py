from pathlib import Path

import numpy as np
from PIL import Image as pillow_im
from matplotlib import pyplot as plt

from artifactID.gradcam import gradcam
from common_utils import data_loader
from test_utils import make_dataset_from_generator_inference


def main(input_size: int, path_model: Path, path_data: Path, save_output: bool, viz_output: bool):
    # =========
    # INFERENCE
    # =========
    print(f"Performing inference...")
    # Inference
    vol = data_loader.load_data(path_data, data_format='dicom', normalize=True)
    vol *= 255.0
    vol = vol.astype(np.uint8)
    dataset_inf = make_dataset_from_generator_inference(path_data, input_size)
    dataset_inf = dataset_inf.batch(64)
    all_heatmaps, all_y_preds = gradcam.main(path_model=str(path_model), dataset_test=dataset_inf)

    print(f"{np.count_nonzero(all_y_preds)}/{vol.shape[-1]} slices contain Gibbs Ringing")
    center = vol.shape[-1] // 2
    cropped_all_y_preds = all_y_preds[center - center // 2:center + center // 2]
    cropped_vol = vol[..., center - center // 2:center + center // 2]
    print(
        f"Central 50% slab: {np.count_nonzero(cropped_all_y_preds)}/{cropped_vol.shape[-1]} slices contain Gibbs Ringing")

    # =========
    # VISUALIZE OUTPUT
    # =========
    all_heatmaps = np.stack(all_heatmaps)
    all_heatmaps = np.moveaxis(all_heatmaps, 0, -1)
    if viz_output:
        # sass.scroll_mask(vol, mask=all_heatmaps)
        for i in range(vol.shape[-1]):
            plt.imshow(vol[..., i], cmap='gray')
            plt.imshow(all_heatmaps[i], cmap='jet', alpha=all_heatmaps[i])
            plt.show()

    # =========
    # SAVE TO DISK
    # =========
    if save_output:
        print("Saving output to disk...")
        path_save = path_data / "output"
        path_save.mkdir(parents=False, exist_ok=True)
        for i in range(vol.shape[-1]):
            print(f"{i + 1}/{vol.shape[-1]}")
            npy = vol[..., i]
            y_pred = all_y_preds[i]
            heatmap = all_heatmaps[i]
            heatmap *= 255.0
            heatmap = heatmap.astype(np.uint8)
            heatmap = pillow_im.fromarray(heatmap)

            # Plotting
            image = pillow_im.fromarray(npy)
            image.paste(heatmap, (0, 0), mask=heatmap)
            image.save(str(path_save / f"{i + 1}.png"))


if __name__ == "__main__":
    input_size = 256
    path_data = Path(r"D:\Sravan\Data\Datagen\ArtifactID\IXI_3T-T1_defaced\tp.txt")
    path_model = Path(r"models/20220902_1515_gamma1_finalpick/best_model.100")

    # Perform inference
    main(
        input_size=input_size,
        path_model=path_model,
        path_data=path_data,
        save_output=False,
        viz_output=True
    )
