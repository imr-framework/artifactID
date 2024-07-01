from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from skimage import transform
from tqdm import tqdm

from artifactID.gradcam import gradcam
from common_utils import data_loader, preprocessor


def main(path_read: Path, path_model: Path):
    # Read all NIFTI volumes
    files_dicom = list(path_read.glob('*/dicom'))
    all_vols = []
    results = {"filename": [], "num_gibbs": []}
    print("Loading data...")
    for f in tqdm(files_dicom[:5]):
        vol = data_loader.load_data(path_data=f, data_format="dicom", normalize=True, central_50pc_crop=True,
                                    target_size=256, return_dicoms=False)
        vol = preprocessor.mask_subject(vol)
        # vol = transform.rotate(vol, 10)
        vol = np.clip(vol, *np.percentile(vol, (0.5, 99.5), axis=(0, 1)))
        # vol = preprocessor.normalize_volume(vol)
        # vol[-50:] = 0
        vol = np.moveaxis(vol, -1, 0) * 255.0

        all_vols.append(vol)
        results["filename"].append(str(f.parent.name))

    # Inference
    print(f"Performing inference...")
    for i, vol in tqdm(enumerate(all_vols)):
        dataset_inf = tf.data.Dataset.from_tensor_slices(vol)
        dataset_inf = dataset_inf.batch(64)
        all_heatmaps, all_y_preds = gradcam.main(path_model=str(path_model), dataset_test=dataset_inf)

        num_gibbs = np.count_nonzero(all_y_preds)
        results["num_gibbs"].append(num_gibbs)
        print(f"{num_gibbs} slices have Gibbs Ringing")
        print("Saving output to disk...")

        # path_save = files_dicom[i].parent / ("gradcam")
        # path_save.mkdir(parents=False, exist_ok=True)
        # for i in range(vol.shape[0]):
        #     npy = vol[i]
        #     heatmap = all_heatmaps[i]
        #     # heatmap *= 255.0
        #     # heatmap = heatmap.astype(np.uint8)
        #     plt.imshow(npy, cmap='gray')
        #     plt.imshow(heatmap, cmap='jet', alpha=0.25)
        #     plt.axis('off')
        #     decision = 'Gibbs ringing' if all_y_preds[i] else 'No Gibbs ringing'
        #     plt.title(decision)
        #     # plt.savefig(str(path_save / f"{i + 1}.png"))
        #     # plt.close()
        #     plt.show()

    # Save output as a CSV
    df = pd.DataFrame(results)
    df.to_csv(str(files_dicom[0].parent.parent / "results.csv"))


if __name__ == '__main__':
    path_read = Path(r"D:\Sravan\Data\Source\202312-T1w_MtSinai_YasminHurd")
    path_model = Path(r"../models/20220902_1515_gamma1_finalpick/best_model.100")
    # path_model = Path(r"../models/20240310_2245/model.500")

    main(path_read, path_model)
