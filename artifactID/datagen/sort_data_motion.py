import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def test_trainingGAN():
    base_dir = Path(r"D:\CU Data\Datagen\artifactID_GE\ArtifactID_motion")
    base_save_dir = Path(r"D:\CU Data\Datagen\artifactID_GE\ArtifactID_motion_sorted")
    images = list(base_dir.glob('*.npy'))
    dataframe = pd.DataFrame(columns=["displacement_norm", "rotation_norm", "image_name"])
    image_names = []
    displacement_norm_values = []
    rotation_norm_values = []
    for counter, image in enumerate(images):
        params = image.stem.split("_")
        displacement_norm = float(params[0])
        rotation_norm = float(params[1])
        displacement_norm_values.append(displacement_norm)
        rotation_norm_values.append(rotation_norm)
        image_names.append(image.name)
    dataframe['displacement_norm'] = displacement_norm_values
    dataframe['rotation_norm'] = rotation_norm_values
    dataframe['image_name'] = image_names

    for row in dataframe.iterrows():
        motion_norm = np.sqrt(
            np.power(np.asarray(row[1].displacement_norm), 2) + np.power(np.asarray(row[1].rotation_norm), 2))

        if motion_norm == 0:
            save_dir = base_save_dir / "noartifact"
        elif motion_norm > 0:
            save_dir = base_save_dir / "motion"
        # if motion_norm == 0:
        #     save_dir = base_save_dir / "M0/"
        # elif motion_norm <= 2.5 and motion_norm > 0:
        #     save_dir = base_save_dir / "M1/"
        # elif motion_norm <= 3.5 and motion_norm > 2.5:
        #     save_dir = base_save_dir / "M2/"
        # elif motion_norm <= 4.5 and motion_norm > 3.5:
        #     save_dir = base_save_dir / "M3/"
        # elif motion_norm > 4.5:
        #     save_dir = base_save_dir / "M4/"

        if not save_dir.exists():
            save_dir.mkdir(parents=False)
        save = save_dir / row[1].image_name
        shutil.copy(src=str(base_dir / row[1].image_name), dst=str(save))


test_trainingGAN()
