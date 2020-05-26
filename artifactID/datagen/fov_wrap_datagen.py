import math
from pathlib import Path

import numpy as np

from artifactID.common.data_ops import glob_brats_t1, load_nifti_vol


def main(path_brats: str, path_save: str):
    arr_wrap_range = [55, 60, 65, 70, 75, 80]

    # =========
    # BRATS PATHS
    # =========
    arr_path_brats_t1 = glob_brats_t1(path_brats=path_brats)
    num_subjects = len(arr_path_brats_t1)

    subjects_per_class = math.ceil(
        len(arr_path_brats_t1) / len(arr_wrap_range))  # Calculate number of subjects per class
    arr_wrap_range = arr_wrap_range * subjects_per_class
    np.random.shuffle(arr_wrap_range)

    path_save = Path(path_save)
    path_all = [path_save / f'wrap{wrap}' for wrap in arr_wrap_range]
    # Make save folders if they do not exist
    for p in path_all:
        if not p.exists():
            p.mkdir(parents=True)

    # =========
    # DATAGEN
    # =========
    for ind, path_t1 in enumerate(arr_path_brats_t1):
        subject_name = path_t1.parts[-1].split('.nii.gz')[0]  # Extract subject name from path

        pc = round((ind + 1) / num_subjects * 100, ndigits=2)
        print(f'{pc}%', end=', ', flush=True)

        vol = load_nifti_vol(path_t1)
        wrap_ex = arr_wrap_range[ind]
        opacity = 0.5
        top, middle, bottom = vol[:wrap_ex], vol[wrap_ex:-wrap_ex], vol[-wrap_ex:]
        middle[:wrap_ex] += bottom * opacity
        middle[-wrap_ex:] += top * opacity
        middle = np.pad(middle, [[wrap_ex, wrap_ex], [0, 0], [0, 0]])
        # Normalize to [0, 1]
        _max = middle.max()
        _min = middle.min()
        middle = (middle - _min) / (_max - _min)

        # Zero pad back to 155
        orig_num_slices = 155
        n_zeros = (orig_num_slices - middle.shape[2]) / 2
        n_zeros = [math.floor(n_zeros), math.ceil(n_zeros)]
        middle = np.pad(middle, [[0, 0], [0, 0], n_zeros])
        middle = middle.astype(np.float16)

        _path_save = str(path_save / f'wrap{wrap_ex}' / subject_name) + '.npy'
        np.save(arr=middle, file=_path_save)  # Save to disk
