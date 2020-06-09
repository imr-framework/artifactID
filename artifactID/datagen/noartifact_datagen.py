import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from artifactID.datagen.data_ops import glob_brats_t1, load_nifti_vol, get_patches


def main(path_read_data: str, path_save_data: str, patch_size: int):
    # =========
    # PATHS
    # =========
    arr_path_read = glob_brats_t1(path_brats=path_read_data)
    path_save_data = Path(path_save_data)

    # =========
    # DATAGEN
    # =========
    for path_t1 in tqdm(arr_path_read):
        vol = load_nifti_vol(path_t1)
        # Normalize to [0, 1]
        _max = vol.max()
        _min = vol.min()
        vol = (vol - _min) / (_max - _min)

        # Zero pad to compatible shape
        pad = []
        shape = vol.shape
        for s in shape:
            if s % patch_size != 0:
                p = patch_size - (s % patch_size)
                pad.append((math.floor(p / 2), math.ceil(p / 2)))
            else:
                pad.append((0, 0))

        # Extract patches
        vol = np.pad(array=vol, pad_width=pad)
        patches = get_patches(arr=vol, patch_size=patch_size)
        patches = patches.reshape((-1, patch_size, patch_size, patch_size))
        patches = patches.astype(np.float16)

        # Save to disk
        for counter, p in enumerate(patches):
            subject = path_t1.name.replace('.nii.gz', '')
            _path_save = path_save_data.joinpath('noartifact', subject)
            _path_save = str(_path_save) + f'_patch{counter}.npy'
            np.save(arr=p, file=_path_save)
