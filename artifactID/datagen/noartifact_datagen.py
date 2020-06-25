import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from artifactID.common.data_ops import glob_brats_t1, glob_nifti, load_nifti_vol, get_patches


def main(path_read_data: str, path_save_data: str, patch_size: int):
    # =========
    # PATHS
    # =========
    if 'miccai' in path_read_data.lower():
        arr_path_read = glob_brats_t1(path_brats=path_read_data)
    else:
        arr_path_read = glob_nifti(path=path_read_data)
    path_save_data = Path(path_save_data) / 'noartifact'
    if not path_save_data.exists():
        path_save_data.mkdir(parents=True)

    # =========
    # DATAGEN
    # =========
    for path_t1 in tqdm(arr_path_read):
        vol = load_nifti_vol(path_t1)

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

        # Save to disk
        for counter, p in enumerate(patches):
            if np.sum(p) != 0:  # Discard empty patches
                # Normalize to [0, 1]
                _max = p.max()
                _min = p.min()
                if _max != _min:
                    p = (p - _min) / (_max - _min)

                    suffix = '.nii.gz' if '.nii.gz' in path_t1.name else '.nii'
                    subject = path_t1.name.replace(suffix, '')
                    _path_save = path_save_data.joinpath(subject)
                    _path_save = str(_path_save) + f'_patch{counter}.npy'
                    np.save(arr=p, file=_path_save)
