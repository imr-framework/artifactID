from pathlib import Path

import numpy as np
from tqdm import tqdm

from artifactID.common import data_ops


def main(path_read_data: Path, path_save_data: Path, patch_size: list):
    # =========
    # PATHS
    # =========
    arr_path_read = data_ops.glob_nifti(path=path_read_data)
    path_save_data = path_save_data / 'noartifact'
    if not path_save_data.exists():
        path_save_data.mkdir(parents=True)

    # =========
    # DATAGEN
    # =========
    for path_t1 in tqdm(arr_path_read):
        vol = data_ops.load_nifti_vol(path_t1)

        # Zero-pad vol, get patches, discard empty patches and uniformly intense patches and normalize each patch
        vol = data_ops.patch_compatible_zeropad(vol=vol, patch_size=patch_size)
        patches, patch_map = data_ops.get_patches(vol=vol, patch_size=patch_size)
        patches = data_ops.normalize_patches(patches=patches)

        # Save to disk
        for counter, p in enumerate(patches):
            suffix = '.nii.gz' if '.nii.gz' in path_t1.name else '.nii'
            subject = path_t1.name.replace(suffix, '')
            _path_save = path_save_data.joinpath(subject)
            _path_save = str(_path_save) + f'_patch{counter}.npy'
            np.save(arr=p, file=_path_save)
