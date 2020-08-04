import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from artifactID.common import data_ops


def main(path_read_data: str, path_save_data: str, patch_size: int):
    arr_rot_range = np.hstack((np.arange(-15, 0), np.arange(1, 16)))
    arr_rot_range = list(arr_rot_range)

    # =========
    # PATHS
    # =========
    if 'miccai' in path_read_data.lower():
        arr_path_read = data_ops.glob_brats_t1(path_brats=path_read_data)
    else:
        arr_path_read = data_ops.glob_nifti(path=path_read_data)
    path_save_data = Path(path_save_data)
    subjects_per_class = math.ceil(
        len(arr_path_read) / len(arr_rot_range))  # Calculate number of subjects per class
    arr_rot_range = arr_rot_range * subjects_per_class
    np.random.shuffle(arr_rot_range)

    # =========
    # DATAGEN
    # =========
    for ind, path_t1 in tqdm(enumerate(arr_path_read)):
        vol = data_ops.load_nifti_vol(path_t1)
        rot = arr_rot_range[ind]
        vol_norm = np.zeros(vol.shape)
        vol_norm = cv2.normalize(vol, vol_norm, 0, 255, cv2.NORM_MINMAX)
        vol_rot = np.zeros(vol.shape)
        for sl in range(vol_norm.shape[-1]):
            slice = Image.fromarray(vol_norm[:, :, sl])
            slice_rot = slice.rotate(rot)
            vol_rot[:, :, sl] = slice_rot

        # Zero-pad vol, get patches, discard empty patches and uniformly intense patches and normalize each patch
        vol_rot = data_ops.patch_compatible_zeropad(vol=vol_rot, patch_size=patch_size)
        patches, patch_map = data_ops.get_patches(vol=vol_rot, patch_size=patch_size)
        patches = data_ops.normalize_patches(patches=patches)

        # Save to disk
        _path_save = path_save_data.joinpath(f'rot{rot}')
        if not _path_save.exists():
            _path_save.mkdir(parents=True)
        for counter, p in enumerate(patches):
            suffix = '.nii.gz' if '.nii.gz' in path_t1.name else '.nii'
            subject = path_t1.name.replace(suffix, '')
            _path_save2 = _path_save.joinpath(subject)
            _path_save2 = str(_path_save2) + f'_patch{counter}.npy'
            np.save(arr=p, file=_path_save2)
