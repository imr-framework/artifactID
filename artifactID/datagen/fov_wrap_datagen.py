import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from artifactID.common.data_ops import glob_brats_t1, glob_nifti, load_nifti_vol, get_patches


def main(path_read_data: str, path_save_data: str, patch_size: int):
    arr_wrap_range = [55, 60, 65, 70, 75, 80]
    arr_wrap_range = [15, 20, 25, 30, 35]

    # =========
    # PATHS
    # =========
    if 'miccai' in path_read_data.lower():
        arr_path_read = glob_brats_t1(path_brats=path_read_data)
    else:
        arr_path_read = glob_nifti(path=path_read_data)
    path_save_data = Path(path_save_data)
    subjects_per_class = math.ceil(
        len(arr_path_read) / len(arr_wrap_range))  # Calculate number of subjects per class
    # arr_wrap_range = arr_wrap_range * subjects_per_class
    arr_wrap_range = np.tile(arr_wrap_range, subjects_per_class)
    # np.random.shuffle(arr_wrap_range)

    # =========
    # DATAGEN
    # =========
    for ind, path_t1 in tqdm(enumerate(arr_path_read)):
        vol = load_nifti_vol(path_t1)
        wrap = arr_wrap_range[ind]
        opacity = 0.5

        # Find extent of brain along axial
        nonzero_idx = np.nonzero(vol)
        first, last = nonzero_idx[0].min(), nonzero_idx[0].max()
        vol_cropped = vol[first:last]

        # Extract regions and construct overlap
        top, middle, bottom = vol_cropped[:wrap], vol_cropped[wrap:-wrap], vol_cropped[-wrap:]
        middle[:wrap] += bottom * opacity
        middle[-wrap:] += top * opacity

        # Now extract the overlapping regions
        # This is because the central unmodified region should not be classified as FOV wrap-around artifact
        top, bottom = middle[:wrap], middle[-wrap:]
        vol = np.append(top, bottom, axis=0)

        # Zero pad to compatible shape
        pad = []
        shape = vol.shape
        for s in shape:
            if s % patch_size != 0:
                p = patch_size - (s % patch_size)
                pad.append((math.floor(p / 2), math.ceil(p / 2)))
            else:
                pad.append((0, 0))
        vol = np.pad(array=vol, pad_width=pad)

        patches = get_patches(arr=vol, patch_size=patch_size)  # Extract patches

        # Save to disk
        _path_save = path_save_data.joinpath(f'wrap{wrap}')
        if not _path_save.exists():
            _path_save.mkdir(parents=True)
        for counter, p in enumerate(patches):
            if np.sum(p) != 0:  # Discard empty patches
                # Normalize to [0, 1]
                _max = p.max()
                _min = p.min()
                if _max != _min:
                    p = (p - _min) / (_max - _min)

                    suffix = '.nii.gz' if '.nii.gz' in path_t1.name else '.nii'
                    subject = path_t1.name.replace(suffix, '')
                    _path_save2 = _path_save.joinpath(subject)
                    _path_save2 = str(_path_save2) + f'_patch{counter}.npy'
                    np.save(arr=p, file=_path_save2)
