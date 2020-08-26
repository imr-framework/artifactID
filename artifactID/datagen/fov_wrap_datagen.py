import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from artifactID.common import data_ops


def main(path_read_data: Path, path_save_data: Path, patch_size: int):
    arr_wrap_range = [15, 20, 25, 30, 35]

    # =========
    # PATHS
    # =========
    if 'miccai' in str(path_read_data).lower():
        arr_path_read = data_ops.glob_brats_t1(path_brats=path_read_data)
    else:
        arr_path_read = data_ops.glob_nifti(path=path_read_data)
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
        vol = data_ops.load_nifti_vol(path_t1)
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

        # Now cycle through top and bottom chunks individually
        counter = 0
        for vol in (top, bottom):
            # Zero-pad vol, get patches, discard empty patches and uniformly intense patches and normalize each patch
            vol = data_ops.patch_size_compatible_zeropad(vol=vol, patch_size=patch_size)
            patches, _ = data_ops.get_patches_per_slice(vol=vol, patch_size=patch_size)
            patches = data_ops.normalize_patches(patches=patches)

            # Save to disk
            _path_save = path_save_data.joinpath(f'wrap{wrap}')
            if not _path_save.exists():
                _path_save.mkdir(parents=True)
            for p in patches:
                suffix = '.nii.gz' if '.nii.gz' in path_t1.name else '.nii'
                subject = path_t1.name.replace(suffix, '')
                _path_save2 = _path_save.joinpath(subject)
                _path_save2 = str(_path_save2) + f'_slice{counter}.npy'
                np.save(arr=p, file=_path_save2)
                counter += 1
