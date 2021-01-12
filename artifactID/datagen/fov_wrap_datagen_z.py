import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from artifactID.common import data_ops


def main(path_read_data: Path, path_save_data: Path, slice_size: int):
    arr_wrap_range = [5, 10, 15, 20, 25]

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

    arr_wrap_range = np.tile(arr_wrap_range, subjects_per_class)
    np.random.shuffle(arr_wrap_range)

    # =========
    # DATAGEN
    # =========
    for ind, path_t1 in tqdm(enumerate(arr_path_read)):
        vol = data_ops.load_nifti_vol(path_t1)
        vol = data_ops.resize(vol, size=slice_size)

        wrap = arr_wrap_range[ind]
        nonzero_idx = np.nonzero(vol)

        first_z, last_z = nonzero_idx[2].min(), nonzero_idx[2].max()
        while len(np.nonzero(np.round(vol[:,:,first_z],2))[0])/(vol.shape[0]*vol.shape[1])*100 < 10:
            first_z += 1
        while len(np.nonzero(np.round(vol[:,:,last_z],2))[0])/(vol.shape[0]*vol.shape[1])*100 < 10:
            last_z -= 1
        vol_cropped_z = vol[:, :, 75:last_z]
        bottom_sl = vol_cropped_z[:, :, 1:wrap + 1]
        opacity = 0.5 * np.linspace(1, 0.2, wrap)
        # Now extract the overlapping regions
        # This is because the central unmodified region should not be classified as FOV wrap-around artifact
        vol_wrapped_z = vol_cropped_z[:, :, -wrap:] + np.flip(bottom_sl * opacity, axis=2)
        vol_wrapped_z = data_ops.resize(vol_wrapped_z, size=slice_size)

        # Convert to float16 to avoid dividing by 0 during normalization - very low max values get zeroed out
        vol_wrapped = vol_wrapped_z.astype(np.float16)
        vol_wrapped_normalized = data_ops.normalize_slices(vol=vol_wrapped)

        # Save to disk
        _path_save = path_save_data.joinpath(f'wrap_z{wrap}')
        if not _path_save.exists():
            _path_save.mkdir(parents=True)
        for i in range(vol_wrapped_normalized.shape[-1]):
            _slice = vol_wrapped_normalized[..., i]
            suffix = '.nii.gz' if '.nii.gz' in path_t1.name else '.nii'
            subject = path_t1.name.replace(suffix, '')
            _path_save2 = _path_save.joinpath(subject)
            _path_save2 = str(_path_save2) + f'_slice{i}.npy'
            np.save(arr=_slice, file=_path_save2)