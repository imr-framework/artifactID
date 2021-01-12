import math
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from artifactID.common import data_ops


def main(path_read_data: Path, path_save_data: Path, slice_size: int):
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

    arr_wrap_range = np.tile(arr_wrap_range, subjects_per_class)
    np.random.shuffle(arr_wrap_range)

    # =========
    # DATAGEN
    # =========
    for ind, path_t1 in tqdm(enumerate(arr_path_read)):
        vol = data_ops.load_nifti_vol(path_t1)
        vol_resized = data_ops.resize_vol(vol, size=slice_size)

        wrap = arr_wrap_range[ind]
        opacity = 0.75
        direction = random.choice(['v', 'h'])  # vertical, horizontal

        nonzero_idx = np.nonzero(vol_resized)
        # Extract regions and construct overlap
        if direction == 'v':
            first, last = nonzero_idx[0].min(), nonzero_idx[0].max()
            vol_cropped = vol_resized[first:last]
            top, middle, bottom = vol_cropped[:wrap], vol_cropped[wrap:-wrap], vol_cropped[-wrap:]

            top_nonzero_idx = np.nonzero(top)
            bottom_nonzero_idx = np.nonzero(bottom)
            sl_min = min(top_nonzero_idx[-1].min(), bottom_nonzero_idx[-1].min())
            sl_max = max(top_nonzero_idx[-1].max(), bottom_nonzero_idx[-1].max())
            top, middle, bottom = top[..., sl_min:sl_max + 1], middle[..., sl_min:sl_max + 1], bottom[...,sl_min:sl_max + 1]

            middle[:wrap] += bottom * opacity
            middle[-wrap:] += top * opacity
            # Now extract the overlapping regions
            # This is because the central unmodified region should not be classified as FOV wrap-around artifact
            #wrap1, wrap2 = middle[:wrap], middle[-wrap:]
            vol_wrapped = data_ops.vol_slice_pad(middle, 0)

        elif direction == 'h':
            first, last = nonzero_idx[1].min(), nonzero_idx[1].max()
            vol_cropped = vol_resized[:, first:last]
            left, middle, right = vol_cropped[:, :wrap], vol_cropped[:, wrap:-wrap], vol_cropped[:, -wrap:]

            left_nonzero_idx = np.nonzero(left)
            right_nonzero_idx = np.nonzero(right)
            sl_min = min(left_nonzero_idx[-1].min(), right_nonzero_idx[-1].min())
            sl_max = max(left_nonzero_idx[-1].max(), right_nonzero_idx[-1].max())
            left, middle, right = left[..., sl_min:sl_max + 1], middle[..., sl_min:sl_max + 1], right[...,sl_min:sl_max + 1]

            middle[:, -wrap:] += left * opacity
            middle[:, :wrap] += right * opacity
            # Now extract the overlapping regions
            # This is because the central unmodified region should not be classified as FOV wrap-around artifact
            #wrap1, wrap2 = middle[:, :wrap], middle[:, -wrap:]
            vol_wrapped = data_ops.vol_slice_pad(middle, 1)

        # Convert to float16 to avoid dividing by 0 during normalization - very low max values get zeroed out
        vol_wrapped = vol_wrapped.astype(np.float16)
        vol_wrapped_normalized = data_ops.normalize_slices(vol=vol_wrapped)

        # Save to disk
        _path_save = path_save_data.joinpath(f'wrap{wrap}')
        if not _path_save.exists():
            _path_save.mkdir(parents=True)
        for i in range(vol_wrapped_normalized.shape[-1]):
            _slice = vol_wrapped_normalized[..., i]
            suffix = '.nii.gz' if '.nii.gz' in path_t1.name else '.nii'
            subject = path_t1.name.replace(suffix, '')
            _path_save2 = _path_save.joinpath(subject)
            _path_save2 = str(_path_save2) + f'_slice{i}.npy'
            np.save(arr=_slice, file=_path_save2)

