import math
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from artifactID.utils import glob_nifti
from common_utils import data_loader, preprocessor


def main(path_read_data: Path, path_save_data: Path, slice_size: int):
    wa_orientation = ['h', 'v']

    # =========
    # PATHS
    # =========
    arr_path_read = glob_nifti(path=path_read_data)
    path_save_data = Path(path_save_data)
    subjects_per_class = math.ceil(
        len(arr_path_read) / len(wa_orientation))  # Calculate number of subjects per class

    wa_orientation = np.tile(wa_orientation, subjects_per_class)
    np.random.shuffle(wa_orientation)

    # =========
    # DATAGEN
    # =========

    for ind, path_t1 in tqdm(enumerate(arr_path_read)):
        vol = data_loader.load_data(path_data=path_t1, data_format='nifti', normalize=True, dataset='IXI_T1',
                                    target_size=slice_size)

        wrap = 45
        opacity = 1  # 0.75
        direction = wa_orientation[ind]  # vertical, horizontal

        # Extract regions and construct overlap

        first, last = 30, 150  # nonzero_idx[1].min(), nonzero_idx[1].max()
        vol_cropped = vol[..., first:last]  # Remove the slices corresponding to the neck level

        # Horizontal wrap-around
        if direction == 'h':
            left, middle, right = vol_cropped[:, :wrap], vol_cropped[:, wrap:-wrap], vol_cropped[:, -wrap:]

            middle[:, -wrap:] += left * opacity
            middle[:, :wrap] += right * opacity

            vol_wrapped = []
            for sl in range(middle.shape[-1]):
                slice = middle[..., sl]
                # Check that there is substantial wrap (5% of the column pixels)
                sum_wrap_left = np.sum(slice[:, 0]) / (slice.shape[0]) * 100
                sum_wrap_right = np.sum(slice[:, -1]) / (slice.shape[0]) * 100
                if sum_wrap_left > 5 and sum_wrap_right > 5:
                    # Crop the same number of pixels top and bottom and resize
                    vol_wrapped.append(cv2.resize(slice[wrap:-(wrap + 1), :], (slice_size, slice_size)))

        # Vertical wrap-around
        else:
            top, middle, bottom = vol_cropped[:wrap], vol_cropped[wrap:-wrap], vol_cropped[-wrap:]

            middle[:wrap] += bottom * opacity
            middle[-wrap:] += top * opacity

            vol_wrapped = []
            for sl in range(middle.shape[-1]):
                slice = middle[..., sl]
                # Check that there is substantial wrap (5% of the row pixels)
                sum_wrap_top = np.sum(slice[0, :]) / (slice.shape[1]) * 100
                sum_wrap_bottom = np.sum(slice[-1, :]) / (slice.shape[1]) * 100
                if sum_wrap_top > 5 and sum_wrap_bottom > 5:
                    # Crop the same number of pixels left and right and resize
                    vol_wrapped.append(cv2.resize(slice[:, wrap:-(wrap + 1)], (slice_size, slice_size)))

        if vol_wrapped:
            vol_wrapped = np.stack(vol_wrapped, axis=-1)

            # Convert to float16 to avoid dividing by 0 during normalization - very low max values get zeroed out
            vol_wrapped = vol_wrapped.astype(np.float16)
            vol_wrapped_normalized = preprocessor.normalize_per_slice(vol=vol_wrapped)

            # Debug - visualize chosen slices
            # sass.scroll(vol_wrapped_normalized.astype(np.float32))

            # Save to disk
            _path_save = path_save_data.joinpath(f'wrap_xy')
            if not _path_save.exists():
                _path_save.mkdir(parents=True)
            for i in range(vol_wrapped_normalized.shape[-1]):
                _slice = vol_wrapped_normalized[..., i]
                suffix = '.nii.gz' if '.nii.gz' in path_t1.name else '.nii'
                subject = path_t1.name.replace(suffix, '')
                _path_save2 = _path_save.joinpath(subject)
                _path_save2 = str(_path_save2) + f'_slice{i}.npy'
                np.save(arr=_slice, file=_path_save2)
