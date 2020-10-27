from pathlib import Path

import numpy as np
from tqdm import tqdm

from artifactID.common import data_ops


def main(path_read_data: Path, path_save_data: Path, slice_size: int):
    # =========
    # PATHS
    # =========
    if 'miccai' in str(path_read_data).lower():
        arr_path_read = data_ops.glob_brats_t1(path_brats=path_read_data)
    else:
        arr_path_read = data_ops.glob_nifti(path=path_read_data)
    path_save_data = Path(path_save_data) / 'noartifact'
    if not path_save_data.exists():
        path_save_data.mkdir(parents=True)

    # =========
    # DATAGEN
    # =========
    for path_t1 in tqdm(arr_path_read):
        vol = data_ops.load_nifti_vol(path_t1)
        vol_resized = data_ops.resize(vol, size=slice_size)
        # Convert to float16 to avoid dividing by 0 during normalization - very low max values get zeroed out
        vol_resized = vol_resized.astype(np.float16)
        vol_normalized = data_ops.normalize_slices(vol_resized)

        # Save to disk
        for i in range(vol_normalized.shape[-1]):
            _slice = vol_normalized[..., i]
            suffix = '.nii.gz' if '.nii.gz' in path_t1.name else '.nii'
            subject = path_t1.name.replace(suffix, '')
            _path_save = path_save_data.joinpath(subject)
            _path_save = str(_path_save) + f'_slice{i}.npy'
            np.save(arr=_slice, file=_path_save)
