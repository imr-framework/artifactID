from pathlib import Path

import numpy as np
from tqdm import tqdm

from artifactID.utils import glob_nifti
from common_utils import data_loader


def main(path_read_data: Path, path_save_data: Path, slice_size: int):
    # =========
    # PATHS
    # =========
    arr_path_read = glob_nifti(path=path_read_data)

    path_save_data = Path(path_save_data) / 'noartifact'
    if not path_save_data.exists():
        path_save_data.mkdir(parents=True)

    # =========
    # DATAGEN
    # =========
    for f in tqdm(arr_path_read[:100]):  # TODO
        vol = data_loader.load_data(path_data=f, dataset='IXI_T1', data_format='nifti', normalize=True,
                                    target_size=slice_size)
        vol = vol[..., 25:125]  # TODO

        # Debug - visualize chosen slices
        # import sass
        # sass.scroll(vol.astype(np.float32))

        # Save to disk
        for i in range(vol.shape[-1]):
            _slice = vol[..., i]
            suffix = '.nii.gz' if '.nii.gz' in f.name else '.nii'
            subject = f.name.replace(suffix, '') + f'_slice{i}.npy'
            _path_save = path_save_data.joinpath(subject)
            np.save(arr=_slice, file=str(_path_save))
