import pydicom as pyd
from pathlib import Path

import numpy as np
from tqdm import tqdm

from artifactID.common import data_ops


def main(path_read_data: Path, path_save_data: Path, slice_size: int):
    # =========
    # PATHS
    # =========
    arr_path_read = Path(r"")
    files = list(arr_path_read.glob('*'))
    vols_dict = {}
    paths_dict={}
    for f in files:
        d = pyd.dcmread(str(f)).pixel_array
        sub = f.stem[4:8]
        if sub in vols_dict:
            vols_dict[sub].append(d)
        else:
            vols_dict[sub] = [d]
            paths_dict[sub] = f

    vols = []
    paths = []
    for k, v in vols_dict.items():
        paths.append(paths_dict[k])
        vols.append(np.stack(v, axis=-1))

    path_save_data = Path(path_save_data) / 'noartifact'
    if not path_save_data.exists():
        path_save_data.mkdir(parents=True)

    # =========
    # DATAGEN
    # =========
    for ind, vol in tqdm(enumerate(vols)):
        vol_resized = data_ops.resize(vol, size=slice_size)
        # Convert to float16 to avoid dividing by 0 during normalization - very low max values get zeroed out
        vol_resized = vol_resized.astype(np.float16)
        vol_normalized = data_ops.normalize_slices(vol_resized)

        # Save to disk
        for i in range(vol_normalized.shape[-1]):
            _slice = vol_normalized[..., i]
            suffix = '.nii.gz' if '.nii.gz' in paths[ind].name else '.nii'
            subject = paths[ind].name.replace(suffix, '')
            _path_save = path_save_data.joinpath(subject)
            _path_save = str(_path_save) + f'_slice{i}.npy'
            np.save(arr=_slice, file=_path_save)
