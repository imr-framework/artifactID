import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from artifactID.common import data_ops


def main(path_read_data: Path, path_save_data: Path, slice_size: int):
    arr_gibbs_range = [52, 64, 76]
    # =========
    # PATHS
    # =========
    if 'miccai' in str(path_read_data).lower():
        arr_path_read = data_ops.glob_brats_t1(path_brats=path_read_data)
    else:
        arr_path_read = data_ops.glob_nifti(path=path_read_data)
    path_save_data = Path(path_save_data)
    subjects_per_class = math.ceil(
        len(arr_path_read) / len(arr_gibbs_range))  # Calculate number of subjects per class

    arr_gibbs_range = np.tile(arr_gibbs_range, subjects_per_class)
    np.random.shuffle(arr_gibbs_range)

    # =========
    # DATAGEN
    # =========
    for ind, path_t1 in tqdm(enumerate(arr_path_read)):
        vol = data_ops.load_nifti_vol(path_t1)
        vol_resized = data_ops.resize(vol, size=slice_size)
        chop = arr_gibbs_range[ind]

        kdat = np.fft.fftshift(np.fft.fftn(vol_resized))
        kdat[:, :chop, :] = 0
        kdat[:, -chop:, :] = 0
        vol_gibbs = np.abs(np.fft.ifftn(np.fft.ifftshift(kdat)))
        # Convert to float16 to avoid dividing by 0 during normalization - very low max values get zeroed out
        vol_gibbs = vol_gibbs.astype(np.float16)
        vol_gibbs_normalized = data_ops.normalize_slices(vol=vol_gibbs)

        # Save to disk
        _path_save = path_save_data.joinpath(f'gibbs{chop}')
        if not _path_save.exists():
            _path_save.mkdir(parents=True)
        for i in range(vol_gibbs_normalized.shape[-1]):
            _slice = vol[..., i]
            subject = path_t1.name.replace('.nii.gz', '')
            _path_save2 = _path_save.joinpath(subject)
            _path_save2 = str(_path_save2) + f'_slice{i}.npy'
            np.save(arr=_slice, file=_path_save2)
