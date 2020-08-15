import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from artifactID.common import data_ops


def main(path_read_data: Path, path_save_data: Path, patch_size: list):
    arr_gibbs_range = [52, 64, 76]
    # =========
    # PATHS
    # =========
    arr_path_read = data_ops.glob_nifti(path=path_read_data)
    subjects_per_class = math.ceil(
        len(arr_path_read) / len(arr_gibbs_range))  # Calculate number of subjects per class

    arr_gibbs_range = np.tile(arr_gibbs_range, subjects_per_class)
    np.random.shuffle(arr_gibbs_range)

    # =========
    # DATAGEN
    # =========
    for ind, path_t1 in tqdm(enumerate(arr_path_read)):
        vol = data_ops.load_nifti_vol(path=path_t1)
        chop = arr_gibbs_range[ind]

        kdat = np.fft.fftshift(np.fft.fftn(vol))
        kdat[:, :chop, :] = 0
        kdat[:, -chop:, :] = 0
        vol_gibbs = np.abs(np.fft.ifftn(np.fft.ifftshift(kdat)))

        # Zero-pad vol, get patches, discard empty patches and uniformly intense patches and normalize each patch
        vol_gibbs = data_ops.patch_compatible_zeropad(vol=vol_gibbs, patch_size=patch_size)
        patches, patch_map = data_ops.get_patches(vol=vol_gibbs, patch_size=patch_size)
        patches = data_ops.normalize_patches(patches=patches)

        # Save to disk
        _path_save = path_save_data.joinpath(f'gibbs{chop}')
        if not _path_save.exists():
            _path_save.mkdir(parents=True)
        for counter, p in enumerate(patches):
            subject = path_t1.name.replace('.nii.gz', '')
            _path_save2 = _path_save.joinpath(subject)
            _path_save2 = str(_path_save2) + f'_patch{counter}.npy'
            np.save(arr=p, file=_path_save2)
