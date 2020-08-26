import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from artifactID.common import data_ops


def main(path_read_data: Path, path_save_data: Path, patch_size: int):
    arr_snr_range = [2, 5, 11, 15, 20]

    # =========
    # PATHS
    # =========
    if 'miccai' in str(path_read_data).lower():
        arr_path_read = data_ops.glob_brats_t1(path_brats=path_read_data)
    else:
        arr_path_read = data_ops.glob_nifti(path=path_read_data)
    path_save_data = Path(path_save_data)
    subjects_per_class = math.ceil(
        len(arr_path_read) / len(arr_snr_range))  # Calculate number of subjects per class
    arr_snr_range = arr_snr_range * subjects_per_class
    np.random.shuffle(arr_snr_range)

    # =========
    # DATAGEN
    # =========
    for ind, path_t1 in tqdm(enumerate(arr_path_read)):
        # Load from disk, comes with ideal (0) noise outside brain
        snr = arr_snr_range[ind]
        vol = data_ops.load_nifti_vol(path_t1)
        idx_object = np.nonzero(vol)
        awgn_std = vol[idx_object].mean() / math.pow(10, snr / 20)
        noise = np.random.normal(loc=0, scale=awgn_std, size=vol.size).reshape(vol.shape)
        vol = vol + noise

        # Zero-pad vol, get patches, discard empty patches and uniformly intense patches and normalize each patch
        vol = data_ops.patch_size_compatible_zeropad(vol=vol, patch_size=patch_size)
        patches, _ = data_ops.get_patches_per_slice(vol=vol, patch_size=patch_size)
        patches = data_ops.normalize_patches(patches=patches)

        # Save to disk
        if snr == 2 or snr == 5:
            snr = 99
        _path_save = path_save_data.joinpath(f'snr{snr}')
        if not _path_save.exists():
            _path_save.mkdir(parents=True)
        for counter, p in enumerate(patches):
            suffix = '.nii.gz' if '.nii.gz' in path_t1.name else '.nii'
            subject = path_t1.name.replace(suffix, '')
            _path_save2 = _path_save.joinpath(subject)
            _path_save2 = str(_path_save2) + f'_slice{counter}.npy'
            np.save(arr=p, file=_path_save2)
