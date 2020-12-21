import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from artifactID.common import data_ops


def main(path_read_data: Path, path_save_data: Path, slice_size: int):
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
        vol_resized = data_ops.resize(vol, size=slice_size)
        idx_object = np.nonzero(vol_resized)  # Locations of non-zero pixels in vol_resized
        awgn_std = vol_resized[idx_object].mean() / math.pow(10, snr / 20)
        noise = np.random.normal(loc=0, scale=awgn_std, size=vol_resized.size).reshape(vol_resized.shape)
        vol_snr = vol_resized + noise
        # Convert to float16 to avoid dividing by 0 during normalization - very low max values get zeroed out
        vol_snr_resized = vol_snr.astype(np.float16)
        vol_snr_normalized = data_ops.normalize_slices(vol_snr_resized)
        from matplotlib import pyplot as plt
        plt.imshow(vol_snr_normalized[..., 150].astype(np.float), cmap='gray')
        plt.axis('off')
        plt.savefig(r'C:\Users\sravan953\Documents\CU\Data\IXI002-Guys-0828-T1\snr.jpg')

        # Save to disk
        if snr == 2 or snr == 5:
            snr = 99
        _path_save = path_save_data.joinpath(f'snr{snr}')
        if not _path_save.exists():
            _path_save.mkdir(parents=True)
        for i in range(vol_snr_normalized.shape[-1]):
            _slice = vol_snr_normalized[..., i]
            suffix = '.nii.gz' if '.nii.gz' in path_t1.name else '.nii'
            subject = path_t1.name.replace(suffix, '')
            _path_save2 = _path_save.joinpath(subject)
            _path_save2 = str(_path_save2) + f'_slice{i}.npy'
            np.save(arr=_slice, file=_path_save2)
