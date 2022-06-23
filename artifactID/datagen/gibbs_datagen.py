from pathlib import Path

import numpy as np
from tqdm import tqdm

from artifactID.utils import glob_nifti
from common_utils import data_loader


def cutart(sli: np.ndarray, sli_gibbs: np.ndarray) -> np.ndarray:
    patch_size_low = 0.3 * sli.shape[0]
    patch_size_high = 0.5 * sli.shape[0]
    patch_size = np.random.randint(low=patch_size_low, high=patch_size_high)
    crop_low_xy = 32
    crop_high_x = 64
    crop_high_y = 128
    crop_x = np.random.randint(low=crop_low_xy, high=crop_high_x)
    crop_y = np.random.randint(low=crop_low_xy, high=crop_high_y)
    sli_gibbs_cutart = sli.copy()
    sli_gibbs_cutart[
    crop_x: crop_x + patch_size, crop_y: crop_y + patch_size
    ] = sli_gibbs[crop_x: crop_x + patch_size, crop_y: crop_y + patch_size]

    # Debug - visualize crop
    # sli_gibbs_cutart[crop_x, crop_y:crop_y + patch_size] = 1
    # sli_gibbs_cutart[crop_x + patch_size, crop_y:crop_y + patch_size] = 1
    # sli_gibbs_cutart[crop_x:crop_x + patch_size, crop_y] = 1
    # sli_gibbs_cutart[crop_x:crop_x + patch_size, crop_y + patch_size] = 1

    return sli_gibbs_cutart


def main(path_read_data: Path, path_save_data: Path, slice_size: int):
    # =========
    # PATHS
    # =========
    path_nii_files = glob_nifti(path=path_read_data)

    path_save_data = Path(path_save_data) / "gibbs"
    if not path_save_data.exists():
        path_save_data.mkdir(parents=True, exist_ok=True)

    # =========
    # DATAGEN
    # =========
    for path_nii in tqdm(path_nii_files):
        vol = data_loader.load_data(
            path_data=path_nii,
            data_format="nifti",
            normalize=True,
            nifti_dataset="HCP-T1",
            target_size=slice_size,
        )

        vol_gibbs = []
        for ind in range(vol.shape[-1]):
            sli = vol[..., ind]

            # Chop in k-space
            kspace = np.fft.fftshift(np.fft.fftn(sli))  # K-space
            num_pe_chop = np.random.randint(
                low=64, high=96
            )  # Number of phase encodes to chop
            pe_chop_start = np.random.randint(
                low=0, high=160
            )  # Start index of phase encode chop
            pe_chop_stop = (
                    pe_chop_start + num_pe_chop
            )  # Stop index of phase encode chop
            idx = list(range(0, sli.shape[0], 12))  # In skips of 12
            for i in idx:
                if abs(i - sli.shape[0] // 2) > 32:
                    kspace[
                    pe_chop_start:pe_chop_stop, i: i + 8
                    ] = 0  # Every 12 rows, chop 8 rows
            sli_gibbs = np.abs(np.fft.ifftn(np.fft.ifftshift(kspace)))
            sli_gibbs = cutart(sli, sli_gibbs)
            vol_gibbs.append(sli_gibbs)

            # Scaling
            # diff = sli_gibbs - sli
            # sli_gibbs_scaled = sli_gibbs + (1.5 * diff)

            # Save to disk
            # suffix = '.nii.gz' if '.nii.gz' in f.name else '.nii'
            # subject = f.name.replace(suffix, '') + f'_slice{ind}.npy'
            # _path_save = path_save_data.joinpath(subject)
            # np.save(arr=sli_gibbs_normalized, file=str(_path_save))
        vol_gibbs = np.stack(vol_gibbs, axis=-1)

        # Debug - visualize
        import sass

        sass.scroll(vol, vol_gibbs, vol-vol_gibbs)
