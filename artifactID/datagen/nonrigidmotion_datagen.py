import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from artifactID.common import data_ops


def main(path_read_data: Path, path_save_data: Path, patch_size: int):
    # =========
    # PATHS
    # =========
    if 'miccai' in str(path_read_data).lower():
        arr_path_read = data_ops.glob_brats_t1(path_brats=path_read_data)
    else:
        arr_path_read = data_ops.glob_nifti(path=path_read_data)
    path_save_data = Path(path_save_data)

    # =========
    # DATAGEN
    # =========
    for ind, path_t1 in tqdm(enumerate(arr_path_read)):

        vol = data_ops.load_nifti_vol(path=path_t1)
        vol = vol.astype(np.float32)  # Convert back to float 32 for CV/PIL compatibility
        N = vol.shape[0]
        vol_norm = np.zeros(vol.shape)
        vol_norm = cv2.normalize(vol, vol_norm, 0, 255, cv2.NORM_MINMAX)

        rot = 0
        trans = 0
        while abs(rot) < 5 or abs(trans) < 5:
            rot = np.random.randint(-10, 11)
            trans = np.random.randint(-10, 11)

        vol_rot = np.zeros(vol.shape)
        for sl in range(vol.shape[-1]):
            slice = Image.fromarray(vol_norm[:, :, sl])
            slice_rot = slice.rotate(rot)
            vol_rot[:, :, sl] = slice_rot

        trans_direction = np.random.choice([0, 1])
        vol_trans = np.roll(vol_norm, trans, axis=trans_direction)

        kdat_orig = np.fft.fftshift(np.fft.fftn(vol_norm))
        kdat_rot = np.fft.fftshift(np.fft.fftn(vol_rot))
        kdat_trans = np.fft.fftshift(np.fft.fftn(vol_trans))
        kdat_nrm = np.fft.fftshift(np.fft.fftn(vol_norm))

        kspace_center = math.ceil(N * 0.04)
        center_lines = range(int(N / 2) - kspace_center, int(N / 2) + kspace_center)
        random_lines = np.random.randint(0, N, int((N - 2 * kspace_center) / 2))
        if trans_direction == 0:
            # TODO: remove hard coded numbers
            kdat_nrm[random_lines[:int(len(random_lines) / 2)], :, :] = kdat_rot[
                                                                        random_lines[:int(len(random_lines) / 2)], :, :]
            kdat_nrm[random_lines[int(len(random_lines) / 2):], :, :] = kdat_trans[
                                                                        random_lines[int(len(random_lines) / 2):], :, :]
            kdat_nrm[center_lines, :, :] = kdat_orig[center_lines, :, :]
        else:
            kdat_nrm[:, random_lines[:int(len(random_lines) / 2)], :] = kdat_rot[:,
                                                                        random_lines[:int(len(random_lines) / 2)], :]
            kdat_nrm[:, random_lines[int(len(random_lines) / 2):], :] = kdat_trans[:,
                                                                        random_lines[int(len(random_lines) / 2):], :]
            kdat_nrm[:, center_lines, :] = kdat_orig[:, center_lines, :]

        vol_nrm = np.abs(np.fft.ifftn(np.fft.ifftshift(kdat_nrm)))
        # Convert to float16 to avoid dividing by 0 during normalization - very low max values get zeroed out
        vol_nrm = vol_nrm.astype(np.float16)

        # Zero-pad vol, get patches, discard empty patches and uniformly intense patches and normalize each patch
        vol_nrm = data_ops.patch_size_compatible_zeropad(vol=vol_nrm, patch_size=patch_size)
        patches, _ = data_ops.get_patches_per_slice(vol=vol_nrm, patch_size=patch_size)
        if len(patches) > 0:
            patches = data_ops.normalize_slices(patches=patches)

            # Save to disk
            _path_save = path_save_data.joinpath(f'nrm')
            if not _path_save.exists():
                _path_save.mkdir(parents=True)
            for counter, p in enumerate(patches):
                subject = path_t1.name.replace('.nii.gz', '')
                _path_save2 = _path_save.joinpath(subject)
                _path_save2 = str(_path_save2) + f'__patch{counter}.npy'
                np.save(arr=p, file=_path_save2)
