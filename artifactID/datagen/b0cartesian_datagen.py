import math
from pathlib import Path

import cv2
import numpy as np
from OCTOPUS.Recon import ORC
from tqdm import tqdm

from artifactID.common import data_ops
from artifactID.datagen import generate_fieldmap


def _gen_fieldmap(_slice, _freq_range):
    field_map = generate_fieldmap.hyperbolic(_slice.shape[0], _freq_range)  # Simulate the field map

    return field_map


def orc_forwardmodel(vol: np.ndarray, freq_range: int, ktraj: np.ndarray):
    """
    Adds off-resonance blurring to simulate B0 inhomogeneity artifacts.

    Parameters
    ==========
    vol : numpy.ndarray
        Image volume having dimensions N x N x N number of slices
    freq_range : int
        Frequency range for the simulated field map
    ktraj : numpy.ndarray
        k-space trajectory coordinates. Dimensions Npoints x Nshots
    seq_params : dict
        Sequence parameters needed for off-resonance corruption

    Returns
    =======
    arr_offres_vol : np.ndarray
        Corrupted image volume having dimensions Slices(with signal > 5%) x N x N
    """

    num_slices = vol.shape[2]
    arr_offres_vol = np.zeros(vol.shape)

    for ind in range(num_slices):
        slice = vol[:, :, ind]

        fieldmap = _gen_fieldmap(_slice=slice, _freq_range=freq_range)
        or_corrupted = ORC.add_or_CPR(slice, ktraj, fieldmap)  # Corrupt the image

        or_corrupted_norm = np.zeros(or_corrupted.shape)
        or_corrupted_norm = cv2.normalize(np.abs(or_corrupted), or_corrupted_norm, 0, 1,
                                          cv2.NORM_MINMAX)  # Normalize [0, 1]
        arr_offres_vol[:, :, ind] = np.float16(or_corrupted_norm)

    # arr_offres_vol = np.stack(arr_offres_vol)
    return arr_offres_vol


def main(path_read_data: str, path_save_data: str, patch_size: int):
    # =========
    # LOAD PREREQUISITES
    # =========
    # 1. k-space trajectory
    dt = 10e-6  # grad raster time

    # BraTS 2018 paths
    if 'miccai' in path_read_data.lower():
        arr_path_read = data_ops.glob_brats_t1(path_brats=path_read_data)
    else:
        arr_path_read = data_ops.glob_nifti(path=path_read_data)
    path_save_data = Path(path_save_data)

    arr_max_freq = [2500, 5000, 7500]  # Hz
    subjects_per_class = math.ceil(len(arr_path_read) / len(arr_max_freq))  # Calculate number of subjects per class
    arr_max_freq *= subjects_per_class
    np.random.shuffle(arr_max_freq)

    '''path_save = Path(path_save)
    path_all = [path_save / f'b0_{freq}' for freq in arr_max_freq]
    # Make save folders if they do not exist
    for p in path_all:
        if not p.exists():
            p.mkdir(parents=True)'''

    # =========
    # DATAGEN
    # =========
    for ind, path_t1 in tqdm(enumerate(arr_path_read)):

        vol = data_ops.load_nifti_vol(path=path_t1)

        N = vol.shape[0]
        ktraj_cart = np.arange(0, N * dt, dt).reshape(1, N)
        ktraj = np.tile(ktraj_cart, (N, 1))

        freq = arr_max_freq[ind]
        vol_b0 = orc_forwardmodel(vol=vol, freq_range=freq, ktraj=ktraj)

        # Zero-pad vol, get patches, discard empty patches and uniformly intense patches and normalize each patch
        vol_b0 = data_ops.patch_compatible_zeropad(vol=vol_b0, patch_size=patch_size)
        patches = data_ops.get_patches(arr=vol_b0, patch_size=patch_size)
        patches, patch_map = data_ops.prune_patches(patches=patches)
        patches = data_ops.normalize_patches(patches=patches)

        # Save to disk
        _path_save = path_save_data.joinpath(f'b0{freq}')
        if not _path_save.exists():
            _path_save.mkdir(parents=True)
        for counter, p in enumerate(patches):
            subject = path_t1.name.replace('.nii.gz', '')
            _path_save2 = _path_save.joinpath(subject)
            _path_save2 = str(_path_save2) + f'_patch{counter}.npy'
            np.save(arr=p, file=_path_save2)
