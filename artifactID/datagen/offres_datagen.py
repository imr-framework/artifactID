import math
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
from OCTOPUS import ORC
from tqdm import tqdm

from artifactID.common import data_ops
from artifactID.datagen import generate_fieldmap


def _gen_fieldmap(_slice, _freq_range):
    field_map, mask = generate_fieldmap.gen_smooth_b0(_slice, _freq_range)  # Simulate the field map

    return field_map, mask


def orc_forwardmodel(vol: np.ndarray, freq_range: int, ktraj: np.ndarray, seq_params: dict):
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

    seq_params['N'] = vol.shape[0]
    num_slices = vol.shape[2]
    arr_offres_vol = np.zeros(vol.shape)

    for ind in range(num_slices):
        slice = vol[:, :, ind]
        fieldmap, mask = _gen_fieldmap(_slice=slice, _freq_range=freq_range)
        or_corrupted = ORC.add_or_CPR(slice, ktraj, fieldmap, 1, seq_params)  # Corrupt the image
        or_corrupted_norm = np.zeros(or_corrupted.shape)
        or_corrupted_norm = cv2.normalize(np.abs(or_corrupted), or_corrupted_norm, 0, 1,
                                          cv2.NORM_MINMAX)  # Normalize [0, 1]
        arr_offres_vol[:, :, ind] = np.float16(or_corrupted_norm * mask)

    return arr_offres_vol


def main(path_read_data: Path, path_save_data: Path, path_ktraj: str, path_dcf: str, patch_size: list):
    # =========
    # LOAD PREREQUISITES
    # =========
    # 1. k-space trajectory
    ktraj = np.load(path_ktraj)
    ktraj_sc = math.pi / abs(np.max(ktraj))
    ktraj = ktraj * ktraj_sc  # pyNUFFT scaling [-pi, pi]

    # 2. Density compensation factor
    dcf = sio.loadmat(path_dcf)['dcf_out'].flatten()

    # 3. Acquisition parameters
    t_vector = (np.arange(ktraj.shape[0]) * 10e-6).reshape(ktraj.shape[0], 1)
    seq_params = {'Npoints': ktraj.shape[0], 'Nshots': ktraj.shape[1], 't_vector': t_vector, 'dcf': dcf}

    # BraTS 2018 paths
    arr_path_read = data_ops.glob_nifti(path=path_read_data)

    arr_max_freq = [250, 500, 750]  # Hz
    subjects_per_class = math.ceil(len(arr_path_read) / len(arr_max_freq))  # Calculate number of subjects per class
    arr_max_freq *= subjects_per_class
    np.random.shuffle(arr_max_freq)

    # =========
    # DATAGEN
    # =========
    for ind, path_t1 in tqdm(enumerate(arr_path_read)):
        vol = data_ops.load_nifti_vol(path=path_t1)
        freq = arr_max_freq[ind]
        vol_b0 = orc_forwardmodel(vol=vol, freq_range=freq, ktraj=ktraj, seq_params=seq_params)

        # Zero-pad vol, get patches, discard empty patches and uniformly intense patches and normalize each patch
        vol_b0 = data_ops.patch_compatible_zeropad(vol=vol_b0, patch_size=patch_size)
        patches, patch_map = data_ops.get_patches(vol=vol_b0, patch_size=patch_size)
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
