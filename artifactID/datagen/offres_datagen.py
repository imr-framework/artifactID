import math
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import scipy.io as sio

from artifactID.utils import glob_brats_t1
from artifactID.datagen import generate_fieldmap
from orc import ORC


def _preprocess_imvol(vol):
    """
    Preprocesses (Rotates and normalizes) a 3D image volume

    Parameters
    ==========
    vol : numpy.ndarray
        Image volume having dimensions N x N x N number of slices

    Returns
    =======
    vol_pp : numpy.ndarray
        Image volume after preprocessing N x N x N number of slices intensity range [0, 1]
    """
    vol = np.rot90(vol, -1, axes=(0, 1))  # Rotation
    vol_pp = (vol - vol.min()) / (vol.max() - vol.min())  # Normalization [0, 1]

    return vol_pp


def _gen_fieldmap(_slice, _freq_range, _ktraj, _seq_params):
    field_map, mask = generate_fieldmap.gen_smooth_b0(_slice, _freq_range)  # Simulate the field map
    or_corrupted, _ = ORC.add_or_CPR(_slice, _ktraj, field_map, nonCart=1,
                                     params=_seq_params)  # Corrupt the image
    or_corrupted_norm = np.zeros(or_corrupted.shape)
    or_corrupted_norm = cv2.normalize(np.abs(or_corrupted), or_corrupted_norm, 0, 1,
                                      cv2.NORM_MINMAX)  # Normalize [0, 1]
    return np.float32(or_corrupted_norm * mask)


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
    arr_offres_vol = []

    for ind in range(num_slices):
        slice = vol[:, :, ind]
        if np.count_nonzero(slice) > 0.05 * slice.size:  # Check if at least 5% of signal is present
            res = _gen_fieldmap(_slice=slice, _freq_range=freq_range, _ktraj=ktraj, _seq_params=seq_params)
            arr_offres_vol.append(res)
    arr_offres_vol = np.stack(arr_offres_vol)
    return arr_offres_vol


def main(path_brats: str, path_save: str, path_ktraj: str, path_dcf: str):
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
    arr_path_brats_t1 = glob_brats_t1(path_brats=path_brats)
    num_subjects = len(arr_path_brats_t1)

    arr_max_freq = [250, 500, 750]  # Hz
    subjects_per_class = math.ceil(len(arr_path_brats_t1) / len(arr_max_freq))  # Calculate number of subjects per class
    arr_max_freq *= subjects_per_class
    np.random.shuffle(arr_max_freq)

    path_save = Path(path_save)
    path_all = [path_save / f'b0_{freq}' for freq in arr_max_freq]
    # Make save folders if they do not exist
    for p in path_all:
        if not p.exists():
            p.mkdir(parents=True)

    # =========
    # DATAGEN
    # =========
    for ind, path_t1 in enumerate(arr_path_brats_t1):
        pc = round((ind + 1) / num_subjects * 100, ndigits=2)
        print(f'{pc}%', end=', ', flush=True)

        vol = nib.load(path_t1).get_data()
        vol_pp = _preprocess_imvol(vol=vol)
        freq = arr_max_freq[ind]
        vol_b0 = orc_forwardmodel(vol=vol_pp, freq_range=freq, ktraj=ktraj, seq_params=seq_params)
        vol_b0 = np.moveaxis(vol_b0, [0, 1, 2], [1, 2, 0])  # Iterate through slices on the last dim

        subject_name = path_t1.parts[-1].split('.nii.gz')[0]  # Extract subject name from path
        _path_save = str(path_save / f'b0_{freq}' / subject_name) + '.npy'
        np.save(arr=vol_b0, file=_path_save)
        # print('Corrupted data saved for subject:' + folder_name)
