import math
from pathlib import Path

import nibabel as nb
import numpy as np


def glob_brats_t1(path_brats: str):
    path_brats = Path(path_brats)
    arr_paths_brats_t1 = list(path_brats.glob('**/*.nii.gz'))
    arr_paths_brats_t1 = list(filter(lambda x: 't1.nii' in str(x), arr_paths_brats_t1))
    return arr_paths_brats_t1


def load_nifti_vol(path: str):
    """
    Read NIFTI file at `path` and return an 3D numpy.ndarray. Keep only slices having 5% or more signal, and zero-pad
    the rest.

    1. Ensure correct orientation of the brain
    2. Normalize between 0-1

    Parameters
    ==========
    path : str
        Path to NIFTI file to be read.

    Returns
    =======
    numpy.ndarray
        3D array of NIFTI file at `path`.
    """
    vol = nb.load(path).get_fdata()
    vol = np.rot90(vol, -1, axes=(0, 1))  # Ensure correct orientation
    orig_num_slices = vol.shape[2]
    slice_content = lambda x: np.count_nonzero(x) > 0.05 * x.size
    slice_content_idx = [slice_content(vol[:, :, i]) for i in
                         range(vol.shape[2])]  # Get indices of slices with >=5% signal
    vol = vol[:, :, slice_content_idx]
    n_zeros = (orig_num_slices - vol.shape[2]) / 2
    n_zeros = [math.floor(n_zeros), math.ceil(n_zeros)]
    vol = np.pad(vol, [[0, 0], [0, 0], n_zeros])
    vol = (vol - vol.min()) / (vol.max() - vol.min())  # Normalize between 0-1
    return vol
