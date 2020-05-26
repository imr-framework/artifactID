from pathlib import Path
from warnings import warn

import nibabel as nb
import numpy as np


def glob_brats_t1(path_brats: str):
    path_brats = Path(path_brats)
    arr_paths_brats_t1 = list(path_brats.glob('**/*.nii.gz'))
    arr_paths_brats_t1 = list(filter(lambda x: 't1.nii' in str(x), arr_paths_brats_t1))
    return arr_paths_brats_t1


def load_nifti_vol(path: str):
    """
    Read NIFTI file at `path` and return an 3D numpy.ndarray. Keep only slices having 5% or more signal.

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
    filter_5pc = lambda x: np.count_nonzero(x) > 0.05 * x.size
    slice_content_idx = [filter_5pc(vol[:, :, i]) for i in
                         range(vol.shape[2])]  # Get indices of slices with >=5% signal
    vol = vol[:, :, slice_content_idx]
    vol = (vol - vol.min()) / (vol.max() - vol.min())  # Normalize between 0-1
    return vol


def get_paths_labels(data_root: str, filter_artifact: str):
    # Construct `x` and `y` training pairs
    if filter_artifact in ['b0', 'snr', 'wrap']:
        glob_pattern = filter_artifact + '*'
    else:
        warning = f'Unknown value for filter_artifact. Valid values are b0, snr and wrap. ' \
                  f'You passed: {filter_artifact}. Globbing all data.'
        warn(warning)
        glob_pattern = '*'

    x_paths = []
    y_labels = []
    for artifact_folder in Path(data_root).glob(glob_pattern):
        files = list(artifact_folder.glob('*.npy'))
        files = list(map(lambda x: str(x), files))  # Convert from Path to str
        x_paths.extend(files)
        label = artifact_folder.name
        if glob_pattern == '*':
            label = label.rstrip('0123456789')
        y_labels.extend([label] * len(files))

    return np.array(x_paths), np.array(y_labels)


def make_generator(x, y=None):
    while True:
        for counter in range(len(x)):
            _x = np.load(x[counter])  # Load volume
            _x = np.expand_dims(_x, axis=3)  # Convert shape to (240, 240, 155, 1)
            _x = _x.astype(np.float16)  # Mixed precision

            if y is not None:
                _y = np.array([y[counter]]).astype(np.int8)
                yield _x, _y
            else:
                yield _x
