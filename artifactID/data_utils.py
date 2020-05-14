import itertools
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
    slice_content = lambda x: np.count_nonzero(x) > 0.05 * x.size
    slice_content_idx = [slice_content(vol[:, :, i]) for i in
                         range(vol.shape[2])]  # Get indices of slices with >=5% signal
    vol = vol[:, :, slice_content_idx]
    vol = (vol - vol.min()) / (vol.max() - vol.min())  # Normalize between 0-1
    return vol


def shuffle_dataset(x, y):
    assert len(x) == len(y)
    random_idx = np.random.randint(len(x), size=len(x))
    x = np.take(x, random_idx)
    y = np.take(y, random_idx)
    return x, y


def data_generator(x_paths, y_labels, mode):
    # Convert from byte string to string
    x_paths = x_paths.astype(np.str)
    y_labels = y_labels.astype(np.str)
    mode = mode.decode('utf-8')

    # Construct dictionary to encode labels as integers
    unique_labels = np.unique(y_labels)
    dict_labels_encoded = dict(zip(unique_labels, itertools.count(0)))

    while True:
        for counter in range(len(x_paths)):
            x = np.load(x_paths[counter])  # Load volume
            x = np.expand_dims(x, axis=3).astype(np.float16)  # Convert shape to (240, 240, 155, 1), mixed precision

            label = y_labels[counter]  # Get label
            y = np.array([dict_labels_encoded[label]]).astype(np.int8)  # Encoded label

            yield x, y
        if mode == 'train':
            # Shuffle after each epoch during training/validation
            x_paths, y_labels = shuffle_dataset(x=x_paths, y=y_labels)
        elif mode == 'eval':
            break
        else:
            raise ValueError(f'Unknown mode. Valid values are train and eval, you passed: {mode}')
