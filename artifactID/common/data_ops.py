import math
from pathlib import Path
from warnings import warn

import cv2
import nibabel as nb
import numpy as np
from skimage.util.shape import view_as_blocks


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


def get_patches(arr: np.ndarray, patch_size):
    shape = arr.shape
    # Convert patch_size to a tuple if necessary
    if isinstance(patch_size, int):
        patch_size = [patch_size] * len(shape)
    patch_size = tuple(patch_size)

    # Check shape compatibility
    for counter, p in enumerate(patch_size):
        if shape[counter] % p != 0:
            raise Exception(f'Incompatible shapes: {shape} and {patch_size}')

    return view_as_blocks(arr_in=arr, block_shape=patch_size)


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


def get_paths(data_root: str):
    files = list(Path(data_root).glob('**/*'))
    files = list(map(lambda x: str(x), files))  # Convert from Path to str
    return np.array(files)


def make_generator_train(x, y=None):
    """
    Generator for training that yields volumes loaded from .npy files specified in `x`. Also yields paired labels from
    `y`, if required. The data are shuffled at the start of every epoch.

    Parameters
    ==========
    x : array-like
        Array of paths to .npy files to load.
    y : array-like, optional
        Array of labels corresponding to the .npy files to be loaded from `x`.\
        If `None`, this generator only loads and yields .npy files.
    """
    while True:
        # Shuffle at the start of every epoch
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        x = x[idx]
        if y is not None:
            y = y[idx]

        for counter in range(len(x)):
            _x = np.load(x[counter])  # Load volume
            _x = np.expand_dims(_x, axis=3)  # Convert shape to (x, y, z 1)
            _x = _x.astype(np.float16)  # Mixed precision

            if y is not None:
                _y = np.array([y[counter]]).astype(np.int8)
                yield _x, _y
            else:
                yield _x


def make_generator_inference(x):
    while True:
        for counter in range(len(x)):
            _x = np.load(x[counter])  # Load volume

            # Resize each slice to 240, 240
            _x_resized = []
            for i in range(_x.shape[2]):  # Shape is (height, width, slices)
                sli = _x[:, :, i]
                sli_resized = cv2.resize(sli, (240, 240))
                _x_resized.append(sli_resized)
            _x = np.stack(_x_resized)  # Slices will be first dimension

            # Pad/crop to 155 slices accordingly
            num_slices = _x.shape[0]  # Shape is (slices, 240, 240)
            if num_slices < 155:  # Pad to 155 slices
                pad_width = (155 - _x.shape[0]) / 2
                if isinstance(pad_width, float):  # pad_width is not an integer, adjust leading and trailing pad widths
                    pad_width = (math.floor(pad_width), math.ceil(pad_width))
                else:  # pad_width is an integer
                    pad_width = (pad_width, pad_width)
                _x = np.pad(_x, (pad_width, (0, 0), (0, 0)))
            elif num_slices > 155:  # Center-crop to 155 slices
                center_slice = _x.shape[0] // 2
                _x = _x[:, :, center_slice - 77:center_slice + 78]
            _x = np.moveaxis(_x, [0, 1, 2], [2, 0, 1])  # Scroll through slices along last dimension
            _x = np.expand_dims(_x, axis=3)  # Convert shape to (240, 240, 155, 1)
            _x = _x.astype(np.float16)  # Mixed precision TF2
            yield _x
