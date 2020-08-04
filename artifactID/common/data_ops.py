import math
from pathlib import Path
from warnings import warn

import nibabel as nb
import numpy as np
import pydicom as pyd
from skimage.util.shape import view_as_blocks
from tqdm import tqdm


def dcmfolder2npy(path: Path, verbose: bool = True):
    if not isinstance(path, Path):
        path = Path(path)

    arr_dcm = list(path.glob('*'))  # List all DICOM files
    arr_npy = []
    for dicom in tqdm(arr_dcm, disable=not verbose):
        dcm = pyd.dcmread(str(dicom))
        npy = dcm.pixel_array
        arr_npy.append(npy)

    try:
        arr_npy = np.stack(arr_npy)
        arr_npy = np.moveaxis(arr_npy, [0, 1, 2], [2, 0, 1])  # Scroll through slices along last dimension
        return arr_npy
    except:
        pass


def get_patches(vol: np.ndarray, patch_size: int):
    if not isinstance(patch_size, list):
        patch_size = [patch_size] * 3

    # Check shape compatibility
    shape = vol.shape
    for i in range(len(shape)):
        s = shape[i]
        p = patch_size[i]
        if s % p != 0:
            raise Exception(f'Incompatible shapes: {shape} and {p}')

    patches = view_as_blocks(arr_in=vol, block_shape=tuple(patch_size))
    original_shape = patches.shape
    patches = patches.reshape((-1,) + tuple(patch_size))

    patch_map = []
    pruned_patches = []
    for p in patches:
        if np.count_nonzero(p) == 0 or p.max() == p.min():  # Invalid patch, discard
            patch_map.append(0)
        else:  # Valid patch
            pruned_patches.append(p)
            patch_map.append(1)

    patch_map = np.array(patch_map).reshape(original_shape[:3])
    return np.array(pruned_patches).astype(np.float16), patch_map


def get_patch_size_from_config(patch_size):
    try:
        patch_size = int(patch_size)
    except:  # patch_size is not an int, must be a tuple
        patch_size = patch_size.split(',')
        patch_size = list(map(lambda to_int: int(to_int), patch_size))
        if len(patch_size) > 3:
            raise ValueError(f'patch_size should either be an int or a list of length 3, you passed {patch_size}')
    return patch_size


def get_paths_labels(data_root: str, filter_artifact: str):
    # Construct `x` and `y` training pairs
    if filter_artifact in ['b0', 'gibbs', 'nrm', 'rot', 'snr', 'wrap']:
        glob_pattern = filter_artifact + '*'
    else:
        warning = f'Unknown value for filter_artifact. Valid values are b0, snr, wrap and rot.'
        warning += f'You passed: {filter_artifact}. Globbing all data.'
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
            label = label.rstrip('0123456789').rstrip('-_')
        y_labels.extend([label] * len(files))

    return np.array(x_paths), np.array(y_labels)


def get_paths(data_root: str):
    files = list(Path(data_root).glob('**/*'))
    files = list(map(lambda x: str(x), files))  # Convert from Path to str
    return np.array(files)


def glob_nifti(path: str):
    path = Path(path)
    arr_path = list(path.glob('**/*.nii.gz'))
    arr_path2 = list(path.glob('**/*.nii'))
    return arr_path + arr_path2


def glob_brats_t1(path_brats: str):
    path_brats = Path(path_brats)
    arr_path_brats_t1 = list(path_brats.glob('**/*.nii.gz'))
    arr_path_brats_t1 = list(filter(lambda x: 't1.nii' in str(x), arr_path_brats_t1))
    return arr_path_brats_t1


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


def make_generator_train(x, y):
    """
    Training generator indefinitely yielding volumes loaded from .npy files specified in `x`. Also yields paired labels
    from `y`. Every `while` loop iteration counts as one epoch. The data are shuffled at the start of every epoch.

    Parameters
    ==========
    x : array-like
        Array of paths to .npy files to load.
    y : array-like, optional
        Array of labels corresponding to the .npy files to be loaded from `x`.

    Yields
    ======
    _x : np.ndarray
        Array containing a single volume of shape (..., 1) and datatype np.float16.
    _y : np.ndarray
        Array containing a single corresponding label to the volume yielded in `_x` of datatype np.int8.
    """
    while True:
        # Shuffle at the start of every epoch
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        x = x[idx]
        y = y[idx]

        for counter in range(len(x)):
            _x = np.load(x[counter])  # Load volume
            _x = np.expand_dims(_x, axis=3)  # Convert shape to (..., 1)
            _x = _x.astype(np.float16)  # Mixed precision

            _y = np.array([y[counter]]).astype(np.int8)
            yield _x, _y


def make_generator_inference(x):
    """
    Inference generator yielding volumes loaded from .npy files specified in `x`.

    Parameters
    ==========
    x : array-like
        Array of paths to .npy files to load.

    Yields
    ======
    _x : np.ndarray
        Array containing a single volume of shape (..., 1) and datatype np.float16.
    """
    for counter in range(len(x)):
        _x = np.load(x[counter])  # Load volume
        _x = np.expand_dims(_x, axis=3)  # Convert shape to (x, y, z 1)
        yield _x


def normalize_patches(patches):
    arr_patches = []
    for p in patches:
        _max = p.max()
        _min = p.min()
        p = (p - _min) / (_max - _min)
        arr_patches.append(p)
    return arr_patches


def patch_compatible_zeropad(vol, patch_size):
    if not isinstance(patch_size, list):
        patch_size = [patch_size] * 3

    pad = []
    shape = vol.shape
    for i in range(len(shape)):
        s = shape[i]
        p = patch_size[i]

        if s < p or s % p != 0:
            _pad = p - (s % p)
            pad.append((math.floor(_pad / 2), math.ceil(_pad / 2)))
        else:
            pad.append((0, 0))
    return np.pad(array=vol, pad_width=pad)
