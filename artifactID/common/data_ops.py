import math
from pathlib import Path

import nibabel as nb
import numpy as np
import pydicom as pyd
from skimage.util import view_as_windows
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


def get_patches_per_slice(vol: np.ndarray, patch_size: int):
    # Check shape compatibility
    shape = vol.shape
    if shape[0] % patch_size != 0 or shape[1] % patch_size != 0:
        raise Exception(f'Incompatible shapes: {shape} and {patch_size}')

    vol = vol.astype(np.float16)
    patches = []
    for i in range(vol.shape[-1]):
        _sli = vol[..., i]
        _patches = view_as_windows(arr_in=_sli, window_shape=patch_size, step=patch_size)
        _patches = _patches.reshape((-1, patch_size, patch_size))
        for p in _patches:
            if np.count_nonzero(p) >= 0.75 * p.size and p.max() != p.min():  # Valid patch
                patches.append(p)
    return np.array(patches, dtype=np.float16)


def get_y_labels_unique(data_root: str):
    y_labels_unique = []
    for artifact_folder in Path(data_root).glob('*'):
        label = artifact_folder.name
        label = label.rstrip('0123456789').rstrip('-_')
        y_labels_unique.append(label)

    return np.unique(y_labels_unique)


def glob_dicom(path: Path):
    arr_path = list(path.glob('**'))
    return arr_path
    return np.array(files)


def glob_nifti(path: Path):
    path = Path(path)
    arr_path = list(path.glob('**/*.nii.gz'))
    arr_path2 = list(path.glob('**/*.nii'))
    return arr_path + arr_path2


def glob_brats_t1(path_brats: Path):
    path_brats = Path(path_brats)
    arr_path_brats_t1 = list(path_brats.glob('**/*.nii.gz'))
    arr_path_brats_t1 = list(filter(lambda x: 't1.nii' in str(x), arr_path_brats_t1))
    return arr_path_brats_t1


def load_nifti_vol(path: Path):
    """
    Read NIFTI file at `path` and return an 3D numpy.ndarray. Keep only slices having 5% or more signal. Ensure correct
    orientation of the brain.

    Parameters
    ==========
    path : str
        Path to NIFTI file to be read.

    Returns
    =======
    numpy.ndarray
        Numpy data of NIFTI file at `path`.
    """
    vol = nb.load(str(path)).get_fdata().astype(np.float16)
    vol = np.rot90(vol, -1, axes=(0, 1))  # BraTS: Ensure brain is oriented facing up
    vol = (vol - vol.min()) / (vol.max() - vol.min())  # Normalize between 0-1
    return vol


def make_generator_train(path_train, dict_label_int):
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
    dict_label_int = eval(dict_label_int)
    for path in path_train:
        try:
            path = Path(path.decode().strip())
            _x = np.load(str(path))  # Load volume
            _x = np.expand_dims(_x, axis=2)  # Convert shape to (..., 1)
            _x = _x.astype(np.float16)  # Mixed precision

            # y integer label
            label = path.parent.name
            label = label.rstrip('0123456789').rstrip('-_')
            _y = np.array([dict_label_int[label]]).astype(np.int8)
            yield {'input_1': _x, 'input_2': _x}, _y
        except ValueError:
            print(path)


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
    _max = np.max(patches, axis=(1, 2))
    _min = np.min(patches, axis=(1, 2))
    m3 = _min.reshape((-1, 1, 1))
    _range = (_max - _min).reshape((-1, 1, 1))
    patches = (patches - m3) / _range
    return patches


def patch_size_compatible_zeropad(vol: np.ndarray, patch_size: int):
    pad = []
    shape = vol.shape
    for s in shape[:2]:  # Pad per slice, not across volume
        if s < patch_size or s % patch_size != 0:
            p = patch_size - (s % patch_size)
            pad.append((math.floor(p / 2), math.ceil(p / 2)))
        else:
            pad.append((0, 0))
    pad.append((0, 0))  # Do not pad along last dimension
    return np.pad(array=vol, pad_width=pad)
