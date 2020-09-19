import math
from pathlib import Path

import nibabel as nb
import numpy as np
import pydicom as pyd
from skimage.util import view_as_windows
from tqdm import tqdm


def __extract_brain(vol: np.ndarray):
    """
    Masks a magnitude image by thresholding according to: Jenkinson M. (2003). Fast, automated, N-dimensional
    phase-unwrapping algorithm. Magnetic resonance in medicine, 49(1), 193–197. https://doi.org/10.1002/mrm.10354

    Parameters
    ==========
    vol : np.ndarray
        Input brain volume

    Returns
    =======
    vol_masked : np.ndarray
           Brain segmented volume
    """
    vol_masked = np.zeros_like(vol)
    threshold = 0.1 * (np.percentile(vol, 98, axis=(0, 1)) - np.percentile(vol, 2, axis=(0, 1))) \
                + np.percentile(vol, 2, axis=(0, 1))
    idx = np.where(vol >= threshold)
    vol_masked[idx] = vol[idx]
    return vol_masked


def dcmfolder2npy(path: Path, verbose: bool = True):
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
    patches = view_as_windows(arr_in=vol, window_shape=[patch_size, patch_size, 1], step=[patch_size, patch_size, 1])
    pruned_patches = []
    patch_map = np.zeros(patches.shape[:3], dtype=np.int8)
    np_index = np.ndindex(patches.shape[:3])
    for multi_index in np_index:
        p = patches[multi_index]
        p = np.squeeze(p)
        if np.count_nonzero(p) >= 0.5 * p.size and p.max() != p.min():  # Valid patch
            pruned_patches.append(p)
            patch_map[multi_index] = 1
    return np.array(pruned_patches, dtype=np.float16), patch_map


def get_y_labels_unique(data_root: Path):
    """

    Parameters
    ==========
    data_root : Path


    Returns
    =======
    np.ndarray

    """
    y_labels = []
    for artifact_folder in data_root.glob('*'):
        if artifact_folder.is_dir():
            label = artifact_folder.name
            label = label.rstrip('0123456789').rstrip('-_')
            y_labels.append(label)

    return np.unique(y_labels)


def glob_dicom(path: Path):
    arr_path = list(path.glob('**'))
    return arr_path


def glob_nifti(path: Path):
    arr_path = list(path.glob('**/*.nii.gz'))
    arr_path2 = list(path.glob('**/*.nii'))
    return arr_path + arr_path2


def glob_brats_t1(path_brats: Path):
    arr_path_brats_t1 = list(path_brats.glob('**/*.nii.gz'))
    arr_path_brats_t1 = list(filter(lambda x: 't1.nii' in str(x), arr_path_brats_t1))
    return arr_path_brats_t1


def load_nifti_vol(path: Path):
    """
    Read NIFTI file at `path` and return an 3D numpy.ndarray. Ensure correct orientation of the brain.

    Parameters
    ==========
    path : Path
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


def generator_train(x, dict_label_int):
    """
    Training generator yielding volumes loaded from paths to .npy files in `x`. Also yields paired labels `y`. Labels
    are derived from the paths to the .npy files in `x`.

    Parameters
    ==========
    x : array-like
        Array of paths to .npy files to load.
    dict_label_int : str
        Dictionary mapping labels to integers. Used to derive `y` labels for each `x` input to model.

    Yields
    ======
    dict{'input_1': np.ndarray, 'input_2': np.ndarray}
        Dict of two volumes of shape (..., 1) and datatype np.float16.
    _y : np.ndarray
        Array containing a single corresponding label of datatype np.int8.
    """
    dict_label_int = eval(dict_label_int)  # Convert str representation of dict into dict object
    counter = 0
    while True:
        path = x[counter]
        try:
            path = Path(path.decode().strip())
            _x = np.load(str(path))  # Load volume
            _x = np.expand_dims(_x, axis=2)  # Convert shape to (..., 1)
            _x = _x.astype(np.float16)  # Mixed precision

            # y integer label
            label = path.parent.name
            label = label.rstrip('0123456789').rstrip('-_')
            _y = np.array([dict_label_int[label]], dtype=np.int8)
            yield {'input_1': _x, 'input_2': _x}, _y
        except ValueError:
            print(path)

        counter += 1
        if counter == len(x):  # Reset counter if end of array is reached
            counter = 0


def generator_inference(x: list, file_format: str):
    """
    Inference generator yielding volumes loaded from .npy files specified in `x`.

    Parameters
    ==========
    x : array-like
        Array of paths to .npy files to load.
    file_format : str
        File format of

    Yields
    ======
    _x : np.ndarray
        Array containing a single volume of shape (..., 1) and datatype np.float16.
    """
    for path_load in x:
        try:
            if file_format == 'nifti':
                _x = nb.load(str(path_load)).get_fdata()
            elif file_format == 'dicom':
                _x = dcmfolder2npy(path=path_load)
            else:
                raise Exception('Unhandled exception')
            _x = __extract_brain(vol=_x)
            _x = _x.astype(np.float16)  # Mixed precision
            yield _x
        except ValueError:
            print(path_load)


def normalize_patches(patches: np.ndarray):
    """
    Normalize each patch to lie between [0, 1].

    Parameters
    ==========
    patches : np.ndarray
        Array of all patches.

    Returns
    ======
    patches : np.ndarray
        Array of all patches individually normalized to lie between [0, 1].
    """
    _max = np.max(patches, axis=(1, 2))
    _min = np.min(patches, axis=(1, 2))
    m3 = _min.reshape((-1, 1, 1))
    _range = (_max - _min).reshape((-1, 1, 1))
    patches = (patches - m3) / _range
    return patches


def patch_size_compatible_zeropad(vol: np.ndarray, patch_size: int):
    """
    Zero-pad input volume such that it is compatible for windowing of specified size.

    Parameters
    ==========
    vol : np.ndarray
        Input volume.
    patch_size : int
        Patch-size that `vol` needs ot be compatible with for windowing.

    Returns
    ======
    np.ndarray
        Zero-padded volume.
    """
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
