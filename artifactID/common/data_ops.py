import math
from pathlib import Path

import cv2
import nibabel as nb
import numpy as np


def __extract_brain(vol: np.ndarray, return_idx: bool = False):
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

    if return_idx:
        return vol_masked, idx
    return vol_masked


def glob_nifti(path: Path):
    arr_path = list(path.glob('**/*.nii.gz'))
    arr_path2 = list(path.glob('**/*.nii'))
    return arr_path + arr_path2


def glob_brats_t1(path_brats: Path):
    arr_path_brats_t1 = list(path_brats.glob('**/*.nii.gz'))
    arr_path_brats_t1 = list(filter(lambda x: 't1.nii' in str(x), arr_path_brats_t1))
    return arr_path_brats_t1


def load_preprocess_npy(path: Path, target_size: int):
    """
    Read Numpy file at `path` and return an 3D numpy.ndarray.

    Parameters
    ==========
    path : Path
        Path to npy file to be read.

    Returns
    =======
    numpy.ndarray
        Numpy data of NIFTI file at `path`.
    """
    npy = np.load(str(path))
    if len(npy.shape) == 3:
        npy = resize_vol(npy, target_size)
    else:
        npy = resize(npy, target_size)
    npy = __extract_brain(npy)
    npy_normalized = (npy - npy.min()) / (npy.max() - npy.min())  # Normalize between 0-1
    return npy_normalized.astype(np.float16)


def load_nifti_vol(path: Path, target_size: int):
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
    vol = nb.load(str(path)).get_fdata()
    vol = resize_vol(vol, target_size)
    vol = __extract_brain(vol)
    vol = (vol - vol.min()) / (vol.max() - vol.min())  # Normalize between 0-1
    return vol.astype(np.float16)


def generator_train_eval(x, mode: bytes = b'train'):
    """
    Training generator yielding volumes loaded from paths to .npy files in `x`. Also yields paired labels `y`. Labels
    are derived from the paths to the .npy files in `x`.

    Parameters
    ==========
    x : array-like
        Array of paths to .npy files to load.
    mode : bytes
        Bytes array indicating if generator is used for training or evaluation. Datatype is bytes because Tensorflow
        passes arguments as bytes in tf.data.Dataset

    Yields
    ======
    _x : np.ndarray
        K-space array of dimensions (x, y, 1) and datatype np.float16.
    _y : np.ndarray
        Array containing a single integer label indicating artifact-type of datatype np.int8.
    """
    counter = 0
    mode = mode.decode()
    while True:
        path = x[counter]
        if isinstance(path, bytes):
            path = path.decode()
        path = Path(path.strip())
        try:
            _x = np.load(str(path))  # Load volume
            _x = np.abs(np.fft.fftshift(np.fft.fftn(_x)))
            _x = np.squeeze(_x)
            _x = np.expand_dims(_x, axis=2)  # Convert shape to (..., 1)
            _x = _x.astype(np.float16)  # Mixed precision

            if mode == 'eval' or mode == 'evaluate':
                yield _x
            else:
                # y = 0 for no artifact; y = 1 for Gibbs
                folder = path.parent.name.lower()
                if 'gibbs' in folder:
                    _y = np.array([1], dtype=np.int8)
                elif 'noartifact' in folder:
                    _y = np.array([0], dtype=np.int8)
                else:
                    raise ValueError('Not training on Gibbs')

                yield _x, _y
        except Exception as e:
            print(e)

        counter += 1
        if counter == len(x):
            if mode == 'eval' or mode == 'evaluate':
                break  # Break if end of array is reached during evaluation
            counter = 0  # Reset counter if end of array is reached during training


def normalize_slices(vol: np.ndarray):
    """
    Normalize each slice to lie between [0, 1].

    Parameters
    ==========
    vol : np.ndarrays

    Returns
    ======
    vol_normalized : np.ndarray
        Array of all patches individually normalized to lie between [0, 1].
    """
    _max = np.max(vol, axis=(0, 1))
    _min = np.min(vol, axis=(0, 1))
    _range = (_max - _min)
    valid_slices = np.where(_max != _min)[0]
    vol_normalized = (vol[..., valid_slices] - _min[valid_slices]) / _range[valid_slices]
    return vol_normalized


def resize(sli: np.ndarray, size: int):
    return cv2.resize(sli.astype(np.float), (size, size))


def resize_vol(vol: np.ndarray, size: int):
    vol_resized = []
    for i in range(vol.shape[-1]):
        _slice = vol[..., i]
        vol_resized.append(cv2.resize(_slice.astype(np.float), (size, size)))
    vol_resized = np.stack(vol_resized)
    vol_resized = np.moveaxis(vol_resized, (0, 1, 2), (2, 0, 1))
    return vol_resized


def vol_slice_pad(vol: np.ndarray, dim_2_pad: int):
    if dim_2_pad == 0:
        target_N = vol.shape[1]
    else:
        target_N = vol.shape[0]
    pad = target_N - vol.shape[dim_2_pad]
    pad1, pad2 = math.ceil(pad / 2), int(pad / 2)

    if dim_2_pad == 0:
        vol_padded = np.vstack(
            (np.zeros((pad1, vol.shape[1], vol.shape[2])), vol, np.zeros((pad2, vol.shape[1], vol.shape[2]))))
    else:
        vol_padded = np.hstack(
            (np.zeros((vol.shape[0], pad1, vol.shape[2])), vol, np.zeros((vol.shape[0], pad2, vol.shape[2]))))

    return vol_padded
