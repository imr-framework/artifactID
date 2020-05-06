import math
from pathlib import Path

import numpy as np

from artifactID.utils import glob_brats_t1, load_nifti_vol
from artifactID.datagen.snr_class import SNRObj


def _load_vol_as_snrobj(path: str):
    """
    Read NIFTI file at `path` and return an array of SNRObj. Each SNRObj is a slice from the NIFTI file. Only slices
    having 5% or more signal are considered.

    Parameters
    ==========
    path : str
        Path to NIFTI file to be read.

    Returns
    list
        Array of individual slices in NIFTI file at `path`. Each slice is represented as an instance of
        class SNRObj.
    """
    vol = load_nifti_vol(path)
    arr_sliobj = []
    for i in range(vol.shape[2]):
        sli = vol[:, :, i]
        arr_sliobj.append(SNRObj(sli))

    return arr_sliobj


def main(path_brats: str, path_save: str):
    arr_snr_range = [2, 5, 11, 15, 20]

    # =========
    # BRATS PATHS
    # =========
    arr_path_brats_t1 = glob_brats_t1(path_brats=path_brats)
    num_subjects = len(arr_path_brats_t1)

    subjects_per_class = math.ceil(
        len(arr_path_brats_t1) / len(arr_snr_range))  # Calculate number of subjects per class
    arr_snr_range = arr_snr_range * subjects_per_class
    np.random.shuffle(arr_snr_range)

    path_save = Path(path_save)
    path_all = [path_save / "mask", *[path_save / f'snr{snr}' for snr in arr_snr_range]]
    # Make save folders if they do not exist
    for p in path_all:
        if not p.exists():
            p.mkdir(parents=True)

    # =========
    # DATAGEN
    # =========
    for ind, path_t1 in enumerate(arr_path_brats_t1):
        subject_name = path_t1.parts[-1].split('.nii.gz')[0]  # Extract subject name from path

        pc = round((ind + 1) / num_subjects * 100, ndigits=2)
        print(f'{pc}%', end=', ', flush=True)

        # Ideal noise
        arr_ideal_noise_sliobj = _load_vol_as_snrobj(path_t1)  # Array of slice objects

        # Brain masks
        arr_masks = [x.obj_mask for x in arr_ideal_noise_sliobj]  # Array of slice masks
        arr_masks = np.stack(arr_masks)  # Convert from list to numpy.ndarray
        np.moveaxis(arr_masks, [0, 1, 2], [1, 2, 0])  # Iterate through slices on the last dim
        _path_save = str(path_save / 'mask' / subject_name) + '.npy'
        np.save(arr=arr_masks, file=_path_save)  # Save to disk

        # Add real noise (0.001 STD AWGN)
        arr_real_noise_sliobj = [sliobj.add_real_noise() for sliobj in arr_ideal_noise_sliobj]  # Array of slice objects

        # SNR
        snr = arr_snr_range[ind]
        arr_snr_sliobj = [sliobj.add_awgn(target_snr_db=snr) for sliobj in
                          arr_real_noise_sliobj]  # Corrupt to `snr` dB
        arr_snr = [x.data for x in arr_snr_sliobj]
        arr_snr = np.stack(arr_snr)  # Convert from list to numpy.ndarray
        arr_snr = np.moveaxis(arr_snr, [0, 1, 2], [2, 0, 1])  # Iterate through slices on the last dim
        _path_save = str(path_save / f'snr{snr}' / subject_name) + '.npy'
        np.save(arr=arr_snr, file=_path_save)  # Save to disk
