import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from artifactID.common.data_ops import glob_brats_t1, load_nifti_vol, get_patches


class SNRObj:
    def __init__(self, arr, obj_mask=None, noise_mask=None):
        self.data = arr.astype(np.float16)
        if obj_mask is None or noise_mask is None:
            self._make_masks()
        else:
            self.obj_mask = obj_mask
            self.noise_mask = noise_mask

    def _make_masks(self):
        obj_idx = np.nonzero(self.data)

        obj_mask = np.zeros_like(self.data)
        obj_mask[obj_idx] = 1
        self.obj_mask = obj_mask.astype(np.int8)

        noise_mask = np.ones_like(self.data)
        noise_mask[obj_idx] = 0
        self.noise_mask = noise_mask.astype(np.int8)

    def add_real_noise(self):
        data = self.data + np.random.normal(loc=0, scale=0.001, size=self.data.shape)
        return SNRObj(data, self.obj_mask, self.noise_mask)

    def add_awgn(self, target_snr_db: float = None, awgn_std: float = None):
        if target_snr_db is None and awgn_std is None:
            raise ValueError('Either target_snr_db or awgn_std have to be passed.')
        elif target_snr_db is not None and awgn_std is not None:
            raise ValueError('Either target_snr_db or awgn_std have to be passed, not both.')

        if target_snr_db:
            object = np.extract(arr=self.data, condition=self.obj_mask)
            awgn_std = object.mean() / math.pow(10, target_snr_db / 20)
        noise = np.random.normal(loc=0, scale=awgn_std, size=int(self.noise_mask.sum())).astype(np.float16)
        data_noisy = np.copy(self.data)
        np.place(arr=data_noisy, mask=self.noise_mask, vals=noise)
        return SNRObj(data_noisy, self.obj_mask, self.noise_mask)

    def get_snr(self):
        object = np.extract(arr=self.data, condition=self.obj_mask)
        noise = np.extract(arr=self.data, condition=self.noise_mask)
        return 20 * np.log10(object.mean() / noise.std())


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


def main(path_brats: str, path_save: str, patch_size: int):
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
    path_all = []
    for snr in arr_snr_range:
        if snr == 2 or snr == 5:
            snr = 99
        path_all.append(path_save / f'snr{snr}')
    # Make save folders if they do not exist
    for p in path_all:
        if not p.exists():
            p.mkdir(parents=True)

    # =========
    # DATAGEN
    # =========
    arr_patches = []
    for ind, path_t1 in tqdm(enumerate(arr_path_brats_t1)):
        subject_name = path_t1.parts[-1].split('.nii.gz')[0]  # Extract subject name from path

        # Load from disk, comes with ideal (0) noise outside brain
        arr_ideal_noise_sliobj = _load_vol_as_snrobj(path_t1)  # Array of slice objects

        # Add real noise (0.001 STD AWGN)
        arr_real_noise_sliobj = [sliobj.add_real_noise() for sliobj in arr_ideal_noise_sliobj]  # Array of slice objects

        # SNR
        snr = arr_snr_range[ind]
        arr_sliobj = [sliobj.add_awgn(target_snr_db=snr) for sliobj in
                      arr_real_noise_sliobj]  # Corrupt to `snr` dB
        vol = [x.data for x in arr_sliobj]  # Stack slices into a single volume
        vol = np.stack(vol)  # Convert from list to numpy.ndarray
        vol = np.moveaxis(vol, [0, 1, 2], [2, 0, 1])  # Iterate through slices on the last dim
        # Normalize to [0, 1]
        _max = vol.max()
        _min = vol.min()
        vol = (vol - _min) / (_max - _min)

        # Zero pad to compatible shape
        pad = []
        shape = vol.shape
        for s in shape:
            if s % patch_size != 0:
                p = patch_size - (s % patch_size)
                pad.append((math.floor(p / 2), math.ceil(p / 2)))
            else:
                pad.append((0, 0))

        # Extract patches
        vol = np.pad(array=vol, pad_width=pad)
        patches = get_patches(arr=vol, patch_size=patch_size)
        patches = patches.reshape((-1, patch_size, patch_size, patch_size))
        arr_patches.extend(patches)

        # Save to disk
        if snr == 2 or snr == 5:
            snr = 99
        for counter, p in enumerate(arr_patches):
            _path_save = str(path_save / f'snr{snr}' / (subject_name + f'_patch{counter}.npy'))
            np.save(arr=p, file=_path_save)  # Save to disk
