import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from artifactID.common import data_ops


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
    vol = data_ops.load_nifti_vol(path)
    arr_sliobj = []
    for i in range(vol.shape[2]):
        sli = vol[:, :, i]
        arr_sliobj.append(SNRObj(sli))

    return arr_sliobj


def main(path_read_data: str, path_save_data: str, patch_size: int):
    arr_snr_range = [2, 5, 11, 15, 20]

    # =========
    # PATHS
    # =========
    if 'miccai' in path_read_data.lower():
        arr_path_read = data_ops.glob_brats_t1(path_brats=path_read_data)
    else:
        arr_path_read = data_ops.glob_nifti(path=path_read_data)
    path_save_data = Path(path_save_data)
    subjects_per_class = math.ceil(
        len(arr_path_read) / len(arr_snr_range))  # Calculate number of subjects per class
    arr_snr_range = arr_snr_range * subjects_per_class
    np.random.shuffle(arr_snr_range)

    # =========
    # DATAGEN
    # =========
    for ind, path_t1 in tqdm(enumerate(arr_path_read)):
        # Load from disk, comes with ideal (0) noise outside brain
        arr_ideal_noise_sliobj = _load_vol_as_snrobj(path_t1)  # Array of slice objects

        # Add real noise (0.001 STD AWGN)
        arr_real_noise_sliobj = [sliobj.add_real_noise() for sliobj in arr_ideal_noise_sliobj]  # Array of slice objects

        # SNR
        snr = arr_snr_range[ind]
        arr_snr_sliobj = [sliobj.add_awgn(target_snr_db=snr) for sliobj in
                          arr_real_noise_sliobj]  # Corrupt to `snr` dB
        arr_snr = [x.data for x in arr_snr_sliobj]
        arr_snr = np.stack(arr_snr)  # Convert from list to numpy.ndarray
        vol = np.moveaxis(arr_snr, [0, 1, 2], [2, 0, 1])  # Iterate through slices on the last dim

        # Zero-pad vol, get patches, discard empty patches and uniformly intense patches and normalize each patch
        vol = data_ops.patch_compatible_zeropad(vol=vol, patch_size=patch_size)
        patches, original_shape = data_ops.get_patches(arr=vol, patch_size=patch_size)
        patches, patch_map = data_ops.prune_patches(patches=patches, original_shape=original_shape)
        patches = data_ops.normalize_patches(patches=patches)

        # Save to disk
        if snr == 2 or snr == 5:
            snr = 99
        _path_save = path_save_data.joinpath(f'snr{snr}')
        if not _path_save.exists():
            _path_save.mkdir(parents=True)
        for counter, p in enumerate(patches):
            suffix = '.nii.gz' if '.nii.gz' in path_t1.name else '.nii'
            subject = path_t1.name.replace(suffix, '')
            _path_save2 = _path_save.joinpath(subject)
            _path_save2 = str(_path_save2) + f'_patch{counter}.npy'
            np.save(arr=p, file=_path_save2)
