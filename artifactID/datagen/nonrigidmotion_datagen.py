import numpy as np
import math
import cv2
import matplotlib.pyplot as plt


from pathlib import Path
from tqdm import tqdm
from PIL import Image

from artifactID.common.data_ops import glob_brats_t1, glob_nifti, load_nifti_vol, get_patches

def main(path_read_data: str, path_save_data: str, patch_size: int):

    # =========
    # PATHS
    # =========
    if 'miccai' in path_read_data.lower():
        arr_path_read = glob_brats_t1(path_brats=path_read_data)
    else:
        arr_path_read = glob_nifti(path=path_read_data)
    path_save_data = Path(path_save_data)
    #subjects_per_class = math.ceil(
    #   len(arr_path_read) / len(arr_gibbs_range))  # Calculate number of subjects per class

    #arr_gibbs_range = np.tile(arr_gibbs_range, subjects_per_class)
    #np.random.shuffle(arr_gibbs_range)

    # =========
    # DATAGEN
    # =========
    arr_patches = []
    arr_labels = []
    for ind, path_t1 in tqdm(enumerate(arr_path_read)):

        vol = load_nifti_vol(path=path_t1)
        vol_norm = np.zeros(vol.shape)
        vol_norm = cv2.normalize(vol, vol_norm, 0, 255, cv2.NORM_MINMAX)

        rot = 0
        trans = 0
        while abs(rot) < 5 or abs(trans) <5:
            rot = np.random.randint(-10,11)
            trans = np.random.randint(-10,11)

        vol_rot = np.zeros(vol.shape)
        for sl in range(vol.shape[-1]):
            slice = Image.fromarray(vol_norm[:,:,sl])
            slice_rot = slice.rotate(rot)
            vol_rot[:, :, sl] = slice_rot


        trans_direction = np.random.choice([0,1])
        vol_trans = np.roll(vol_norm, trans, axis=trans_direction)

        kdat_orig = np.fft.fftshift(np.fft.fftn(vol_norm))
        kdat_rot = np.fft.fftshift(np.fft.fftn(vol_rot))
        kdat_trans = np.fft.fftshift(np.fft.fftn(vol_trans))
        kdat_nrm = np.fft.fftshift(np.fft.fftn(vol_norm))

        random_lines = np.random.randint(0, 240, 100)
        if trans_direction == 0:
            kdat_nrm[random_lines[:50],:, :] = kdat_rot[random_lines[:50], :,:]
            kdat_nrm[random_lines[50:], :, :] = kdat_trans[random_lines[50:],:, :]
            kdat_nrm[110:130, :, :] = kdat_orig[110:130, :, :]
        else:
            kdat_nrm[ :,random_lines[:50], :] = kdat_rot[ :,random_lines[:50], :]
            kdat_nrm[ :,random_lines[50:], :] = kdat_trans[ :,random_lines[50:], :]
            kdat_nrm[ :,110:130, :] = kdat_orig[ :,110:130, :]

        vol_nrm = np.abs(np.fft.ifftn(np.fft.ifftshift(kdat_nrm)))
        # Zero pad to compatible shape
        pad = []
        shape = vol_nrm.shape
        for s in shape:
            if s % patch_size != 0:
                p = patch_size - (s % patch_size)
                pad.append((math.floor(p / 2), math.ceil(p / 2)))
            else:
                pad.append((0, 0))

        # Extract patches
        vol_nrm = np.pad(array=vol_nrm, pad_width=pad)
        patches = get_patches(arr=vol_nrm, patch_size=patch_size)
        arr_patches.extend(patches)
        arr_labels.extend(['nrm'] * len(patches))

        _path_save = path_save_data.joinpath(f'nrm')
        if not _path_save.exists():
            _path_save.mkdir(parents=True)
        for counter, p in enumerate(patches):

            subject = path_t1.name.replace('.nii.gz', '')
            _path_save2 = _path_save.joinpath(subject)
            if np.count_nonzero(p) == 0 or p.min() == p.max():
               pass
            else:
                # Normalize to [0, 1]
                _max = p.max()
                _min = p.min()
                p = (p - _min) / (_max - _min)

                _path_save2 = str(_path_save2) + f'_patch{counter}.npy'
                np.save(arr=p, file=_path_save2)