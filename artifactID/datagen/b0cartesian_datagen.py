import math
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import scipy.io as sio


from artifactID.datagen import generate_fieldmap
from artifactID.common.data_ops import glob_brats_t1, load_nifti_vol, get_patches, glob_nifti
from OCTOPUS.Recon import ORC


def _gen_fieldmap(_slice, _freq_range):
    field_map = generate_fieldmap.hyperbolic(_slice.shape[0], _freq_range)  # Simulate the field map

    return field_map


def orc_forwardmodel(vol: np.ndarray, freq_range: int, ktraj: np.ndarray):
    """
    Adds off-resonance blurring to simulate B0 inhomogeneity artifacts.

    Parameters
    ==========
    vol : numpy.ndarray
        Image volume having dimensions N x N x N number of slices
    freq_range : int
        Frequency range for the simulated field map
    ktraj : numpy.ndarray
        k-space trajectory coordinates. Dimensions Npoints x Nshots
    seq_params : dict
        Sequence parameters needed for off-resonance corruption

    Returns
    =======
    arr_offres_vol : np.ndarray
        Corrupted image volume having dimensions Slices(with signal > 5%) x N x N
    """

    num_slices = vol.shape[2]
    arr_offres_vol = np.zeros(vol.shape)

    for ind in range(num_slices):
        slice = vol[:, :, ind]


        fieldmap= _gen_fieldmap(_slice=slice, _freq_range=freq_range)
        or_corrupted = ORC.add_or_CPR(slice, ktraj, fieldmap)  # Corrupt the image

        or_corrupted_norm = np.zeros(or_corrupted.shape)
        or_corrupted_norm = cv2.normalize(np.abs(or_corrupted), or_corrupted_norm, 0, 1,
                                          cv2.NORM_MINMAX)  # Normalize [0, 1]
        arr_offres_vol[:,:,ind] = np.float16(or_corrupted_norm)

    #arr_offres_vol = np.stack(arr_offres_vol)
    return arr_offres_vol


def main(path_read_data: str, path_save_data: str, patch_size: int):
    # =========
    # LOAD PREREQUISITES
    # =========
    # 1. k-space trajectory
    dt = 10e-6  # grad raster time


    # BraTS 2018 paths
    if 'miccai' in path_read_data.lower():
        arr_path_read = glob_brats_t1(path_brats=path_read_data)
    else:
        arr_path_read = glob_nifti(path=path_read_data)
    path_save_data = Path(path_save_data)

    arr_max_freq = [2500, 5000, 7500]  # Hz
    subjects_per_class = math.ceil(len(arr_path_read) / len(arr_max_freq))  # Calculate number of subjects per class
    arr_max_freq *= subjects_per_class
    np.random.shuffle(arr_max_freq)

    '''path_save = Path(path_save)
    path_all = [path_save / f'b0_{freq}' for freq in arr_max_freq]
    # Make save folders if they do not exist
    for p in path_all:
        if not p.exists():
            p.mkdir(parents=True)'''

    # =========
    # DATAGEN
    # =========
    arr_patches = []
    arr_labels = []
    for ind, path_t1 in tqdm(enumerate(arr_path_read)):

        vol = load_nifti_vol(path=path_t1)

        N = vol.shape[0]
        ktraj_cart = np.arange(0, N * dt, dt).reshape(1, N)
        ktraj = np.tile(ktraj_cart, (N, 1))

        #t_vector = (np.arange(ktraj.shape[0]) * 10e-6).reshape(ktraj.shape[0], 1)
        #seq_params = {'Npoints': ktraj.shape[0], 'Nshots': ktraj.shape[1], 't_vector': t_vector, 'dcf': dcf}

        freq = arr_max_freq[ind]
        vol_b0 = orc_forwardmodel(vol=vol, freq_range=freq, ktraj=ktraj)
        #vol_b0 = np.moveaxis(vol_b0, [0, 1, 2], [2, 0, 1])  # Iterate through slices on the last dim

        # Zero pad back to 155
        '''orig_num_slices = 155
        n_zeros = (orig_num_slices - vol_b0.shape[2]) / 2
        n_zeros = [math.floor(n_zeros), math.ceil(n_zeros)]
        vol_b0 = np.pad(vol_b0, [[0, 0], [0, 0], n_zeros])'''

        # Zero pad to compatible shape
        pad = []
        shape = vol_b0.shape
        for s in shape:
            if s % patch_size != 0:
                p = patch_size - (s % patch_size)
                pad.append((math.floor(p / 2), math.ceil(p / 2)))
            else:
                pad.append((0, 0))

        # Extract patches
        vol_b0 = np.pad(array=vol_b0, pad_width=pad)
        patches = get_patches(arr=vol_b0, patch_size=patch_size)
        arr_patches.extend(patches)
        arr_labels.extend([freq] * len(patches))

        _path_save = path_save_data.joinpath(f'b0{freq}')
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

        '''subject_name = path_t1.parts[-1].split('.nii.gz')[0]  # Extract subject name from path
        _path_save = str(path_save / f'b0_{freq}' / subject_name) + '.npy'
        np.save(arr=vol_b0, file=_path_save)'''
        # print('Corrupted data saved for subject:' + folder_name)
