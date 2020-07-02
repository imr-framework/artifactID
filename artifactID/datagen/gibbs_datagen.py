
import numpy as np
import math


from pathlib import Path
from tqdm import tqdm

from artifactID.common.data_ops import glob_brats_t1, glob_nifti, load_nifti_vol, get_patches

def main(path_read_data: str, path_save_data: str, patch_size: int):

    arr_gibbs_range = [52, 64, 76]
    # =========
    # PATHS
    # =========
    if 'miccai' in path_read_data.lower():
        arr_path_read = glob_brats_t1(path_brats=path_read_data)
    else:
        arr_path_read = glob_nifti(path=path_read_data)
    path_save_data = Path(path_save_data)
    subjects_per_class = math.ceil(
        len(arr_path_read) / len(arr_gibbs_range))  # Calculate number of subjects per class

    arr_gibbs_range = np.tile(arr_gibbs_range, subjects_per_class)
    np.random.shuffle(arr_gibbs_range)

    # =========
    # DATAGEN
    # =========
    arr_patches = []
    arr_labels = []
    for ind, path_t1 in tqdm(enumerate(arr_path_read)):

        vol = load_nifti_vol(path=path_t1)
        chop = arr_gibbs_range[ind]

        kdat = np.fft.fftshift(np.fft.fftn(vol))
        kdat[:, :chop, :] = 0
        kdat[:, -chop:, :] = 0
        vol_gibbs = np.abs(np.fft.ifftn(np.fft.ifftshift(kdat)))

        # Zero pad to compatible shape
        pad = []
        shape = vol_gibbs.shape
        for s in shape:
            if s % patch_size != 0:
                p = patch_size - (s % patch_size)
                pad.append((math.floor(p / 2), math.ceil(p / 2)))
            else:
                pad.append((0, 0))

        # Extract patches
        vol_gibbs = np.pad(array=vol_gibbs, pad_width=pad)
        patches = get_patches(arr=vol_gibbs, patch_size=patch_size)
        arr_patches.extend(patches)
        arr_labels.extend([chop] * len(patches))

        _path_save = path_save_data.joinpath(f'gibbs{chop}')
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