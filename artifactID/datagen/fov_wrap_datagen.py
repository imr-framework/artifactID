import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from artifactID.datagen.data_ops import glob_brats_t1, load_nifti_vol, get_patches


def main(path_read_data: str, path_save_data: str, patch_size: int):
    arr_wrap_range = [55, 60, 65, 70, 75, 80]

    # =========
    # PATHS
    # =========
    arr_path_read = glob_brats_t1(path_brats=path_read_data)
    path_save_data = Path(path_save_data)
    subjects_per_class = math.ceil(
        len(arr_path_read) / len(arr_wrap_range))  # Calculate number of subjects per class
    arr_wrap_range = arr_wrap_range * subjects_per_class
    np.random.shuffle(arr_wrap_range)

    # =========
    # DATAGEN
    # =========
    arr_patches = []
    arr_labels = []
    for ind, path_t1 in tqdm(enumerate(arr_path_read)):
        vol = load_nifti_vol(path_t1)
        wrap = arr_wrap_range[ind]
        opacity = 0.5
        top, middle, bottom = vol[:wrap], vol[wrap:-wrap], vol[-wrap:]
        middle[:wrap] += bottom * opacity
        middle[-wrap:] += top * opacity
        middle = np.pad(middle, [[wrap, wrap], [0, 0], [0, 0]])
        # Normalize to [0, 1]
        _max = middle.max()
        _min = middle.min()
        middle = (middle - _min) / (_max - _min)

        # Zero pad to compatible shape
        pad = []
        shape = middle.shape
        for s in shape:
            if s % patch_size != 0:
                p = patch_size - (s % patch_size)
                pad.append((math.floor(p / 2), math.ceil(p / 2)))
            else:
                pad.append((0, 0))

        # Extract patches
        middle = np.pad(array=middle, pad_width=pad)
        patches = get_patches(arr=middle, patch_size=patch_size)
        patches = patches.reshape((-1, patch_size, patch_size, patch_size)).astype(np.float16)
        arr_patches.extend(patches)
        arr_labels.extend([wrap] * len(patches))


        _path_save = path_save_data.joinpath(f'wrap{wrap}')
        if not _path_save.exists():
            _path_save.mkdir(parents=True)
        for counter, p in enumerate(patches):
            subject = path_t1.name.replace('.nii.gz', '')
            _path_save2 = _path_save.joinpath(subject)
            _path_save2 = str(_path_save2) + f'_patch{counter}.npy'
            np.save(arr=p, file=_path_save2)
