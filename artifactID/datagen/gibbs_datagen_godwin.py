import math
from pathlib import Path

import numpy as np
import pydicom as pyd
from tqdm import tqdm

from artifactID.common import data_ops


def main(arr_path_read: Path, path_save_data: Path, slice_size: int):
    arr_gibbs_range = [64, 76, 88]
    # =========
    # PATHS
    # =========
    files = list(arr_path_read.glob('noartifact\*'))

    path_save_data = Path(path_save_data)
    subjects_per_class = math.ceil(
        len(files) / len(arr_gibbs_range))  # Calculate number of subjects per class

    arr_gibbs_range = np.tile(arr_gibbs_range, subjects_per_class)
    np.random.shuffle(arr_gibbs_range)

    # =========
    # DATAGEN
    # =========
    for ind, f in tqdm(enumerate(files)):
        sli = np.load(str(f))
        if sli.shape != (slice_size, slice_size):
            sli_resized = data_ops.resize(sli, size=slice_size)
        else:
            sli_resized = sli
        sli_resized, idx = data_ops.__extract_brain(sli_resized,
                                                    return_idx=True)  # Re-extract brain to obtain masking idx
        sli_resized = np.expand_dims(sli_resized, axis=2)
        sli_resized_normalized = data_ops.normalize_slices(vol=sli_resized)
        chop = arr_gibbs_range[ind]

        kdat = np.fft.fftshift(np.fft.fftn(sli_resized_normalized))
        num_pe_chop = np.random.randint(low=32, high=64)
        rand_start = np.random.randint(low=0, high=160)
        rand_stop = rand_start + num_pe_chop
        # print(num_pe_chop, rand_start, rand_stop)
        kdat[rand_start:rand_stop, :chop, :] = 0
        kdat[rand_start:rand_stop, -chop:, :] = 0
        sli_gibbs = np.abs(np.fft.ifftn(np.fft.ifftshift(kdat)))
        sli_gibbs_masked = np.zeros_like(sli_gibbs)
        sli_gibbs_masked[idx] = sli_gibbs[idx]
        # Convert to float16 to avoid dividing by 0 during normalization - very low max values get zeroed out
        sli_gibbs = sli_gibbs_masked.astype(np.float16)

        # Save to disk
        _path_save = path_save_data.joinpath(f'gibbs{chop}')
        if not _path_save.exists():
            _path_save.mkdir(parents=True)
        _path_save = _path_save / f.name
        np.save(arr=sli_gibbs, file=_path_save)
