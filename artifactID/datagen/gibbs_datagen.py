import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from artifactID.common import data_ops


def main(path_read_data: Path, path_save_data: Path, slice_size: int):
    arr_gibbs_range = [64, 76, 88]
    # =========
    # PATHS
    # =========
    arr_path_read_data = list(path_read_data.glob('noartifact\*'))

    path_save_data = Path(path_save_data)
    subjects_per_class = math.ceil(
        len(arr_path_read_data) / len(arr_gibbs_range))  # Calculate number of subjects per class

    arr_gibbs_range = np.tile(arr_gibbs_range, subjects_per_class)
    np.random.shuffle(arr_gibbs_range)

    # =========
    # DATAGEN
    # =========
    for ind, f in tqdm(enumerate(arr_path_read_data)):
        sli = data_ops.load_preprocess_npy(f, slice_size)
        sli_masked, mask_idx = data_ops.__extract_brain(sli, return_idx=True)  # Re-extract brain to obtain masking idx
        sli_masked = np.expand_dims(sli_masked, axis=2)
        fe_chop = arr_gibbs_range[ind]

        # K-space
        kdat = np.fft.fftshift(np.fft.fftn(sli_masked))
        num_pe_chop = np.random.randint(low=32, high=64)
        pe_chop_start = np.random.randint(low=0, high=160)
        pe_chop_stop = pe_chop_start + num_pe_chop
        kdat[pe_chop_start:pe_chop_stop, :fe_chop, :] = 0
        kdat[pe_chop_start:pe_chop_stop, -fe_chop:, :] = 0
        sli_gibbs = np.abs(np.fft.ifftn(np.fft.ifftshift(kdat)))

        # Values outside brain should be 0
        sli_gibbs_masked = np.zeros_like(sli_gibbs)
        sli_gibbs_masked[mask_idx] = sli_gibbs[mask_idx]

        # Convert to float16 to avoid dividing by 0 during normalization - very low max values get zeroed out
        sli_gibbs = sli_gibbs_masked.astype(np.float16)

        # Save to disk
        _path_save = path_save_data.joinpath(f'gibbs{fe_chop}')
        if not _path_save.exists():
            _path_save.mkdir(parents=True)
        _path_save = _path_save / f.name
        np.save(arr=sli_gibbs, file=_path_save)
