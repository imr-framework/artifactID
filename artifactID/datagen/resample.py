from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_to_output
from tqdm import tqdm

from common_utils.preprocessor import normalize_volume


def crop_or_pad(arr: np.ndarray):
    target_size = 256

    def _crop_dim(arr: np.ndarray, dim: int):
        current_size = arr.shape[dim]
        if current_size != target_size:
            # Crop or pad arr at dim
            arr_out = np.moveaxis(arr, dim, 0)

            crop = current_size - target_size
            before_crop = abs(crop) // 2
            after_crop = before_crop
            if abs(crop) % 2 != 0:
                after_crop += 1

            if crop < 0:
                arr_out = np.pad(arr_out, ((before_crop, after_crop), (0, 0), (0, 0)))
            else:
                arr_out = arr_out[before_crop:-after_crop]

            arr_out = np.moveaxis(arr_out, 0, dim)
            return arr_out
        else:
            return arr

    arr_out = arr.copy()
    # Crop or pad each in-plane dim
    for i in range(2):
        arr_out = _crop_dim(arr_out, dim=i)

    assert arr_out.shape[:2] == (256, 256)
    # print(arr.shape, arr_out.shape)

    return arr_out


def main(path_read: Path, path_save: Path):
    path_save.mkdir(parents=False, exist_ok=True)

    files = list(path_read.glob('*.gz'))

    for f in tqdm(files):
        nb = nib.load(str(f))
        zooms = nb.header.get_zooms()
        nb_resampled = resample_to_output(nb, (zooms[0], 1, 1))
        vol_resampled = nb_resampled.get_data()

        print(nb.get_data().shape, vol_resampled.shape)

        # Fix orientation and save to disk
        vol_resampled = np.moveaxis(vol_resampled, 0, -1)  # Slices in last dim
        vol_resampled = np.fliplr(np.rot90(vol_resampled, 1, axes=(0, 1)))  # Fix orientation
        vol_resampled = crop_or_pad(vol_resampled)  # Crop to 256x256
        vol_resampled = normalize_volume(vol_resampled)  # Normalize

        path_save_vol = path_save / f.name
        nb_resampled = nib.Nifti2Image(vol_resampled, affine=None)
        # nib.save(nb_resampled, path_save_vol)


if __name__ == '__main__':
    path_read = Path(r"D:\MMJ\Data\LAC-T1")
    path_save = Path(r"D:\Sravan\Data\Sou3rce\LAC-T1_resampled")

    main(path_read, path_save)
