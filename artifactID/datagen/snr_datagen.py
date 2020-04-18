from pathlib import Path

import nibabel as nb
import numpy as np

from classes.snr_class import SNRObj


def _load_vol(path: str):
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
    vol = nb.load(path).get_fdata()
    vol = np.rot90(vol, -1, axes=(0, 1))
    vol = (vol - vol.min()) / (vol.max() - vol.min())  # Normalize between 0-1
    arr_sliobj = []
    for i in range(vol.shape[2]):
        sli = vol[:, :, i]
        if np.count_nonzero(sli) > 0.05 * sli.size:  # Check if at least 5% of signal is present
            arr_sliobj.append(SNRObj(sli))

    return arr_sliobj


if __name__ == '__main__':
    # First make the save destination folders: real noise, object mask and each SNR
    path_save_real = Path(r"C:\Users\sravan953\Documents\CU\Projects\imr-framework\ArtifactID\Data\real_noise")
    path_save_obj_mask = Path(r"C:\Users\sravan953\Documents\CU\Projects\imr-framework\ArtifactID\Data\mask")
    for p in [path_save_real, path_save_obj_mask]:
        if not p.exists():
            p.mkdir(parents=True)
    snr_range = [2, 5, 11, 15, 20]
    for snr in snr_range:
        path_save_snr = Path(
            r"C:\Users\sravan953\Documents\CU\Projects\imr-framework\ArtifactID\Data\snr{}".format(snr))
        if not path_save_snr.exists():
            path_save_snr.mkdir(parents=True)
    path_save_snr = Path(r"C:\Users\sravan953\Documents\CU\Projects\imr-framework\ArtifactID\Data\snr")

    # BraTS 2018 paths
    path_read_brats = r"C:\Users\sravan953\Documents\CU\Projects\imr-framework\DART\Data\MICCAI_BraTS_2018_Data_Training"
    path_read_brats = Path(path_read_brats)
    arr_brats_paths = list(path_read_brats.glob('**/*.nii.gz'))
    arr_brats_paths = list(filter(lambda x: 't1.nii' in str(x), arr_brats_paths))
    num_subjects = len(arr_brats_paths)

    for i, p in enumerate(arr_brats_paths):
        pc = round((i + 1) / num_subjects * 100, ndigits=2)
        print(f'{pc}%', end=', ', flush=True)

        # Ideal noise
        arr_brats_ideal_sliobj = _load_vol(p)
        arr_obj_masks = [x.obj_mask for x in arr_brats_ideal_sliobj]  # Object mask
        arr_obj_masks = np.stack(arr_obj_masks)
        np.moveaxis(arr_obj_masks, [0, 1, 2], [1, 2, 0])  # Iterate through slices on the last dim
        path_save = path_save_obj_mask / (p.parts[-1].split('.')[0] + '.npy')
        np.save(arr=arr_obj_masks, file=path_save)  # Save to disk

        # Real noise
        arr_brats_real_sliobj = [sliobj.add_real_noise() for sliobj in
                                 arr_brats_ideal_sliobj]  # Add real noise (0.001 STD AWGN)
        arr_brats_real = [x.data for x in arr_brats_real_sliobj]
        arr_brats_real = np.stack(arr_brats_real)
        np.moveaxis(arr_brats_real, [0, 1, 2], [1, 2, 0])  # Iterate through slices on the last dim
        path_save = path_save_real / (p.parts[-1].split('.')[0] + '.npy')
        np.save(arr=arr_brats_real, file=path_save)  # Save to disk

        # SNR
        for snr in snr_range:
            arr_brats_snr_sliobj = [sliobj.add_awgn(target_snr_db=snr) for sliobj in
                                    arr_brats_real_sliobj]  # Corrupt to `snr` dB
            arr_brats_snr = [x.data for x in arr_brats_snr_sliobj]
            arr_brats_snr = np.stack(arr_brats_snr)
            np.moveaxis(arr_brats_snr, [0, 1, 2], [1, 2, 0])  # Iterate through slices on the last dim
            path_save = Path(str(path_save_snr) + str(snr)) / (p.parts[-1].split('.')[0] + '.npy')
            np.save(arr=arr_brats_snr, file=path_save)  # Save to disk
