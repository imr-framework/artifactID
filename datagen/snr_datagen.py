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
    path_real = Path(r"C:\Users\sravan953\Documents\CU\Projects\imr-framework\ArtifactID\Data\real_noise")
    if not path_real.exists():
        path_real.mkdir(parents=True)

    # First make the SNR save destination folders
    snr_range = [2, 5, 11, 15, 20]
    for snr in snr_range:
        path_snr = Path(r"C:\Users\sravan953\Documents\CU\Projects\imr-framework\ArtifactID\Data\snr{}".format(snr))
        if not path_snr.exists():
            path_snr.mkdir(parents=True)
    path_snr = Path(r"C:\Users\sravan953\Documents\CU\Projects\imr-framework\ArtifactID\Data\snr")

    path_brats = r"C:\Users\sravan953\Documents\CU\Projects\imr-framework\DART\Data\MICCAI_BraTS_2018_Data_Training"
    path_brats = Path(path_brats)

    arr_brats_paths = list(path_brats.glob('**/*.nii.gz'))
    arr_brats_paths = list(filter(lambda x: 't1.nii' in str(x), arr_brats_paths))
    num_subjects = len(arr_brats_paths)
    for i, p in enumerate(arr_brats_paths):
        pc = round((i + 1) / num_subjects * 100, ndigits=2)
        print(f'{pc}%', end=', ', flush=True)

        # Ideal noise
        arr_brats_ideal_sliobj = _load_vol(p)
        arr_brats_ideal = [x.data for x in arr_brats_ideal_sliobj]

        # Real noise
        arr_brats_real_sliobj = [sliobj.add_real_noise() for sliobj in
                                 arr_brats_ideal_sliobj]  # Add real noise (0.001 STD AWGN)
        arr_brats_real = [x.data for x in arr_brats_real_sliobj]
        path_save = path_real / (p.parts[-1].split('.')[0] + '.npy')
        np.save(arr=arr_brats_real, file=path_save)  # Save to disk

        for snr in snr_range:
            arr_brats_snr_sliobj = [sliobj.add_awgn(target_snr_db=snr) for sliobj in
                                    arr_brats_real_sliobj]  # Corrupt to `snr` dB SNR
            arr_brats_snr = [x.data for x in arr_brats_snr_sliobj]
            path_save = Path(str(path_snr) + str(snr)) / (p.parts[-1].split('.')[0] + '.npy')
            np.save(arr=arr_brats_snr, file=path_save)  # Save to disk
