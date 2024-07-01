from pathlib import Path

import numpy as np
from skimage import transform
from tqdm import tqdm

from artifactID.datagen.utils import glob_nifti
from common_utils import data_loader, preprocessor


def cutart(sli: np.ndarray, sli_gibbs: np.ndarray) -> np.ndarray:
    patch_size_low = 0.3 * sli.shape[0]
    patch_size_high = 0.5 * sli.shape[0]
    patch_size = np.random.randint(low=patch_size_low, high=patch_size_high)
    crop_low_xy = 32
    crop_high_x = 64
    crop_high_y = 128
    crop_x = np.random.randint(low=crop_low_xy, high=crop_high_x)
    crop_y = np.random.randint(low=crop_low_xy, high=crop_high_y)
    sli_gibbs_cutart = sli.copy()
    sli_gibbs_cutart[
    crop_x: crop_x + patch_size, crop_y: crop_y + patch_size
    ] = sli_gibbs[crop_x: crop_x + patch_size, crop_y: crop_y + patch_size]

    # # Debug - visualize crop
    # from matplotlib import pyplot as plt
    # plt.subplot(221)
    # plt.imshow(sli, cmap='gray')
    # plt.axis('off')
    #
    # plt.subplot(222)
    # plt.imshow(sli_gibbs, cmap='gray')
    # plt.axis('off')
    #
    # plt.subplot(223)
    # cutart = np.zeros_like(sli)
    # cutart[sli_gibbs_cutart - sli != 0] = 1
    # plt.imshow(cutart, cmap='gray')
    # plt.axis('off')
    #
    # plt.subplot(224)
    # plt.imshow(sli_gibbs_cutart, cmap='gray')
    # plt.axis('off')
    #
    # plt.tight_layout()
    # plt.show()

    return sli_gibbs_cutart


def gibbs(img: np.ndarray, crop: int) -> np.ndarray:
    # Convert to k-space and crop to simulate Gibbs Ringing
    kspace = np.fft.fftshift(np.fft.fftn(img))
    kspace_chopped = np.zeros_like(kspace)

    center = kspace.shape[0] // 2
    slice_obj = slice(center - crop // 2, center + crop // 2)  # Crop to 100x100

    kspace_chopped[slice_obj, slice_obj] = kspace[slice_obj, slice_obj]
    kspace = kspace_chopped

    # Convert to image-space and perform CutArt
    img_gibbs = np.abs(np.fft.ifftn(np.fft.ifftshift(kspace)))

    return img_gibbs


def main(path_read_data: Path, path_save_data: Path, slice_size: int, nifti_dataset: str):
    # =========
    # PATHS
    # =========
    path_nii_files = glob_nifti(path=path_read_data)[:100]

    path_save_noartifact = path_save_data / "noartifact"
    path_save_gibbs = path_save_data / "gibbs"
    for path_save_folder in (path_save_noartifact, path_save_gibbs):
        path_save_folder.mkdir(parents=True, exist_ok=True)

    # =========
    # DATAGEN
    # =========
    for subject, path_nii in enumerate(tqdm(path_nii_files[:100])):
        subject = str(subject + 1)

        # Load clean dataset and crop
        vol = data_loader.load_data(
            path_data=path_nii,
            data_format="nifti",
            normalize=True,
            nifti_dataset=nifti_dataset,
            central_50pc_crop=True,
            target_size=slice_size,
        )

        vol_gibbs = []  # Debug
        # Random Gibbs Ringing crop in range [100, 130)
        crop = np.random.randint(low=100, high=129)
        for slice_index in range(vol.shape[-1]):
            sli = vol[..., slice_index]
            sli_gibbs = gibbs(sli, crop=crop)
            sli_gibbs = cutart(sli, sli_gibbs)

            # # Debug - visualize
            # from matplotlib import pyplot as plt
            # plt.imshow(sli - sli_gibbs, cmap="gray")
            # plt.axis("off")
            # plt.show()

            vol_gibbs.append(sli_gibbs)

        vol_gibbs = np.stack(vol_gibbs, axis=-1)
        vol_gibbs = preprocessor.normalize_volume(vol_gibbs)

        for slice_index in range(25, vol.shape[-1]):
            sli = vol[..., slice_index]
            sli_gibbs = vol_gibbs[..., slice_index]

            # Save to disk
            # Make clean/subject and gibbs/subject folders
            (path_save_noartifact / subject).mkdir(parents=True, exist_ok=True)
            (path_save_gibbs / subject).mkdir(parents=True, exist_ok=True)
            np.save(arr=sli, file=str(path_save_noartifact / subject / f"{slice_index}.npy"))  # Clean
            np.save(arr=sli_gibbs, file=str(path_save_gibbs / subject / f"{slice_index}.npy"))  # Gibbs

        # # Debug - visualize
        # import sass
        # sass.scroll(vol, vol_gibbs, vol - vol_gibbs)
