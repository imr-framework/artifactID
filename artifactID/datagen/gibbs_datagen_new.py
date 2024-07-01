from pathlib import Path
from typing import Tuple

import numpy as np
from tqdm import tqdm

from artifactID.datagen.utils import glob_nifti
from common_utils import data_loader, preprocessor


def cutart(sli: np.ndarray, sli_gibbs: np.ndarray, debug_viz: bool) -> Tuple[np.ndarray, np.ndarray]:
    final_cutart_mask = np.ones_like(sli)  # Composite mask of all CutArt patches
    sli_gibbs_cutart = sli.copy()
    passes = np.random.randint(low=1, high=4)  # Randomly apply CutArt between 1 and 3 times
    for i in range(passes):
        rerun_gibbs = True  # Flag to rerun CutArt if it fails
        while rerun_gibbs:
            # CutArt size and location parameters
            if passes == 1:
                patch_size_low = int(0.20 * sli.shape[0])
                patch_size_high = int(0.40 * sli.shape[0])
            else:
                patch_size_low = int(0.15 * sli.shape[0])
                patch_size_high = int(0.30 * sli.shape[0])
            patch_size_x = np.random.randint(low=patch_size_low, high=patch_size_high)
            patch_size_y = np.random.randint(low=patch_size_low, high=patch_size_high)
            crop_low_xy = 32
            crop_high_y = 160
            crop_y = np.random.randint(low=crop_low_xy, high=crop_high_y)
            crop_high_x = 96
            crop_x = np.random.randint(low=crop_low_xy, high=crop_high_x)

            cutart_mask = np.zeros_like(sli)
            cutart_mask[crop_x: crop_x + patch_size_x, crop_y: crop_y + patch_size_y] = 1

            cutout = sli_gibbs[cutart_mask == 1]
            if cutout.mean() > 0.1 * sli_gibbs.max():
                # Add this mask to composite final CutArt mask
                final_cutart_mask[crop_x: crop_x + patch_size_x, crop_y: crop_y + patch_size_y] = 0
                sli_gibbs_cutart[cutart_mask == 1] = cutout  # Apply CutArt
                rerun_gibbs = False  # Success, can move onto next pass
            else:
                rerun_gibbs = True  # Fail, need to rerun esp. since we only have 1 pass

    if debug_viz:
        # Debug - visualize crop
        from matplotlib import pyplot as plt
        plt.subplot(221)
        plt.imshow(sli, cmap='gray')
        plt.axis('off')

        plt.subplot(222)
        plt.imshow(sli_gibbs, cmap='gray')
        plt.axis('off')

        plt.subplot(223)
        cutart = np.ones_like(sli)
        cutart[(sli - sli_gibbs_cutart) != 0] = 0
        plt.imshow(cutart, cmap='gray')
        plt.axis('off')

        plt.subplot(224)
        plt.imshow(sli_gibbs_cutart, cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return sli_gibbs_cutart, final_cutart_mask


def gibbs(img: np.ndarray) -> np.ndarray:
    # Convert to k-space and crop to simulate Gibbs Ringing
    kspace = np.fft.fftshift(np.fft.fftn(img))
    size = kspace.shape[0]
    crop = np.random.randint(low=48, high=72)
    if np.random.randint(low=0, high=2) == 0:
        kspace[:, :crop] = 0
        kspace[:, size - crop:] = 0
    else:
        kspace[:crop] = 0
        kspace[size - crop:] = 0

    # Convert to image-space
    img_gibbs = np.abs(np.fft.ifftn(np.fft.ifftshift(kspace)))
    img_gibbs = preprocessor.normalize_per_slice(img_gibbs) * img.max()
    residual = np.abs(img - img_gibbs)
    scale = np.random.randint(low=2, high=4)
    img_gibbs += scale * residual
    img_gibbs = preprocessor.normalize_per_slice(img_gibbs) * img.max()

    return img_gibbs


def add_noise(img: np.ndarray) -> np.ndarray:
    var = np.random.uniform(low=1e-6, high=2e-5)
    std = np.sqrt(var)
    mean = 0
    noise = np.random.normal(mean, std, img.shape)
    img_noisy = img + noise

    # # Debug - visualize
    # from matplotlib import pyplot as plt
    # plt.imshow(img_noisy, cmap='gray')
    # plt.axis('off')
    # plt.show()

    return img_noisy


def main(path_read_data: Path, path_save_data: Path, slice_size: int, nifti_dataset: str, save: bool, debug_viz: bool):
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
    # =========3
    for subject, path_nii in enumerate(tqdm(path_nii_files)):
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
        for slice_index in range(vol.shape[-1]):
            sli = vol[..., slice_index]
            sli = add_noise(sli)
            vol[..., slice_index] = sli  # Replace clean with noisy in original vol
        vol = preprocessor.normalize_volume(vol)

        vol_gibbs = []  # Debug
        # Random Gibbs Ringing crop in range [100, 130)
        for slice_index in range(vol.shape[-1]):
            sli = vol[..., slice_index]
            sli_gibbs = gibbs(sli)
            sli_gibbs_cutart, _ = cutart(sli, sli_gibbs, debug_viz)

            # # Debug - visualize
            # from matplotlib import pyplot as plt
            # plt.imshow(sli - sli_gibbs, cmap="gray")
            # plt.axis("off")
            # plt.show()

            vol_gibbs.append(sli_gibbs_cutart)

        vol_gibbs = np.stack(vol_gibbs, axis=-1)
        vol_gibbs = np.clip(vol_gibbs, a_min=0.0, a_max=1.0)

        if save:
            # Save to disk
            for slice_index in range(vol.shape[-1]):
                sli = vol[..., slice_index]
                sli_gibbs = vol_gibbs[..., slice_index]

                # Save to disk - make clean/subject and gibbs/subject folders
                (path_save_noartifact / subject).mkdir(parents=True, exist_ok=True)
                (path_save_gibbs / subject).mkdir(parents=True, exist_ok=True)
                np.save(arr=sli, file=str(path_save_noartifact / subject / f"{slice_index}.npy"))  # Clean
                np.save(arr=sli_gibbs, file=str(path_save_gibbs / subject / f"{slice_index}.npy"))  # Gibbs

        # # Debug - visualize
        # import sass
        # sass.scroll(vol, vol_gibbs, (np.abs(vol - vol_gibbs)),
        #             cmap=['gray', 'gray', 'jet'])
