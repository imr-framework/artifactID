from pathlib import Path

import numpy as np
from skimage.draw import disk
from skimage.transform import rotate
from tqdm import tqdm

from artifactID.datagen.utils import glob_nifti
from common_utils import data_loader


def cutart(sli: np.ndarray, sli_gibbs: np.ndarray) -> np.ndarray:
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
            cutart_mask = rotate(cutart_mask, angle=np.random.randint(low=-15, high=15))  # Rotate CutArt mask

            cutout = sli_gibbs[cutart_mask == 1]
            if cutout.mean() > 0.05:
                sli_gibbs_cutart[cutart_mask == 1] = cutout  # Apply CutArt
                rerun_gibbs = False  # Success, can move onto next pass
            else:
                rerun_gibbs = True  # Fail, need to rerun esp. since we only have 1 pass

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

    rr, cc = disk(center=(128, 128), radius=crop // 2, shape=img.shape)
    mask = np.zeros_like(kspace)
    mask[rr, cc] = 1
    kspace_chopped = kspace * mask

    # Convert to image-space and perform CutArt
    img_gibbs = np.abs(np.fft.ifftn(np.fft.ifftshift(kspace_chopped)))

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

        vol_gibbs = []  # Debug
        # Random Gibbs Ringing crop in range [100, 130)
        crop = np.random.randint(low=100, high=129)
        for slice_index in range(40, vol.shape[-1]):
            sli = vol[..., slice_index]
            sli_gibbs = gibbs(sli, crop=crop)
            sli_gibbs_cutart = cutart(sli, sli_gibbs)

            # # Debug - visualize
            # from matplotlib import pyplot as plt
            # plt.imshow(sli - sli_gibbs, cmap="gray")
            # plt.axis("off")
            # plt.show()

            vol_gibbs.append(sli_gibbs)

        vol_gibbs = np.stack(vol_gibbs, axis=-1)
        vol_gibbs = np.clip(vol_gibbs, a_min=0.0, a_max=1.0)

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
