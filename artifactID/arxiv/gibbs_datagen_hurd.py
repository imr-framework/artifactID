from pathlib import Path

import numpy as np
from skimage.transform import rotate

from common_utils import data_loader
from common_utils import preprocessor


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
            if cutout.mean() > 0.2:
                sli_gibbs_cutart[cutart_mask == 1] = cutout  # Apply CutArt
                rerun_gibbs = False  # Success, can move onto next pass
            else:
                rerun_gibbs = True  # Fail, need to rerun esp. since we only have 1 pass

    # Debug - visualize crop
    from matplotlib import pyplot as plt
    # sli = preprocessor.resize(sli, 128)
    # sli_gibbs = preprocessor.resize(sli_gibbs, 128)
    # sli_gibbs_cutart = preprocessor.resize(sli_gibbs_cutart, 128)

    plt.subplot(221)
    plt.imshow(sli, cmap='gray')
    plt.axis('off')

    plt.subplot(222)
    plt.imshow(sli_gibbs, cmap='gray')
    plt.axis('off')

    plt.subplot(223)
    cutart = np.zeros_like(sli)
    cutart[sli_gibbs_cutart - sli != 0] = 1
    plt.imshow(cutart, cmap='gray')
    plt.axis('off')

    plt.subplot(224)
    plt.imshow(sli_gibbs_cutart, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return sli_gibbs_cutart


def gibbs(img: np.ndarray, crop: int) -> np.ndarray:
    # Convert to k-space and crop to simulate Gibbs Ringing
    kspace = np.fft.fftshift(np.fft.fftn(img))

    # rr, cc = disk(center=(128, 128), radius=crop // 2, shape=img.shape)
    # mask = np.zeros_like(kspace)
    # mask[rr, cc] = 1
    kspace_chopped = np.zeros_like(kspace)
    kspace_chopped[128 - 50:128 + 50, 128 - 50:128 + 50] = kspace[128 - 50:128 + 50, 128 - 50:128 + 50]

    # Convert to image-space and perform CutArt
    img_gibbs = np.abs(np.fft.ifftn(np.fft.ifftshift(kspace_chopped)))

    return img_gibbs


def main(path_read_data: Path, path_save_data: Path, slice_size: int):
    # =========
    # PATHS
    # =========
    path_save_noartifact = path_save_data / "noartifact"
    path_save_gibbs = path_save_data / "gibbs"
    for path_save_folder in (path_save_noartifact, path_save_gibbs):
        path_save_folder.mkdir(parents=True, exist_ok=True)

    # =========
    # DATAGEN
    # =========
    # Load clean dataset and crop
    vol = data_loader.load_data(
        path_data=path_read_data,
        data_format="dicom",
        normalize=True,
        central_50pc_crop=True,
        target_size=slice_size,
    )

    vol_gibbs = []  # Debug
    # Random Gibbs Ringing crop in range [100, 130)
    crop = 128  # np.random.randint(low=100, high=129)
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
    # print("min", vol.min(), vol_gibbs.min())
    # print("max", vol.max(), vol_gibbs.max())
    vol_gibbs = np.clip(vol_gibbs, a_min=0.0, a_max=1.0)

    for slice_index in range(vol.shape[-1]):
        sli = vol[..., slice_index]
        sli_gibbs = vol_gibbs[..., slice_index]

        # Data augmentation
        # if np.random.uniform(low=0, high=1, size=1) > .3:
        #     random_angle = np.random.randint(low=-5, high=6)
        #     sli = transform.rotate(sli, angle=random_angle)
        #     sli_gibbs = transform.rotate(sli_gibbs, angle=random_angle)

        # Save to disk
        # Make clean/subject and gibbs/subject folders
        # np.save(arr=sli, file=str(path_save_noartifact / f"{slice_index}.npy"))  # Clean
        # np.save(arr=sli_gibbs, file=str(path_save_gibbs / f"{slice_index}.npy"))  # Gibbs

        # # Debug - visualize
        # import sass
        # sass.scroll(vol, vol_gibbs, (np.abs(vol - vol_gibbs)),
        #             cmap=['gray', 'gray', 'jet'])


if __name__ == '__main__':
    path_read_data = Path(r"D:\Sravan\Data\Source\202312-T1w_MtSinai_YasminHurd\1009902394\dicom")
    path_save_data = Path(r"D:\Sravan\Data\Source\202312-T1w_MtSinai_YasminHurd\1009902394\dicom").parent / "datagen"
    slice_size = 256
    main(
        path_read_data=path_read_data,
        path_save_data=path_save_data,
        slice_size=slice_size,
    )
