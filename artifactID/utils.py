from pathlib import Path

import cv2
import numpy as np
import pydicom


def dcm2npy(target_shape: tuple, path_dicom_folder: Path) -> dict:
    # Check if DICOM folder exists
    if path_dicom_folder.exists() and path_dicom_folder.is_dir():
        ext = ['*.dcm', '*.ima']
        arr_dcm_filenames = []
        [arr_dcm_filenames.extend(list(path_dicom_folder.glob(e))) for e in ext]
    else:
        raise FileNotFoundError(f'{path_dicom_folder} does not exist or is not a valid directory.')

    arr_dcm = []
    arr_npy = np.empty((len(arr_dcm_filenames), *target_shape))
    idx_to_remove = []  # Invalid DICOMs to remove
    for i, f in enumerate(arr_dcm_filenames):
        dcm = pydicom.dcmread(f)  # Read the dicom
        if 'PixelData' in dcm:  # Sometimes, DICOMs do not have image data
            arr_dcm.append(dcm)
            dcm2npy = dcm.pixel_array  # Save it as np array
            npy_resized = cv2.resize(dcm2npy, target_shape)  # Resize to match the model's input shape
            npy_norm = (npy_resized - npy_resized.min()) / (npy_resized.max() - npy_resized.min())  # Normalize [0,1]
            arr_npy[i] = npy_norm
        else:
            idx_to_remove.append(i)

    arr_dcm = np.stack(arr_dcm)
    arr_dcm_filenames = np.stack(arr_dcm_filenames)
    arr_npy = arr_npy.astype(np.float16)

    arr_dcm_filenames = np.delete(arr_dcm_filenames, idx_to_remove)  # Cleanup - remove invalid DICOM filenames

    dict_fnames_dcm_npy = {'filenames': arr_dcm_filenames,
                           'dcm': arr_dcm,
                           'npy': arr_npy}
    return dict_fnames_dcm_npy


def superimpose_dcm(img: np.ndarray, dcm: pydicom.dataset.FileDataset,
                    heatmap: np.ndarray) -> pydicom.dataset.FileDataset:
    target_size = dcm.pixel_array.shape
    if len(target_size) != 2:
        target_size = target_size[:2]
    # Convert img to RGB
    img = (img * 255).astype(np.float32)
    img = cv2.resize(src=img, dsize=(target_size[1], target_size[0]))
    img_rgb = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_GRAY2RGB)

    # Normalize heatmap to lie in range [0, 255] of uint8 datatype
    heatmap_quantized = np.array(heatmap * 255, dtype=np.uint8)
    heatmap_resized = cv2.resize(heatmap_quantized, (target_size[1], target_size[0]))
    # Convert heatmap to jet colormap
    heatmap_rgb = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET).astype(np.float32)

    # Superimpose colorized heatmap on RGB input image
    superimposed = cv2.addWeighted(img_rgb, 0.75, heatmap_rgb, 0.25, 0).astype('uint8')

    # Modify DICOM tags
    dcm.PhotometricInterpretation = 'RGB'
    dcm.SamplesPerPixel = 3
    dcm.BitsAllocated = 8
    dcm.BitsStored = 8
    dcm.HighBit = 7
    dcm.add_new(0x00280006, 'US', 0)
    dcm.is_little_endian = True
    dcm.fix_meta_info()

    # Save pixel data and dicom file
    dcm.PixelData = img.tobytes()
    dcm.PixelData = superimposed.tobytes()

    return dcm
