import os
from pathlib import Path

import nibabel as nb
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def _download_if_missing(url, unzip_src):
    if not os.path.exists(unzip_src):
        print(f'Downloading {unzip_src}...')
        tf.keras.utils.get_file(unzip_src, origin=url, extract=False)
    else:
        print(f'{unzip_src} already present, skipping download...')


def download_from_storage_bucket():
    """Download data from Google storage bucket"""
    data_root = Path(r"C:\Users\sravan953\Documents\CU\Projects\imr-framework\ArtifactID\Data")
    folders = ['noartifact', 'snr', 'b0_inhomogeneity', 'wraparound']
    for f in tqdm(folders):
        filename = f + '_data.zip'
        url = 'https://storage.googleapis.com/adl_a4_preprocessing/' + filename
        unzip_src = data_root / filename
        _download_if_missing(url=url, unzip_src=unzip_src)


def convert_miccai_npy():
    # No artifact files are NIFTI1
    # Re-orient each brain and save as .npy
    # Move no-artifact subjects into data folder
    common_data = Path(r'C:\Users\sravan953\Documents\CU\Projects\imr-framework\ArtifactID\Data\data')
    if not (common_data / 'noartifact').exists():  # Make data/noartifact first
        (common_data / 'noartifact').mkdir()
    miccai = Path(r'C:\Users\sravan953\Documents\CU\Projects\imr-framework\DART\Data\MICCAI_BraTS_2018_Data_Training')
    subjects = list(miccai.glob('**/*.nii.gz'))
    subjects = list(filter(lambda x: 't1.nii.gz' in str(x), subjects))
    print(f'{len(subjects)} subjects in noartifact')
    for s in tqdm(subjects):
        vol = nb.load(s).get_fdata()
        vol = np.rot90(vol, -1, axes=(0, 1))
        s = s.name
        s = s.replace('nii.gz', 'npy')
        dst = common_data / 'noartifact' / s
        # Normalize volume to [0,1]
        max = vol.max()
        min = vol.min()
        vol = (vol - min) / (max - min)
        np.save(str(dst), vol)


def shape_check():
    # Randomly check a few subjects to see if zero-padding was done right on 10 random subjects
    random_files = list(Path('data').glob('**/*.npy'))
    random_files = np.random.choice(random_files, size=10)
    random_files = [np.load(str(rf)) for rf in random_files]
    required_shape = (240, 240, 155)
    shapes_match = np.all([rf.shape == required_shape for rf in random_files])
    print(f'10 randomly sampled shapes are matching: {shapes_match}')


def min_max_check():
    data_root = Path(r"C:\Users\sravan953\Documents\CU\Projects\imr-framework\ArtifactID\Data\data")
    all_max = dict()
    all_min = dict()

    folders = list(data_root.glob('wrap*'))
    for folder in tqdm(folders):
        files = list(folder.glob('*.npy'))
        for f in files:
            vol = np.load(f)
            key = f.parts[-2] + '/' + f.parts[-1]
            all_max[key] = vol.max()
            all_min[key] = vol.min()

    max = np.all(np.array(list(all_max.values())) == 1)
    if not max:
        idx = np.where(np.array(list(all_max.values())) != 1)
        print(np.array(list(all_max.keys()))[idx[:10]])
    min = np.all(np.array(list(all_min.values())) == 0)
    if not min:
        idx = np.where(np.array(list(all_min.values())) != 0)
        print(np.array(list(all_min.keys()))[idx[:10]])
    print(f'Max: {max}')
    print(f'Min: {min}')


# convert_miccai_npy()
# min_max_check()
