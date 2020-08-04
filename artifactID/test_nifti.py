import configparser
from pathlib import Path

import nibabel as nb
import numpy as np
from tqdm import tqdm

import artifactID.test
from artifactID.common import data_ops


def main(path_read_data: str, path_model_root: str, patch_size, viz: bool = True):
    # =========
    # DATA LOADING
    # =========
    path_read_data = Path(path_read_data)
    arr_files = list(path_read_data.glob('**/*'))  # Get all files and folders
    arr_files = list(filter(lambda isfile: isfile.is_file(), arr_files))  # Filter for files

    # Read NIFTIs
    print('Reading NIFTIs...')
    arr_vols = []
    for f in tqdm(arr_files):
        npy = nb.load(f).get_fdata()
        # For HCP, move axis
        npy = np.swapaxes(npy, 1, 2)
        arr_vols.append(npy)

    # Perform inference
    artifactID.test.main(arr_files_folders=arr_files, arr_vols=arr_vols, path_read_data=path_read_data,
                         path_model_root=path_model_root, patch_size=patch_size, viz=viz)


if __name__ == '__main__':
    # Read settings.ini configuration file
    path_settings = 'settings.ini'
    config = configparser.ConfigParser()
    config.read(path_settings)

    config_eval = config['TEST']
    patch_size = config_eval['patch_size']
    patch_size = data_ops.get_patch_size_from_config(patch_size=patch_size)
    path_read_data = config_eval['path_read_data']
    path_pretrained_model = config_eval['path_pretrained_model']

    if not Path(path_read_data).exists():
        raise Exception(f'{path_read_data} does not exist')
    if not Path(path_pretrained_model).exists():
        raise Exception(f'{path_pretrained_model} does not exist')

    main(patch_size=patch_size, path_model_root=path_pretrained_model, path_read_data=path_read_data, viz=True)
