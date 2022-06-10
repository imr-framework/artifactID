if __name__ == '__main__':
    import sys
    import os

    script_path = os.path.abspath(__file__)
    SEARCH_PATH = script_path[:script_path.index('artifactID') + len('artifactID') + 1]
    sys.path.insert(0, SEARCH_PATH)

import configparser
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pydicom

from common_utils import preprocessor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from artifactID.gradcam import make_gradcam_heatmap
from artifactID.utils import dcm2npy, superimpose_dcm
from model_wrapper import ModelWrapper

TIME_SLEEP = 10  # seconds

# =========
# TENSORFLOW INIT
# =========
# Prevent crash
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

# =========
input_shape = (256, 256)

model_wa_xy: ModelWrapper = None
model_wa_z: ModelWrapper = None
model_g: ModelWrapper = None


# =========
def check_data_folders(path_dicom_input: Path, path_dicom_output: Path, path_dicom_processed: Path):
    if not path_dicom_input.is_dir() or not path_dicom_input.exists():
        raise FileNotFoundError(f'{path_dicom_input} is not a directory or does not exist.')

    if not path_dicom_output.exists():
        print(f'{path_dicom_output} not found, making folder...')
        path_dicom_output.mkdir(parents=False)

    if not path_dicom_processed.exists():
        print(f'{path_dicom_processed} not found, making folder...')
        path_dicom_processed.mkdir(parents=False)


def move_folder(src: Path, dst: Path):
    try:
        shutil.move(src=str(src), dst=str(dst))
    except:  # Catch crash if destination folder already exists
        # We want to save anyway, so rename and append timestamp
        from time import time
        new_dst = dst.with_name(f'{dst.stem}_{time()}').with_suffix(dst.suffix)
        shutil.move(src=str(src), dst=str(new_dst))

    print(f'{src} has been moved to {dst}.')


def get_heatmap(arr_npy_artifact: np.ndarray, artifact_type: str, model: ModelWrapper, input_shape):
    arr_heatmaps = []
    for i in range(len(arr_npy_artifact)):
        npy_slice = arr_npy_artifact[i]

        if artifact_type != 'Gibbs ringing':
            keras_model = model.get_keras_model()
            heatmap = make_gradcam_heatmap(npy_slice, keras_model)
            heatmap = cv2.resize(heatmap, input_shape)

            arr_heatmaps.append(heatmap)

    return arr_heatmaps


def read_settings(path_settings: str) -> (Path, Path, Path, Path, Path, Path):
    config = configparser.ConfigParser()
    config.read(path_settings)

    config_data = config['DATA']
    path_dicom_input = Path(config_data['input'])
    path_dicom_output = Path(config_data['output'])
    path_dicom_processed = Path(config_data['processed'])

    config_model = config['MODEL']
    path_model_wraparound_xy = Path(config_model['path_model_wraparound_xy'])
    path_model_wraparound_z = Path(config_model['path_model_wraparound_z'])
    path_model_gibbs = Path(config_model['path_model_gibbs'])

    return path_dicom_input, path_dicom_output, path_dicom_processed, path_model_wraparound_xy, \
           path_model_wraparound_z, path_model_gibbs


def main_dir(path_dicom_dir: Path, paths: list):
    path_dicom_input, path_dicom_output, path_dicom_processed = paths[:3]

    # =========
    # READ DATA
    # =========
    # Convert all DICOMs into a single .npy file
    print(f'Reading DICOMs in {path_dicom_dir.stem}...')
    dict_fnames_dcm_npy = dcm2npy(target_shape=input_shape, path_dicom_folder=path_dicom_dir)
    arr_npy_input = dict_fnames_dcm_npy['npy']
    # TODO remove
    # arr_npy_input = preprocessor.mask_subject(np.swapaxes(arr_npy_input, 0, 2))
    # arr_npy_input = np.swapaxes(arr_npy_input, 2, 0)
    # TODO remove
    arr_npy_ksp_input = np.abs(np.fft.fftshift(np.fft.fftn(arr_npy_input))).astype(np.float16)

    # =========
    # INFERENCE
    # =========
    # Run predictions
    print('Performing inference...')
    y_pred_wa_xy = model_wa_xy.predict_batch(arr_npy_input)
    y_pred_wa_z = model_wa_z.predict_batch(arr_npy_input)
    y_pred_g = model_g.predict_batch(arr_npy_ksp_input)

    # Idx of artifacts
    idx_wa_xy = np.where(y_pred_wa_xy == 1)[0]
    idx_wa_z = np.where(y_pred_wa_z == 1)[0]
    idx_g = np.where(y_pred_g == 1)[0]

    # Number of artifacts
    n_wa_xy = len(idx_wa_xy)
    n_wa_z = len(idx_wa_z)
    n_g = len(idx_g)

    # =========
    # PRINT RESULTS, RECOMMENDATIONS
    # =========
    print()
    print(f'Number of slices in dataset *possibly* containing in-plane wrap around: {n_wa_xy}')
    if n_wa_xy:
        print('--> Rescan subject with larger FOV (xy-direction) if possible')
    print(f'Number of slices in dataset *possibly* containing through-plane wrap around: {n_wa_z}')
    if n_wa_z:
        print('--> Rescan subject with larger FOV (z-direction) or with slice oversampling if possible')
    print(f'Number of slices in dataset *possibly* containing Gibbs ringing: {n_g}')
    if n_g > 0:
        print('--> Rescan subject with increased number of phase encodes if possible')

    # =========
    # HEATMAPS FOR WRAP-AROUND
    # =========
    print('\nGenerating heatmaps...')
    if n_wa_xy != 0:
        slices_wa_xy = arr_npy_input[idx_wa_xy]  # Extract slices which have the artifact
        heatmaps = get_heatmap(model=model_wa_xy, input_shape=input_shape,
                               artifact_type='In-plane wrap around',
                               arr_npy_artifact=slices_wa_xy)
        heatmaps_wa_xy = np.zeros_like(arr_npy_input)
        heatmaps_wa_xy[idx_wa_xy] = heatmaps
    if n_wa_z != 0:
        slices_wa_z = arr_npy_input[idx_wa_z]  # Extract slices which have the artifact
        heatmaps = get_heatmap(model=model_wa_z, input_shape=input_shape,
                               artifact_type='Through-plane wrap around',
                               arr_npy_artifact=slices_wa_z)
        heatmaps_wa_z = np.zeros_like(arr_npy_input)
        heatmaps_wa_z[idx_wa_z] = heatmaps

    # =========
    # WRITE TO DISK
    # =========
    print('Saving DICOMs to disk...')
    path_processed = path_dicom_processed / path_dicom_dir.stem
    if not path_processed.exists():
        path_processed.mkdir(parents=False)
    for artifact_type, arr_artifact_idx in [['gibbs_ringing', idx_g],
                                            ['wraparound_xy', idx_wa_xy],
                                            ['wraparound_z', idx_wa_z]]:

        if len(arr_artifact_idx) > 0:  # Were there any artifacts?
            # Make save directory
            path_save = path_processed / artifact_type
            if not path_save.exists():
                path_save.mkdir(parents=False)

            # Write DICOM slices to disk
            arr_dcm_filenames = dict_fnames_dcm_npy['filenames']
            arr_dcm = dict_fnames_dcm_npy['dcm']
            for slice_id in arr_artifact_idx:
                path_save2 = path_save / arr_dcm_filenames[slice_id].name
                dcm = arr_dcm[slice_id]
                if artifact_type == 'gibbs_ringing':
                    superimposed_dcm = dcm  # Gibbs ringing has no heatmaps
                else:
                    if artifact_type == 'wraparound_xy':
                        heatmap = heatmaps_wa_xy[slice_id]
                    elif artifact_type == 'wraparound_z':
                        heatmap = heatmaps_wa_z[slice_id]
                    superimposed_dcm = superimpose_dcm(dcm=dcm, img=arr_npy_input[slice_id], heatmap=heatmap)
                pydicom.dcmwrite(path_save2, superimposed_dcm)

    # =========
    # MOVE FOLDER TO OUTPUT DIRECTORY
    # =========
    path_save = path_dicom_output / path_dicom_dir.stem
    move_folder(src=path_dicom_dir, dst=path_save)


def main(path_settings: str):
    paths = read_settings(path_settings)
    check_data_folders(*paths[:3])  # Check and make destination folders if required
    path_dicom_input = paths[0]
    path_model_wraparound_xy, path_model_wraparound_z, path_model_gibbs = paths[3:]

    # Load Keras models using wrappers
    print('Loading Keras models...')
    global model_wa_xy
    global model_wa_z
    global model_g
    model_wa_xy = ModelWrapper(path_model_wraparound_xy)
    model_wa_z = ModelWrapper(path_model_wraparound_z)
    model_g = ModelWrapper(path_model_gibbs)
    print('Done.\n')

    print('Waiting for new DICOM data... press Ctrl-c to quit')
    while True:
        for dicom_dir in path_dicom_input.iterdir():
            if dicom_dir.is_dir():
                print(f'Found {dicom_dir.stem}')
                input('Press any key to begin processing...')  # Wait for user input TODO
                main_dir(dicom_dir, paths)
                print('=========\n')
                print('Waiting for new DICOM data... press Ctrl-c to quit')
        time.sleep(TIME_SLEEP)  # Seconds


if __name__ == '__main__':
    path_settings = 'settings.ini'
    main(path_settings)
