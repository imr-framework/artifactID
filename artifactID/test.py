import configparser
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

import sass
from artifactID.common import data_ops

# =========
# TENSORFLOW INIT
# =========
# Prevent crash
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)


def __get_dict_from_log(log: list):
    for line in log:
        if '{' in line and '}' in line:
            return line


def __make_heatmap(patch_map: np.ndarray, patch_size: int, vol: np.ndarray, y_pred: np.ndarray):
    vol = (vol - vol.min()) / (vol.max() - vol.min())  # Normalize volume

    all_maps = []
    for artifact in y_pred.T:  # Iterate through each prediction class across all patches
        artifact_map = np.zeros_like(patch_map, dtype=np.float16)
        np.place(arr=artifact_map, mask=patch_map, vals=artifact)  # Place predictions onto heatmap

        for counter, p in enumerate((patch_size, patch_size, 1)):  # Upscale to original resolution
            artifact_map = np.repeat(artifact_map, p, axis=counter)
        all_maps.append(artifact_map)

    # 1x2 film of volumes
    film_vol = np.concatenate((vol, vol), axis=1)
    _row2 = np.concatenate((vol, vol, vol), axis=1)
    film_vol = np.concatenate((_row1, _row2), axis=0)

    # 1x2 film of mask overlays; first overlay is zeros to show the original volume as is
    film_mask = np.concatenate((np.zeros_like(vol), all_maps[0]), axis=1)
    _row2 = np.concatenate((all_maps[2], all_maps[3], all_maps[4]), axis=1)
    film_mask = np.concatenate((_row1, _row2), axis=0)

    # Construct alpha
    num_classes = len(y_pred[0])
    threshold = 1 / num_classes
    film_alpha = np.zeros_like(film_vol)
    film_alpha[film_mask > 0] = 0.40

    return film_vol, film_mask, film_alpha


def __show_heatmap(film_vol: np.ndarray, film_mask: np.ndarray, film_alpha: np.ndarray):
    # Convert to float64 for matplotlib
    film_alpha = film_alpha.astype(np.float)
    film_mask = film_mask.astype(np.float)
    film_vol = film_vol.astype(np.float)
    sass.scroll_mask(film_vol, mask=film_mask, alpha=film_alpha)


def main(arr_files: List[Path], batch_size: int, format: str, path_pretrained_model: Path, path_read_data: Path,
         input_shape: int, save: bool, viz: bool = True):
    # =========
    # SET UP TESTING
    # =========
    path_model_load = path_pretrained_model / 'model.hdf5'  # Keras model
    path_model_log = path_pretrained_model / 'log.txt'  # Log file
    print('Loading model...')
    model = load_model(str(path_model_load))  # Load model
    log = open(str(path_model_log), 'r').readlines()  # Read log file

    # Make generator for feeding data to the model
    def __generator_patches(patches: list):
        for p in patches:
            p = p.astype(np.float16)
            p = np.expand_dims(a=p, axis=2)
            yield {'input_1': p, 'input_2': p}

    # Dictionary to map integer predictions back to labels
    dict_label_int = __get_dict_from_log(log=log)
    dict_label_int = eval(dict_label_int)  # Convert str representation of dict into dict object
    dict_int_label = dict(zip(dict_label_int.values(), dict_label_int.keys()))

    # =========
    # TESTING
    # =========
    print(f'Performing inference...')
    dict_path_pred = dict()
    input_output_shape = (input_shape, input_shape, 1)
    output_types = (tf.float16)
    output_shapes = (tf.TensorShape(input_output_shape))
    for counter, vol in enumerate(data_ops.generator_inference(x=arr_files, file_format=format)):
        print(arr_files[counter])
        vol = data_ops.resize(vol=vol, size=input_shape)
        vol_normalized = data_ops.normalize_slices(vol)
        vol_reshaped = np.moveaxis(vol_normalized, (0, 1, 2), (1, 2, 0))
        vol_reshaped = np.expand_dims(vol_reshaped, 3)
        folder = arr_files[counter].parent.parts[-1]  # To make a dictionary of files/folders-predictions

        # Make dataset from generator
        dataset = tf.data.Dataset.from_tensor_slices(vol_reshaped).batch(batch_size=batch_size)

        # Inference
        y_pred = model.predict(x=dataset)

        # Convert integer outputs to labels and map results
        y_pred_unique = np.unique(np.argmax(y_pred, axis=1))
        y_pred_label = [dict_int_label[pred] for pred in y_pred_unique]
        dict_path_pred[folder] = y_pred_label

        # =========
        # SAVE TO DISK AND/OR VIZ. PREDICTIONS
        # =========
        sass.scroll(vol_normalized, labels=[np.squeeze(y_pred)])
        if save or viz:
            results = __make_heatmap(patch_map=patch_map, patch_size=patch_size, vol=vol, y_pred=y_pred)
            film_vol, film_mask, film_alpha = results

            if save:
                path_save = arr_files[counter].parent
                np.save(arr=film_vol, file=str(path_save / 'vol.npy'))
                np.save(arr=film_mask, file=str(path_save / 'mask.npy'))
                np.save(arr=film_alpha, file=str(path_save / 'alpha.npy'))
            if viz:
                __show_heatmap(film_vol=film_vol, film_mask=film_mask, film_alpha=film_alpha)

    # =========
    # SAVE STATS TO DISK
    # =========
    print('Updating test log...')
    with open(str(path_pretrained_model / 'test_log.txt'), 'a') as f:
        time_string = datetime.now().strftime('%y%m%d_%H%M')  # Time stamp when saving to log
        write = time_string + '\n'
        write += str(path_model_load) + ' \n'
        write += str(path_read_data) + '\n'
        for key, value in dict_path_pred.items():
            write += f'{key}: {value}\n'
        write += '=========\n\n'
        f.write(write)
    print('Done')


if __name__ == '__main__':
    # =========
    # READ CONFIG
    # =========
    # Read settings.ini configuration file
    path_settings = 'settings.ini'
    config = configparser.ConfigParser()
    config.read(path_settings)

    config_test = config['TEST']
    batch_size = int(config_test['batch_size'])
    input_shape = int(config_test['input_shape'])  # Patch size
    path_pretrained_model = config_test['path_pretrained_model']
    path_read_data = config_test['path_read_data']
    save = bool(config_test['save'])

    # =========
    # DATA CHECK
    # =========
    path_read_data = Path(path_read_data)
    if not path_read_data.exists():
        raise Exception(f'{path_read_data} does not exist')
    path_pretrained_model = Path(path_pretrained_model)
    if not path_pretrained_model.exists():
        raise Exception(f'{path_pretrained_model} does not exist')

    arr_files = data_ops.glob_nifti(path=path_read_data)
    format = 'nifti'
    if len(arr_files) == 0:
        arr_files = data_ops.glob_dicom(path=path_read_data)
        format = 'dicom'
    if len(arr_files) == 0:
        raise ValueError(f'No NIFTI or DICOM files found at {path_read_data}')

    # Perform inference
    main(arr_files=arr_files,
         batch_size=batch_size,
         format=format,
         input_shape=input_shape,
         path_read_data=path_read_data,
         path_pretrained_model=path_pretrained_model,
         save=save,
         viz=True)
