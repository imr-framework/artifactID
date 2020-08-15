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
tf.config.experimental.set_memory_growth(gpus[0], True)


def __get_dict_from_log(log: list):
    for line in log:
        if '{' in line and '}' in line:
            return line


def __make_heatmap(patch_map: np.ndarray, patch_size: list, vol: np.ndarray, y_pred: np.ndarray):
    vol = vol.astype(np.float16)
    vol = (vol - vol.min()) / (vol.max() - vol.min())  # Normalize volume

    all_maps = []
    for artifact in y_pred.T:  # Iterate through each prediction class across all patches
        artifact_map = np.empty(shape=patch_map.shape, dtype=np.float16)
        np.place(arr=artifact_map, mask=patch_map, vals=artifact)  # Place predictions onto heatmap

        for counter, p in enumerate(patch_size):  # Upscale to original resolution
            artifact_map = np.repeat(artifact_map, p, axis=counter)
        all_maps.append(artifact_map)

    # 2x4 film of volumes
    _row1 = np.concatenate((vol, vol, vol, vol), axis=1)
    _row2 = np.concatenate((vol, vol, vol, vol), axis=1)
    film_vol = np.concatenate((_row1, _row2), axis=0).astype(np.float16)

    # 2x4 film of mask overlays; first overlay is zeros to show the original volume as is
    _row1 = np.concatenate((np.zeros_like(vol), all_maps[0], all_maps[1], all_maps[2]), axis=1)
    _row2 = np.concatenate((all_maps[3], all_maps[4], all_maps[5], all_maps[6]), axis=1)
    film_mask = np.concatenate((_row1, _row2), axis=0).astype(np.float16)

    # Construct alpha
    num_classes = len(y_pred[0])
    threshold = 1 / num_classes
    film_alpha = np.zeros_like(film_vol, dtype=np.float16)
    film_alpha[film_mask >= threshold] = 0.5

    return film_vol, film_mask, film_alpha


def __show_heatmap(film_vol: np.ndarray, film_mask: np.ndarray, film_alpha: np.ndarray):
    sass.scroll_mask(film_vol, mask=film_mask, alpha=film_alpha)


def main(arr_files: List[Path], batch_size: int, format: str, path_pretrained_model: Path, path_read_data: Path,
         patch_size: list, save: bool, viz: bool = True):
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
            p = np.expand_dims(a=p, axis=3)
            yield p

    # Dictionary to map integer predictions back to labels
    dict_label_int = __get_dict_from_log(log=log)
    dict_label_int = eval(dict_label_int)  # Convert str representation of dict into dict object
    dict_int_label = dict(zip(dict_label_int.values(), dict_label_int.keys()))

    # =========
    # TESTING
    # =========
    print(f'Performing inference...')
    dict_path_pred = dict()
    output_shape = tf.TensorShape(patch_size + [1])
    for counter, vol in enumerate(data_ops.generator_inference(x=arr_files, file_format=format)):
        print(arr_files[counter])
        vol = data_ops.patch_compatible_zeropad(vol=vol, patch_size=patch_size)
        patches, patch_map = data_ops.get_patches(vol=vol, patch_size=patch_size)
        patches = data_ops.normalize_patches(patches=patches)
        folder = arr_files[counter].parent.parts[-1]  # To make a dictionary of files/folders-predictions

        # Make dataset from generator
        dataset = tf.data.Dataset.from_generator(generator=__generator_patches,
                                                 args=[patches],
                                                 output_types=tf.float16,
                                                 output_shapes=output_shape).batch(batch_size=batch_size)

        # Inference
        y_pred = model.predict(x=dataset)

        # Convert integer outputs to labels and map results
        y_pred_unique = np.unique(np.argmax(y_pred, axis=1))
        y_pred_label = [dict_int_label[pred] for pred in y_pred_unique]
        dict_path_pred[folder] = y_pred_label

        # =========
        # SAVE TO DISK AND/OR VIZ. PREDICTIONS
        # =========
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

    config_eval = config['TEST']
    batch_size = int(config_eval['batch_size'])
    patch_size = config_eval['patch_size']
    patch_size = data_ops.get_patch_size_from_config(patch_size=patch_size)
    path_read_data = config_eval['path_read_data']
    path_pretrained_model = config_eval['path_pretrained_model']
    save = bool(config_eval['save'])

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
        raise ValueError(f'No NIFTI or DICOM files found at f{path_read_data}')

    # Perform inference
    main(arr_files=arr_files[:3], batch_size=batch_size, format=format, path_read_data=path_read_data,
         path_pretrained_model=path_pretrained_model, patch_size=patch_size, save=save, viz=False)
