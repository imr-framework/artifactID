from datetime import datetime
from pathlib import Path

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


def __viz_heatmap(heatmap: np.ndarray, patch_size: int, vol: np.ndarray, y_pred: np.ndarray):
    if not isinstance(patch_size, list):
        patch_size = [patch_size] * 3

    # Place predictions onto heatmap
    np.place(arr=heatmap, mask=heatmap, vals=y_pred)

    # Upscale heatmap
    for counter, p in enumerate(patch_size):
        heatmap = np.repeat(heatmap, p, axis=counter)

    sass.scroll_mask(volume=vol, mask=heatmap, mask_vmin=0, mask_vmax=6)  # Plot


def main(arr_vols, arr_files_folders, path_read_data: str, path_model_root: str, patch_size, viz: bool = True):
    # =========
    # DATA PRE-PROCESSING
    # =========
    print('Pre-processing data...')
    # Zero-pad vol, get patches, discard empty patches and uniformly intense patches and normalize each patch
    arr_patches = []
    arr_patch_maps = []
    for i, vol in enumerate(arr_vols):
        vol = data_ops.patch_compatible_zeropad(vol=vol, patch_size=patch_size)
        arr_vols[i] = vol  # Replace old vol with padded vol
        patches, patch_map = data_ops.get_patches(vol=vol, patch_size=patch_size)
        patches = data_ops.normalize_patches(patches=patches)
        arr_patches.append(patches)
        arr_patch_maps.append(patch_map)

    # =========
    # SET UP TESTING
    # =========
    path_model_root = Path(path_model_root)
    path_model_load = path_model_root / 'model.hdf5'  # Keras model
    path_model_log = path_model_root / 'log.txt'  # Log file
    print('Loading model...')
    model = load_model(path_model_load)  # Load model
    log = open(str(path_model_log), 'r').readlines()  # Read log file

    # Make generator for feeding data to the model
    def make_generator(patches):
        for p in patches:
            p = p.astype(np.float16)
            p = np.expand_dims(a=p, axis=3)
            yield p

    # Dictionary to map integer predictions back to labels
    dict_label_int = __get_dict_from_log(log=log)
    dict_label_int = eval(dict_label_int)  # Convert str representation of dict into dict object
    dict_int_label = dict(zip(dict_label_int.values(), dict_label_int.keys()))

    output_shape = tf.TensorShape(patch_size + [1])

    # =========
    # TESTING
    # =========
    print(f'Performing inference...')
    dict_path_pred = dict()
    for i in range(len(arr_vols)):  # Inference on each volume
        folder = arr_files_folders[i]  # To make a dictionary of files/folders-predictions
        patches = arr_patches[i]
        patch_map = arr_patch_maps[i]
        vol = arr_vols[i]

        # Make dataset from generator
        dataset = tf.data.Dataset.from_generator(generator=make_generator,
                                                 args=[patches],
                                                 output_types=(tf.float16),
                                                 output_shapes=(output_shape)).batch(batch_size=32)

        # Inference
        y_pred = model.predict(x=dataset)
        y_pred = np.argmax(y_pred, axis=1).astype(np.int32)

        # Convert integer outputs to labels
        y_pred_unique = np.unique(y_pred)
        y_pred_label = [dict_int_label[pred] for pred in y_pred_unique]

        dict_path_pred[folder] = y_pred_label

        # Construct and visualize heatmap
        if viz:
            __viz_heatmap(heatmap=patch_map, patch_size=patch_size, vol=vol, y_pred=y_pred)

    # =========
    # SAVE PREDICTIONS TO DISK
    # =========
    print('Saving predictions to disk...')
    with open(str(path_model_root / 'test_log.txt'), 'a') as f:
        time_string = datetime.now().strftime('%y%m%d_%H%M')  # Time stamp when saving to log
        write = time_string + '\n'
        write += str(path_model_load) + ' \n'
        write += str(path_read_data) + '\n'
        for key, value in dict_path_pred.items():
            key = Path(key).relative_to(path_read_data.parent)
            write += f'{key}: {value}\n'
        write += '=========\n\n'
        f.write(write)
    print('Done')
