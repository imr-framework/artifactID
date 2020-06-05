import configparser
import itertools
import math
from pathlib import Path

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

from artifactID.common.data_ops import get_paths, make_generator_inference

# =========
# TENSORFLOW INIT
# =========
# Prevent crash
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def _viz(dict_path_pred):
    counter = 0
    arr_x = []
    arr_y_pred = []
    for path, pred in dict_path_pred.items():
        if counter < 6:  # Hardcode to 6 plots
            vol = np.load(path).astype(np.float)
            arr_x.append(vol)
            arr_y_pred.append(pred)
        else:
            break
        counter += 1

    # Get center slice number
    shape = arr_x[0].shape
    center_slice = np.squeeze(shape)[2] // 2

    for counter, vol in enumerate(arr_x):
        plt.subplot(2, 3, counter + 1)
        plt.imshow(vol[:, :, center_slice], cmap='gray')
        text = f'Pred: {arr_y_pred[counter]}'
        plt.title(text)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def main(data_root: str, aid_main_path: str, aid_wrap_path: str, aid_snr_path: str):
    # =========
    # DATA LOADING
    # =========
    # Get paths and labels
    test_x_paths = get_paths(data_root=data_root)

    # =========
    # INFERENCE - MAIN
    # =========
    batch_size = 1
    main_generator = tf.data.Dataset.from_generator(generator=make_generator_inference,
                                                    args=[test_x_paths],
                                                    output_types=(tf.float16),
                                                    output_shapes=(tf.TensorShape([240, 240, 155, 1]))).batch(
        batch_size=batch_size)

    print('Loading all models...')
    aid_main_model = load_model(aid_main_path)
    aid_wrap_model = load_model(aid_wrap_path)
    aid_snr_model = load_model(aid_snr_path)

    print(f'\nPerforming inference on {len(test_x_paths)} samples...')
    main_steps_per_epoch = math.ceil(len(test_x_paths) / batch_size)
    y_pred_main = aid_main_model.predict(x=main_generator, steps=main_steps_per_epoch)
    y_pred_main = np.argmax(y_pred_main, axis=1)
    dict_path_pred_general = dict(zip(test_x_paths, y_pred_main))

    # Determine samples to be sub-classified
    todo_snr = []
    todo_wrap = []
    dict_label_int_general = {'b0_': 0, 'noartifact': 1, 'snr': 2, 'wrap': 3}
    for counter in y_pred_main:
        if y_pred_main[counter] == dict_label_int_general['snr']:
            todo_snr.append(test_x_paths[counter])
        elif y_pred_main[counter] == dict_label_int_general['wrap']:
            todo_wrap.append(test_x_paths[counter])

    # =========
    # INFERENCE - WRAP
    # =========
    dict_path_pred_wrap = dict()
    if len(todo_wrap) != 0:
        wrap_generator = tf.data.Dataset.from_generator(generator=make_generator_inference,
                                                        args=[todo_wrap],
                                                        output_types=(tf.float16),
                                                        output_shapes=(tf.TensorShape([240, 240, 155, 1]))).batch(
            batch_size=batch_size)

        wrap_steps_per_epoch = math.ceil(len(todo_wrap) / batch_size)
        y_pred_wrap = aid_wrap_model.predict(x=wrap_generator, steps=wrap_steps_per_epoch)
        y_pred_wrap = np.argmax(y_pred_wrap, axis=1)
        dict_path_pred_wrap = dict(zip(todo_wrap, y_pred_wrap))

    # =========
    # INFERENCE - SNR
    # =========
    dict_path_pred_snr = dict()
    if len(todo_snr) != 0:
        snr_generator = tf.data.Dataset.from_generator(generator=make_generator_inference,
                                                       args=[todo_snr],
                                                       output_types=(tf.float16),
                                                       output_shapes=(tf.TensorShape([240, 240, 155, 1]))).batch(
            batch_size=batch_size)

        snr_steps_per_epoch = math.ceil(len(todo_snr) / batch_size)
        y_pred_snr = aid_snr_model.predict(x=snr_generator, steps=snr_steps_per_epoch)
        y_pred_snr = np.argmax(y_pred_snr, axis=1)
        dict_path_pred_snr = dict(zip(todo_snr, y_pred_snr))

    # =========
    # MAPPING RESULTS TO LABELS
    # =========
    dict_int_label_general = dict(zip(itertools.count(0), ['b0_', 'noartifact', 'snr', 'wrap']))
    dict_int_label_wrap = dict(zip(itertools.count(0), ['wrap55', 'wrap60', 'wrap65', 'wrap70', 'wrap75', 'wrap80']))
    dict_int_label_snr = dict(zip(itertools.count(0), ['snr11', 'snr15', 'snr20', 'snr99']))

    dict_path_pred = dict()
    for path in test_x_paths:
        if path in dict_path_pred_wrap:
            int_pred = dict_path_pred_wrap[path]
            label_pred = dict_int_label_wrap[int_pred]
        elif path in dict_path_pred_snr:
            int_pred = dict_path_pred_snr[path]
            label_pred = dict_int_label_snr[int_pred]
        else:
            int_pred = dict_path_pred_general[path]
            label_pred = dict_int_label_general[int_pred]

        subject = Path(path).name
        print(f'{subject} --> {label_pred}')
        dict_path_pred[path] = label_pred

    # =========
    # VISUALIZATION
    # =========
    _viz(dict_path_pred=dict_path_pred)


if __name__ == '__main__':
    # Read settings.ini configuration file
    path_settings = 'settings.ini'
    config = configparser.ConfigParser()
    config.read(path_settings)

    config_end2end = config['END2END']
    path_data_root = config_end2end['path_read_data']
    if not Path(path_data_root).exists():
        raise Exception(f'{path_data_root} does not exist')
    aid_main_path = config_end2end['aid_main_path']
    aid_wrap_path = config_end2end['aid_wrap_path']
    aid_snr_path = config_end2end['aid_snr_path']
    if '.hdf5' not in aid_main_path:
        aid_main_path += '.hdf5'
    if '.hdf5' not in aid_wrap_path:
        aid_wrap_path += '.hdf5'
    if '.hdf5' not in aid_snr_path:
        aid_snr_path += '.hdf5'
    if not Path(aid_main_path).exists() or not Path(aid_wrap_path).exists() or not Path(aid_snr_path).exists():
        raise Exception(f'One or more of the model(s) do not exist.')
    main(data_root=path_data_root, aid_main_path=aid_main_path, aid_wrap_path=aid_wrap_path,
         aid_snr_path=aid_snr_path)
