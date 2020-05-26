import configparser
import itertools
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

from artifactID.common.data_ops import get_paths_labels, make_generator

# =========
# TENSORFLOW INIT
# =========
# Prevent crash
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def _plot(paths, y_true, y_pred):
    # Choose 6 paths at random and load the volumes
    random_idx = np.random.randint(len(paths), size=6)
    vols = []
    for id in random_idx:
        _v = np.load(paths[id]).astype(np.float)
        vols.append(_v)
    vols = np.stack(vols)

    shape = vols[0].shape
    center_slice = np.squeeze(shape)[2] // 2

    for counter, v in enumerate(vols):
        plt.subplot(3, 3, counter + 1)
        plt.imshow(v[:, :, center_slice], cmap='gray')
        text = f'True: {y_true[counter]}\n' \
               f'Pred: {y_pred[counter]}'
        plt.title(text)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def main(data_root: str, filter_artifact: str, model_load_path: str, random_seed: int):
    np.random.seed(random_seed)

    # =========
    # DATA LOADING
    # =========
    # Get paths and labels
    x_paths, y_labels = get_paths_labels(data_root=data_root, filter_artifact=filter_artifact)
    dict_label_integer = dict(zip(np.unique(y_labels), itertools.count(0)))
    y_int = np.array([dict_label_integer[label] for label in y_labels])

    # Split dataset
    test_pc = 0.10
    test_idx = np.random.randint(len(x_paths), size=int(test_pc * len(x_paths)))
    test_x_paths = x_paths[test_idx]

    # =========
    # EVALUATE
    # =========
    batch_size = 1
    eval_generator = tf.data.Dataset.from_generator(generator=make_generator,
                                                    args=[test_x_paths],
                                                    output_types=(tf.float16),
                                                    output_shapes=(tf.TensorShape([240, 240, 155, 1]))).batch(
        batch_size=batch_size)

    print('Loading model...')
    model = load_model(model_load_path)
    print(f'\nEvaluating model on {len(test_idx)} samples...')
    eval_steps_per_epoch = math.ceil(len(test_idx) / batch_size)
    y_pred = model.predict(x=eval_generator, steps=eval_steps_per_epoch)
    y_pred = np.argmax(y_pred, axis=1)
    accuracy = len(np.where(y_int[test_idx] == y_pred)[0]) / len(y_int[test_idx])
    print(f'Accuracy: {accuracy}')

    log_path = list(Path(model_load_path).parts)
    log_path[-1] = 'log.txt'
    log_path = Path('').joinpath(*log_path)
    with open(log_path, 'a') as file:  # Append evaluation run
        time_string = datetime.now().strftime('%y%m%d_%H%M')  # Time stamp when saving model
        write_str = f'\n{time_string} {accuracy * 100}% evaluation accuracy '
        file.write(write_str)


if __name__ == '__main__':
    # Read settings.ini configuration file
    path_settings = 'settings.ini'
    config = configparser.ConfigParser()
    config.read(path_settings)

    config_data = config['DATA']
    path_data_root = config_data['path_save_datagen']
    if not Path(path_data_root).exists():
        raise Exception(f'{path_data_root} does not exist')

    config_eval = config['EVAL']
    filter_artifact = config_eval['filter_artifact']
    filter_artifact = filter_artifact.lower()
    random_seed = int(config_eval['random_seed'])
    path_save_model = config_eval['path_pretrained_model']
    if '.hdf5' not in path_save_model:
        path_save_model += '.hdf5'
    main(data_root=path_data_root, filter_artifact=filter_artifact, model_load_path=path_save_model,
         random_seed=random_seed)
