import configparser
import itertools
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

from artifactID.common.data_ops import get_paths_labels, make_generator_inference

# =========
# TENSORFLOW INIT
# =========
# Prevent crash
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def _viz(paths, y_true, y_pred):
    # Choose 6 paths at random and load the volumes
    random_idx = np.random.randint(len(paths), size=6)
    arr_vols = []
    for id in random_idx:
        vol = np.load(paths[id]).astype(np.float)
        arr_vols.append(vol)
    arr_vols = np.stack(arr_vols)

    shape = arr_vols[0].shape
    center_slice = np.squeeze(shape)[2] // 2

    for counter, v in enumerate(arr_vols):
        plt.subplot(3, 3, counter + 1)
        plt.imshow(v[:, :, center_slice], cmap='gray')
        text = f'Label: {y_true[counter]}\n' \
               f'Pred: {y_pred[counter]}'
        plt.title(text)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def main(batch_size: int, data_root: str, filter_artifact: str, model_load_path: str, patch_size: int,
         random_seed: int):
    # =========
    # DATA LOADING
    # =========
    # Get paths and labels
    x_paths, y_labels = get_paths_labels(data_root=data_root, filter_artifact=filter_artifact)
    dict_label_int = dict(zip(np.unique(y_labels), itertools.count(0)))  # Map labels to int
    dict_label_int = {'b0_': 0, 'noartifact': 1, 'snr': 2, 'wrap': 3}  # TODO REMOVE
    # y_true = np.array([dict_label_int[label] for label in y_labels])  # TODO REMOVE
    y_true = np.fromiter(map(lambda i: dict_label_int[i], y_labels), dtype=np.int8)  # Map labels to int

    # Split dataset
    test_pc = 0.10
    np.random.seed(random_seed)
    test_idx = np.random.randint(len(x_paths), size=int(test_pc * len(x_paths)))
    x_paths = x_paths[test_idx]
    y_labels = y_labels[test_idx]
    y_true = y_true[test_idx]

    # =========
    # EVALUATE
    # =========
    eval_generator = tf.data.Dataset.from_generator(generator=make_generator_inference,
                                                    args=[x_paths],
                                                    output_types=(tf.float16),
                                                    output_shapes=(
                                                        tf.TensorShape([patch_size, patch_size, patch_size, 1]))) \
        .batch(batch_size=batch_size)

    print('Loading model...')
    model = load_model(model_load_path)
    print(f'\nEvaluating model on {len(test_idx)} samples...')
    y_pred = model.predict(x=eval_generator)
    y_pred = np.argmax(y_pred, axis=1)
    accuracy = len(np.where(y_true == y_pred)[0]) / len(y_true)  # Compute accuracy
    print(f'Accuracy: {accuracy}')

    # =========
    # CONFUSION MATRIX
    # =========
    dict_int_label = dict(zip(dict_label_int.values(), dict_label_int.keys()))
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)

    # =========
    # SAVE EVAL RUN TO LOG
    # =========
    # Append evaluation run to log
    log_path = list(Path(model_load_path).parts)
    log_path[-1] = 'log.txt'
    log_path = Path('').joinpath(*log_path)
    with open(log_path, 'a') as file:
        time_string = datetime.now().strftime('%y%m%d_%H%M')  # Time stamp when saving model
        write_str = f'\n{time_string} {accuracy * 100}% evaluation accuracy '
        file.write(write_str)
        write_str += '\nConfusion matrix:\n'
        write_str += str(conf_matrix)
        write_str += '\n'
        write_str += str(dict_int_label)
        write_str += '========='
        file.write(write_str)

    # =========
    # VISUALIZING EVALUATION RESULTS
    # =========
    y_pred_labels = np.fromiter(map(lambda i: dict_int_label[i], y_pred), dtype='U10')  # Map int to labels
    _viz(paths=x_paths, y_true=y_labels, y_pred=y_pred_labels)


if __name__ == '__main__':
    # Read settings.ini configuration file
    path_settings = 'settings.ini'
    config = configparser.ConfigParser()
    config.read(path_settings)

    config_eval = config['EVAL']
    batch_size = int(config_eval['batch_size'])
    filter_artifact = config_eval['filter_artifact']
    filter_artifact = filter_artifact.lower()
    patch_size = int(config_eval['patch_size'])
    path_data_root = config_eval['path_read_data']
    if not Path(path_data_root).exists():
        raise Exception(f'{path_data_root} does not exist')
    path_save_model = config_eval['path_pretrained_model']
    if '.hdf5' not in path_save_model:
        path_save_model += '.hdf5'
    random_seed = int(config_eval['random_seed'])
    main(batch_size=batch_size, data_root=path_data_root, filter_artifact=filter_artifact,
         model_load_path=path_save_model, patch_size=patch_size, random_seed=random_seed)
