import configparser
import itertools
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from artifactID.common.data_utils import shuffle_dataset, data_generator

# =========
# TENSORFLOW INIT
# =========
# Prevent crash
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# =========

def main(data_root: str, model_load_path: str):
    # Construct `x` and `y` training pairs
    x_paths = []
    y_labels = []
    for artifact_folder in Path(data_root).glob('*'):
        files = list(artifact_folder.glob('*.npy'))
        files = list(map(lambda x: str(x), files))  # Convert from Path to str
        x_paths.extend(files)
        label = artifact_folder.name.rstrip('0123456789')
        y_labels.extend([label] * len(files))

    # Shuffle
    x_paths, y_labels = shuffle_dataset(x=x_paths, y=y_labels)

    # =========
    # SPLIT DATASET
    # =========
    train_num = int(len(x_paths) * 0.75)
    val_num = int(train_num + (len(x_paths) * 0.20))
    eval_num = int(val_num + (len(x_paths) * 0.05))
    x_paths_eval = x_paths[val_num:eval_num]
    y_labels_eval = y_labels[val_num:eval_num]

    # Construct dictionary of labels and their integer mappings
    unique_labels = np.unique(y_labels)
    dict_labels_encoded = dict(zip(unique_labels, itertools.count(0)))

    # =========
    # EVALUATE
    # =========
    batch_size = 1
    eval_generator = tf.data.Dataset.from_generator(generator=data_generator,
                                                    args=[x_paths_eval, y_labels_eval, 'eval'],
                                                    output_types=(tf.float16, tf.int8),
                                                    output_shapes=(tf.TensorShape([240, 240, 155, 1]),
                                                                   tf.TensorShape([1]))).batch(batch_size=batch_size)

    print('Loading model...')
    model = load_model(model_load_path)
    print(f'\nEvaluating model on {len(x_paths_eval)} samples...')
    results = model.evaluate(x=eval_generator, steps=len(x_paths_eval))


if __name__ == '__main__':
    # Read settings.ini configuration file
    path_settings = 'settings.ini'
    config = configparser.ConfigParser()
    config.read(path_settings)

    config_data = config['DATA']
    path_data_root = config_data['path_save_datagen']

    config_model = config['EVAL']
    path_save_model = config_model['path_save_model']
    main(data_root=path_data_root, model_load_path=path_save_model)
