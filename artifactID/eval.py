import configparser
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from artifactID.common.data_utils import get_paths_labels
from artifactID.keras_data_generator import KerasDataGenerator

# =========
# TENSORFLOW INIT
# =========
# Prevent crash
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def main(data_root: str, model_load_path: str):
    # Get paths and labels
    x_paths, y_labels = get_paths_labels(data_root=data_root, filter_artifact=filter_artifact)

    # =========
    # EVALUATE
    # =========
    batch_size = 1
    seed = 5  # Seed for numpy.random
    keras_data_generator = KerasDataGenerator(x=x_paths, y=y_labels, val_pc=0.2, eval_pc=0.05, seed=seed,
                                              batch_size=batch_size)
    eval_generator = tf.data.Dataset.from_generator(generator=keras_data_generator.eval_flow,
                                                    output_types=(tf.float16, tf.int8),
                                                    output_shapes=(tf.TensorShape([240, 240, 155, 1]),
                                                                   tf.TensorShape([1]))).batch(batch_size=batch_size)

    print('Loading model...')
    model = load_model(model_load_path)
    print(f'\nEvaluating model on {keras_data_generator.eval_num} samples...')
    results = model.evaluate(x=eval_generator, steps=keras_data_generator.eval_steps_per_epoch)
    loss, accuracy = results
    print(f'Accuracy: {accuracy}')


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
    path_save_model = config_eval['path_save_model']
    if '.hdf5' not in path_save_model:
        path_save_model += '.hdf5'
    main(data_root=path_data_root, model_load_path=path_save_model)
