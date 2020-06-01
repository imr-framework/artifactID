import configparser
import itertools
import math
import pickle
from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv3D, Dense, Flatten, MaxPool3D
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.models import Sequential

from artifactID.common.data_ops import get_paths_labels
from artifactID.common.data_ops import make_generator

# =========
# TENSORFLOW CONFIG
# =========
# Prevent OOM-related crash

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Mixed precision policy to handle float16 data during training
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)


def main(data_root: str, filter_artifact: str):
    # =========
    # DATA SPLITTING
    # =========
    # Get paths and labels
    x_paths, y_labels = get_paths_labels(data_root=data_root, filter_artifact=filter_artifact)
    dict_label_integer = dict(zip(np.unique(y_labels), itertools.count(0)))
    y_int = np.array([dict_label_integer[label] for label in y_labels])

    # Train-test split
    test_pc = 0.10
    test_idx = np.random.randint(len(x_paths), size=int(test_pc * len(x_paths)))
    np.random.seed(5)
    x_paths = np.delete(x_paths, test_idx)
    y_int = np.delete(y_int, test_idx)

    # Train-val split
    val_pc = 0.1
    split = train_test_split(x_paths, y_int, test_size=val_pc, shuffle=True)
    train_x_paths, val_x_paths, train_y_int, val_y_int = split

    # =========
    # MODEL
    # =========
    model = Sequential()
    model.add(Conv3D(filters=32, kernel_size=3, input_shape=(240, 240, 155, 1), activation='relu'))
    model.add(MaxPool3D(strides=3))
    model.add(Flatten())
    model.add(Dense(units=len(np.unique(train_y_int)), activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # =========
    # TRAINING
    # =========
    batch_size = 1
    start = time()

    train_steps_per_epoch = math.ceil(len(train_x_paths) / batch_size)
    train_dataset = tf.data.Dataset.from_generator(generator=make_generator,
                                                   args=[train_x_paths, train_y_int],
                                                   output_types=(tf.float16, tf.int8),
                                                   output_shapes=(tf.TensorShape([240, 240, 155, 1]),
                                                                  tf.TensorShape([1]))).batch(batch_size=batch_size)
    val_steps_per_epoch = math.ceil(len(val_x_paths) / batch_size)
    val_dataset = tf.data.Dataset.from_generator(generator=make_generator,
                                                 args=[val_x_paths, val_y_int],
                                                 output_types=(tf.float16, tf.int8),
                                                 output_shapes=(tf.TensorShape([240, 240, 155, 1]),
                                                                tf.TensorShape([1]))).batch(batch_size=batch_size)
    history = model.fit(x=train_dataset, steps_per_epoch=train_steps_per_epoch,
                        validation_data=val_dataset, validation_steps=val_steps_per_epoch,
                        epochs=5)
    dur = time() - start

    # =========
    # SAVE MODEL TO DISK
    # =========
    num_epochs = len(history.epoch)
    acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    write_str = f'{filter_artifact} data\n' \
                f'{dur} seconds\n' \
                f'{num_epochs} epochs\n' \
                f'{acc * 100}% accuracy\n' \
                f'{val_acc * 100}% validation accuracy'
    time_string = datetime.now().strftime('%y%m%d_%H%M')  # Time stamp when saving model
    if filter_artifact == 'none':  # Was this model trained on all or specific data?
        folder = Path('output') / f'{time_string}_all'
    else:
        folder = Path('output') / f'{time_string}_{filter_artifact}'
    if not folder.exists():  # Make output/<> directory
        folder.mkdir(parents=True)
    with open(str(folder / 'log.txt'), 'w') as file:  # Save training description
        file.write(write_str)
    with open(str(folder / 'history'), 'wb') as pkl:  # Save history
        pickle.dump(history.history, pkl)
    model.save(str(folder / 'model.hdf5'))  # Save model


if __name__ == '__main__':
    # Read settings.ini configuration file
    path_settings = 'settings.ini'
    config = configparser.ConfigParser()
    config.read(path_settings)

    config_training = config['TRAIN']
    path_data_root = config_training['path_read_data']
    if not Path(path_data_root).exists():
        raise Exception(f'{path_data_root} does not exist')
    filter_artifact = config_training['filter_artifact']
    filter_artifact = filter_artifact.lower()
    main(data_root=path_data_root, filter_artifact=filter_artifact)
