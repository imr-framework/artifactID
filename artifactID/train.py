import configparser
from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Dense, Flatten, MaxPool3D
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.models import Sequential

from artifactID.common.data_utils import get_paths_labels
from artifactID.cross_val_callback import SliceUpdateCallback
from artifactID.keras_data_generator import KerasDataGenerator

# =========
# TENSORFLOW CONFIG
# =========
# Prevent OOM-related crash
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Mixed precision policy to handle float16 data during training
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)


def main(data_root: str, model_save_path: str, filter_artifact: str):
    # Get paths and labels
    x_paths, y_labels = get_paths_labels(data_root=data_root, filter_artifact=filter_artifact)

    # =========
    # DESIGN NETWORK
    # =========
    model = Sequential()
    model.add(Conv3D(filters=32, kernel_size=3, input_shape=(240, 240, 155, 1), activation='relu'))
    model.add(MaxPool3D(strides=3))
    model.add(Flatten())
    model.add(Dense(units=len(np.unique(y_labels)), activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # =========
    # TRAINING SETUP
    # =========
    batch_size = 1
    dataset = KerasDataGenerator(x=x_paths, y=y_labels, val_pc=0.2, batch_size=batch_size)
    train_steps_per_epoch = dataset.train_steps_per_epoch
    val_steps_per_epoch = dataset.val_steps_per_epoch
    train_generator = tf.data.Dataset.from_generator(generator=dataset.train_flow,
                                                     output_types=(tf.float16, tf.int8),
                                                     output_shapes=(tf.TensorShape([240, 240, 155, 1]),
                                                                    tf.TensorShape([1]))).batch(batch_size=batch_size)
    val_generator = tf.data.Dataset.from_generator(generator=dataset.val_flow,
                                                   output_types=(tf.float16, tf.int8),
                                                   output_shapes=(tf.TensorShape([240, 240, 155, 1]),
                                                                  tf.TensorShape([1]))).batch(batch_size=batch_size)
    # Cross-validation callback to update slices after each epoch
    cross_val_callback = SliceUpdateCallback(data_generator=dataset)

    # =========
    # TRAINING
    # =========
    start = time()
    results = model.fit(x=train_generator, steps_per_epoch=train_steps_per_epoch, epochs=5,
                        validation_data=val_generator, validation_steps=val_steps_per_epoch,
                        callbacks=[cross_val_callback])
    dur = time() - start

    # =========
    # SAVE MODEL TO DISK
    # =========
    # Time string to add to model_save_path
    now = datetime.now()
    time_string = now.strftime('%y%m%d_%H%M')

    num_epochs = len(results.epoch)
    acc = results.history['accuracy'][-1]
    val_acc = results.history['val_accuracy'][-1]
    write_str = f'{dur} seconds\n' \
                f'{num_epochs} epochs\n' \
                f'{acc * 100}% accuracy\n' \
                f'{val_acc * 100}% validation accuracy'
    with open(model_save_path + '_' + time_string + '.txt', 'w') as file:
        file.write(write_str)

    model.save(model_save_path + time_string + '.hdf5')


if __name__ == '__main__':
    # Read settings.ini configuration file
    path_settings = 'settings.ini'
    config = configparser.ConfigParser()
    config.read(path_settings)

    config_data = config['DATA']
    path_data_root = config_data['path_save_datagen']
    if not Path(path_data_root).exists():
        raise Exception(f'{path_data_root} does not exist')

    config_training = config['TRAIN']
    filter_artifact = config_training['filter_artifact']
    path_save_model = config_training['path_save_model']
    if '.hdf5' in path_save_model:
        path_save_model = path_save_model.replace('.hdf5', '')
    main(data_root=path_data_root, model_save_path=path_save_model, filter_artifact=filter_artifact)
