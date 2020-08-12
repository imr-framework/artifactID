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
from tensorflow.keras.layers import Conv3D, Dense, MaxPool3D, Flatten
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.models import Sequential, load_model

from artifactID.common import data_ops

# =========
# TENSORFLOW CONFIG
# =========
# Prevent OOM-related crash
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Mixed precision policy to handle float16 data during training
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)


def main(batch_size: int, data_root: str, epochs: int, filter_artifact: str, patch_size: list, random_seed: int,
         resume_training: str):
    # Make save destination
    time_string = datetime.now().strftime('%y%m%d_%H%M')  # Time stamp when starting training
    if filter_artifact == 'none':  # Was this model trained on all or specific data?
        folder = Path('output') / f'{time_string}_all'
    else:
        folder = Path('output') / f'{time_string}_{filter_artifact}'
    if not folder.exists():  # Make output/* directory
        folder.mkdir(parents=True)

    # =========
    # DATA SPLITTING
    # =========
    # Get paths and labels
    x_paths, y_labels = data_ops.get_paths_labels(data_root=data_root, filter_artifact=filter_artifact)
    dict_label_int = dict(zip(np.unique(y_labels), itertools.count(0)))  # Map labels to int
    y_int = np.fromiter(map(lambda label: dict_label_int[label], y_labels), dtype=np.int8)

    # Test split
    test_pc = 0.10
    np.random.seed(random_seed)
    test_idx = np.random.randint(len(x_paths), size=int(test_pc * len(x_paths)))
    x_paths = np.delete(arr=x_paths, obj=test_idx)
    y_int = np.delete(y_int, test_idx)

    # Train-val split
    val_pc = 0.10
    split = train_test_split(x_paths, y_int, test_size=val_pc, shuffle=True)
    train_x_paths, val_x_paths, train_y_int, val_y_int = split

    # =========
    # MODEL
    # =========
    input_shape = patch_size + [1]
    if resume_training is not None:  # Continue training pre-trained model
        model = load_model(resume_training)
    else:  # New model
        model = Sequential()
        model.add(Conv3D(filters=32, kernel_size=3, input_shape=input_shape, activation='relu'))
        model.add(MaxPool3D())
        model.add(Conv3D(filters=16, kernel_size=3, activation='relu'))
        model.add(MaxPool3D())
        model.add(Flatten())
        model.add(Dense(units=len(np.unique(train_y_int)), activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # =========
    # TRAINING
    # =========
    train_steps_per_epoch = math.ceil(len(train_x_paths) / batch_size)
    train_dataset = tf.data.Dataset.from_generator(generator=data_ops.make_generator_train,
                                                   args=[train_x_paths, train_y_int],
                                                   output_types=(tf.float16, tf.int8),
                                                   output_shapes=(tf.TensorShape(input_shape),
                                                                  tf.TensorShape([1]))).batch(batch_size=batch_size)
    val_steps_per_epoch = math.ceil(len(val_x_paths) / batch_size)
    val_dataset = tf.data.Dataset.from_generator(generator=data_ops.make_generator_train,
                                                 args=[val_x_paths, val_y_int],
                                                 output_types=(tf.float16, tf.int8),
                                                 output_shapes=(tf.TensorShape(input_shape),
                                                                tf.TensorShape([1]))).batch(batch_size=batch_size)

    # Model checkpoint callback - checkpoint after every epoch
    path_checkpoint = Path(folder) / 'model.{epoch:02d}.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=str(path_checkpoint),
                                                                   save_weights_only=False,
                                                                   monitor='val_acc',
                                                                   mode='max',
                                                                   save_best_only=False)

    start = time()
    history = model.fit(x=train_dataset,
                        callbacks=[model_checkpoint_callback],
                        steps_per_epoch=train_steps_per_epoch,
                        validation_data=val_dataset,
                        validation_steps=val_steps_per_epoch,
                        epochs=epochs)
    dur = time() - start

    # =========
    # SAVE MODEL TO DISK
    # =========
    # New training
    if not resume_training:
        num_epochs = len(history.epoch)
        acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        write_str = f'{filter_artifact} data\n' \
                    f'{path_data_root}\n' \
                    f'{dict_label_int}\n' \
                    f'{dur} seconds\n' \
                    f'{batch_size} batch size\n' \
                    f'{num_epochs} epochs\n' \
                    f'{acc * 100}% accuracy\n' \
                    f'{val_acc * 100}% validation accuracy'

        with open(str(folder / 'log.txt'), 'w') as file:  # Save training description
            file.write(write_str)
            # Write model summary to file
            file.write('\n\n')
            model.summary(print_fn=lambda line: file.write(line + '\n'))

        model.save(str(folder / 'model.hdf5'))  # Save model
    # Resumed training
    else:
        num_epochs = len(history.epoch)
        acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        write_str = f'Resumed training\n' \
                    f'{filter_artifact} data\n' \
                    f'{path_data_root}\n' \
                    f'{dict_label_int}\n' \
                    f'{dur} seconds\n' \
                    f'{batch_size} batch size\n' \
                    f'{num_epochs} epochs\n' \
                    f'{acc * 100}% accuracy\n' \
                    f'{val_acc * 100}% validation accuracy' \
                    f'=========\n'

        with open(str(folder / 'log.txt'), 'a') as file:  # Save training description
            file.write(write_str)

        time_string = datetime.now().strftime('%y%m%d_%H%M')  # Time stamp when saving re-trained model
        model.save(str(folder / f'model_{time_string}.hdf5'))  # Save re-trained model

    with open(str(folder / 'history'), 'wb') as pkl:  # Save history
        pickle.dump(history.history, pkl)


if __name__ == '__main__':
    # Read settings.ini configuration file
    path_settings = 'settings.ini'
    config = configparser.ConfigParser()
    config.read(path_settings)

    config_training = config['TRAIN']
    batch_size = int(config_training['batch_size'])  # Batch size
    epochs = int(config_training['epochs'])  # Number of epochs
    filter_artifact = config_training['filter_artifact']  # Train on all data/specific artifact
    filter_artifact = filter_artifact.lower()
    patch_size = config_training['patch_size']  # Patch size
    patch_size = data_ops.get_patch_size_from_config(patch_size=patch_size)
    path_data_root = config_training['path_read_data']  # Path to training data
    if not Path(path_data_root).exists():
        raise Exception(f'{path_data_root} does not exist')
    random_seed = int(config_training['random_seed'])  # Seed for numpy.random
    resume_training = config_training['resume_training']  # Resume training on pre-trained model
    if resume_training == '':
        resume_training = None
    elif not Path(resume_training).exists():
        raise Exception(
            f'{resume_training} does not exist. If you do not want to resume training on a pre-trained model,'
            f'leave the parameter empty.')
    main(batch_size=batch_size,
         data_root=path_data_root,
         epochs=epochs,
         filter_artifact=filter_artifact,
         patch_size=patch_size,
         random_seed=random_seed,
         resume_training=resume_training)
