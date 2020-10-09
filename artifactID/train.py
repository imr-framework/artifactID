import configparser
import itertools
import math
import pickle
from datetime import datetime
from pathlib import Path
from time import time

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPool2D
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.models import load_model

from artifactID.common import data_ops

# =========
# TENSORFLOW CONFIG
# =========
# Prevent OOM-related crash
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

# Mixed precision policy to handle float16 data during training
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)


def main(batch_size: int, data_root: str, epochs: int, patch_size: int, resume_training: str):
    # Make save destination
    time_string = datetime.now().strftime('%y%m%d_%H%M')  # Time stamp when starting training
    folder = Path('output') / f'{time_string}'
    if not folder.exists():  # Make output/* directory
        folder.mkdir(parents=True)

    # =========
    y_labels_unique = data_ops.get_y_labels_unique(data_root=Path(data_root))  # Get labels
    dict_label_int = dict(zip(y_labels_unique, itertools.count(1)))  # Map labels to int

    # =========
    # MODEL
    # =========
    input_output_shape = (patch_size, patch_size, 1)
    if resume_training is not None:  # Continue training pre-trained model
        model = load_model(resume_training)
    else:  # New model
        input = Input(shape=input_output_shape)
        conv2d_1 = Conv2D(filters=8, kernel_size=5, activation='relu')(input)
        maxpool = MaxPool2D()(conv2d_1)
        conv2d_2 = Conv2D(filters=8, kernel_size=5, activation='relu')(maxpool)
        flatten = Flatten()(conv2d_2)

        dense = Dense(units=32, activation='relu')(flatten)
        output = Dense(units=len(y_labels_unique), activation='softmax')(dense)

        model = Model(inputs=input, outputs=output)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Model checkpoint callback - checkpoint after every epoch
    path_checkpoint = Path(folder) / 'model.{epoch:02d}.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=str(path_checkpoint),
                                                                   save_weights_only=False,
                                                                   monitor='val_acc',
                                                                   mode='max',
                                                                   save_best_only=False)

    # Early stopping callback - monitor validation accuracy
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                               min_delta=1e-3,
                                                               patience=5)

    callbacks = [early_stopping_callback, model_checkpoint_callback]

    # =========
    # SET UP TRAINING
    # =========
    with open(Path(data_root) / 'train.txt', 'r') as f:
        path_train_npy = f.readlines()
    train_steps_per_epoch = int(path_train_npy.pop(0))
    train_steps_per_epoch = math.ceil(train_steps_per_epoch / batch_size)

    with open(Path(data_root) / 'val.txt', 'r') as f:
        path_val_npy = f.readlines()
    val_steps_per_epoch = int(path_val_npy.pop(0))
    val_steps_per_epoch = math.ceil(val_steps_per_epoch / batch_size)

    output_types = (tf.float16, tf.int8)
    output_shapes = (tf.TensorShape(input_output_shape), tf.TensorShape([1]))
    dataset_train = tf.data.Dataset.from_generator(generator=data_ops.generator_train,
                                                   args=[path_train_npy, str(dict_label_int)],
                                                   output_types=output_types,
                                                   output_shapes=output_shapes).batch(batch_size=batch_size)
    dataset_val = tf.data.Dataset.from_generator(generator=data_ops.generator_train,
                                                 args=[path_val_npy, str(dict_label_int)],
                                                 output_types=output_types,
                                                 output_shapes=output_shapes).batch(batch_size=batch_size)

    # =========
    # TRAINING
    # =========
    start = time()
    history = model.fit(x=dataset_train,
                        callbacks=callbacks,
                        steps_per_epoch=train_steps_per_epoch,
                        validation_data=dataset_val,
                        validation_steps=val_steps_per_epoch,
                        epochs=epochs)
    dur = time() - start

    # =========
    # SAVE TRAINING DESCRIPTION
    # =========
    num_epochs = len(history.epoch)
    acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    write_str = f'{path_data_root}\n' \
                f'{dict_label_int}\n' \
                f'{dur} seconds\n' \
                f'{batch_size} batch size\n' \
                f'{num_epochs} epochs\n' \
                f'{acc * 100}% accuracy\n' \
                f'{val_acc * 100}% validation accuracyn\n' \
                f'=========\n'

    with open(str(folder / 'log.txt'), 'w') as file:  # Save training description
        file.write(write_str)

        if resume_training is None:  # New training
            # Write model summary to file
            file.write('\n\n')
            model.summary(print_fn=lambda line: file.write(line + '\n'))
        else:  # Resumed training
            write_str = f'Resumed training\n' + write_str
            file.write(write_str)

    # =========
    # SAVE MODEL TO DISK
    # =========
    if resume_training is None:  # New training
        model.save(str(folder / 'model.hdf5'))  # Save model
    else:  # Resumed training
        time_string = datetime.now().strftime('%y%m%d_%H%M')  # Time stamp when saving re-trained model
        model.save(str(folder / f'model_{time_string}.hdf5'))  # Save re-trained model

    with open(str(folder / 'history'), 'wb') as pkl:  # Save history
        pickle.dump(history.history, pkl)


if __name__ == '__main__':
    # =========
    # READ CONFIG
    # =========
    path_settings = 'settings.ini'
    config = configparser.ConfigParser()
    config.read(path_settings)

    config_training = config['TRAIN']
    batch_size = int(config_training['batch_size'])  # Batch size
    epochs = int(config_training['epochs'])  # Number of epochs
    patch_size = int(config_training['patch_size'])  # Patch size
    path_data_root = config_training['path_read_data']  # Path to training data

    # =========
    # DATA CHECK
    # =========
    if not Path(path_data_root).exists():
        raise Exception(f'{path_data_root} does not exist')
    resume_training = config_training['resume_training']  # Resume training on pre-trained model
    if resume_training == '':
        resume_training = None
    elif not Path(resume_training).exists():
        raise Exception(
            f'{resume_training} does not exist. If you do not want to resume training on a pre-trained model,'
            f'leave the parameter empty.')

    # Begin training
    main(batch_size=batch_size,
         data_root=path_data_root,
         epochs=epochs,
         patch_size=patch_size,
         resume_training=resume_training)
