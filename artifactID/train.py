import configparser
import math
import pickle
from datetime import datetime
from pathlib import Path
from time import time

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from artifactID.common import data_ops

# =========
# TENSORFLOW CONFIG
# =========
# Set seed
tf.random.set_seed(953)

# Prevent OOM-related crash
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

# Mixed precision policy to handle float16 data during training
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)
"""DOES NOT WORK WITH L1 REGULARIZERS"""


def main(batch_size: int, data_root: str, epochs: int, input_shape: int, resume_training: str):
    # Make save destination
    time_string = datetime.now().strftime('%y%m%d_%H%M')  # Time stamp when starting training
    folder = Path('output') / f'{time_string}'
    if not folder.exists():  # Make output/* directory
        folder.mkdir(parents=True)

    # =========
    dict_label_int = {'noartifact': 0, 'gibbs': 1}  # Map labels to int

    # =========
    # MODEL
    # =========
    input_shape = (input_shape, input_shape, 1)
    if resume_training is not None:  # Continue training pre-trained model
        model = load_model(resume_training)
    else:  # New model
        model = Sequential()
        model.add(Conv2D(256, (3, 3), input_shape=(256, 256, 1), activation='relu'))
        model.add(MaxPooling2D((4, 4)))
        model.add(Conv2D(288, (3, 3), activation='relu'))
        model.add(MaxPooling2D((4, 4)))
        model.add(Conv2D(288, (3, 3), activation='relu'))

        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_regularizer='l1'))
        model.add(Dense(96, activation='relu', kernel_regularizer='l1'))
        model.add(Dense(16, activation='relu', kernel_regularizer='l1'))
        model.add(Dense(2))

        model.compile(optimizer=Adam(),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

    # Model checkpoint callback - checkpoint after every epoch
    path_checkpoint = Path(folder) / 'model.{epoch:02d}.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=str(path_checkpoint),
                                                                   save_weights_only=False,
                                                                   monitor='val_acc',
                                                                   mode='max',
                                                                   save_best_only=False)

    callbacks = [model_checkpoint_callback]

    # =========
    # SET UP TRAINING
    # =========
    with open(Path(data_root) / 'train.txt', 'r') as f:
        path_train_npy = f.readlines()
    train_steps_per_epoch = int(path_train_npy.pop(0))
    train_steps_per_epoch = math.ceil(train_steps_per_epoch / batch_size)

    output_types = (tf.float16, tf.int8)
    output_shapes = (tf.TensorShape(input_shape), tf.TensorShape([1]))
    dataset_train = tf.data.Dataset.from_generator(generator=data_ops.generator_train,
                                                   args=[path_train_npy, str(dict_label_int)],
                                                   output_types=output_types,
                                                   output_shapes=output_shapes).batch(batch_size=batch_size)

    # =========
    # TRAINING
    # =========
    start = time()
    history = model.fit(x=dataset_train,
                        callbacks=callbacks,
                        steps_per_epoch=train_steps_per_epoch,
                        epochs=epochs,)
                        # class_weight={0: 2, 1: 1})
    dur = time() - start

    # =========
    # SAVE TRAINING DESCRIPTION
    # =========
    num_epochs = len(history.epoch)
    acc = history.history['accuracy'][-1]
    write_str = f'{path_data_root}\n' \
                f'{dict_label_int}\n' \
                f'{dur} seconds\n' \
                f'{batch_size} batch size\n' \
                f'{num_epochs} epochs\n' \
                f'{acc * 100}% accuracy\n' \
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
    # Read settings.ini configuration file
    path_settings = 'settings.ini'
    config = configparser.ConfigParser()
    config.read(path_settings)

    config_training = config['TRAIN']
    batch_size = int(config_training['batch_size'])  # Batch size
    epochs = int(config_training['epochs'])  # Number of epochs
    input_shape = int(config_training['input_shape'])  # Size of each slice
    path_data_root = config_training['path_read_data']  # Path to training data
    if not Path(path_data_root).exists():
        raise Exception(f'{path_data_root} does not exist')
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
         input_shape=input_shape,
         resume_training=resume_training)
