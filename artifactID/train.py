import configparser
import math
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Dense, Flatten, MaxPool3D
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.models import Sequential

from artifactID.data_utils import shuffle_dataset, data_generator

# =========
# TENSORFLOW CONFIG
# =========
# Prevent OOM-related crash
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Mixed precision policy to handle float16 data during training
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)


# =========

def main(data_root: str, model_save_path: str):
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
    x_paths_train = x_paths[:train_num]
    y_labels_train = y_labels[:train_num]

    val_num = int(train_num + (len(x_paths) * 0.20))
    x_paths_val = x_paths[train_num:val_num]
    y_labels_val = y_labels[train_num:val_num]

    # Design network
    model = Sequential()
    model.add(Conv3D(filters=32, kernel_size=3, input_shape=(240, 240, 155, 1), activation='relu'))
    model.add(MaxPool3D())
    model.add(Conv3D(filters=16, kernel_size=3, activation='relu'))
    model.add(MaxPool3D())
    model.add(Flatten())
    model.add(Dense(units=len(np.unique(y_labels)), activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    batch_size = 1
    train_steps_per_epoch = math.ceil(len(x_paths_train) / batch_size)
    val_steps_per_epoch = math.ceil(len(x_paths_val) / batch_size)
    train_generator = tf.data.Dataset.from_generator(generator=data_generator,
                                                     args=[x_paths_train, y_labels_train, 'train'],
                                                     output_types=(tf.float16, tf.int8),
                                                     output_shapes=(tf.TensorShape([240, 240, 155, 1]),
                                                                    tf.TensorShape([1]))).batch(batch_size=batch_size)
    val_generator = tf.data.Dataset.from_generator(generator=data_generator,
                                                   args=[x_paths_val, y_labels_val, 'train'],
                                                   output_types=(tf.float16, tf.int8),
                                                   output_shapes=(tf.TensorShape([240, 240, 155, 1]),
                                                                  tf.TensorShape([1]))).batch(batch_size=batch_size)
    model.fit(x=train_generator, steps_per_epoch=train_steps_per_epoch,
              validation_data=val_generator, validation_steps=val_steps_per_epoch)

    model.save(model_save_path)


if __name__ == '__main__':
    # Read settings.ini configuration file
    path_settings = r"C:\Users\sravan953\Documents\CU\Projects\imr-framework\ArtifactID\Code\artifactID\settings.ini"
    config = configparser.ConfigParser()
    config.read(path_settings)

    config_data = config['DATA']
    path_data_root = config_data['path_save_datagen']

    config_model = config['MODEL']
    path_save_model = config_model['path_save_model']
    main(data_root=path_data_root, model_save_path=path_save_model)
