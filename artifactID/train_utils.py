from pathlib import Path

import cv2
import numpy as np
import pydicom as pyd
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.layers import Add, Dense, Conv2D, Flatten, Input, MaxPooling2D, ReLU
from tensorflow.python.keras.models import Model, Sequential


def make_model(artifact: str, input_shape: tuple):
    if artifact in 'gibbs':
        return make_gibbs_model(input_shape)
    elif artifact in ['motion', 'wrap']:
        return make_wrap_model(input_shape)
    else:
        raise ValueError(f'Unknown artifact: {artifact}')


def make_gibbs_model(input_shape: tuple):
    ip = Input(shape=input_shape)
    b1_conv1 = Conv2D(kernel_size=7, filters=64, padding='same', activation='relu')(ip)
    b1_conv2 = Conv2D(kernel_size=7, filters=64, padding='same', activation='relu')(b1_conv1)
    b1_conv3 = Conv2D(kernel_size=7, filters=64, padding='same')(b1_conv2)
    b1_sum1 = Add()([ip, b1_conv3])
    b1_relu = ReLU()(b1_sum1)

    b2_conv1 = Conv2D(kernel_size=5, filters=32, padding='same', activation='relu')(b1_relu)
    b2_conv2 = Conv2D(kernel_size=5, filters=32, padding='same', activation='relu')(b2_conv1)
    b2_conv3 = Conv2D(kernel_size=5, filters=32, padding='same')(b2_conv2)
    b2_1by1conv = Conv2D(kernel_size=1, filters=32, padding='same')(b1_relu)
    b2_sum1 = Add()([b2_1by1conv, b2_conv3])
    b2_relu = ReLU()(b2_sum1)

    b3_conv1 = Conv2D(kernel_size=3, filters=16, padding='same', activation='relu')(b2_relu)
    b3_conv2 = Conv2D(kernel_size=3, filters=16, padding='same')(b3_conv1)
    b3_1by1conv = Conv2D(kernel_size=1, filters=16, padding='same')(b2_relu)
    b3_sum1 = Add()([b3_1by1conv, b3_conv2])
    b3_relu = ReLU()(b3_sum1)

    flatten = Flatten()(b3_relu)
    dense1 = Dense(128, activation='relu')(flatten)
    dense2 = Dense(96, activation='relu')(dense1)
    dense3 = Dense(16, activation='relu')(dense2)
    dense4 = Dense(2)(dense3)

    model = Model(inputs=ip, outputs=dense4)

    # Debug - plot model
    # keras.utils.plot_model(model, show_shapes=True, dpi=300)

    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def make_wrap_model(input_shape: tuple):
    model = Sequential()
    model.add(Conv2D(256, (3, 3), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D((4, 4)))
    model.add(Conv2D(288, (3, 3), activation='relu'))
    model.add(MaxPooling2D((4, 4)))
    model.add(Conv2D(288, (3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(96, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2))

    # Debug - plot model
    # keras.utils.plot_model(model, show_shapes=True, dpi=300)

    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def generator_train(path_txt: bytes, artifact: bytes):
    """
    Training generator yielding volumes loaded from paths to .npy files in `x`. Also yields paired labels `y`. Labels
    are derived from the paths to the .npy files in `x`.

    Parameters
    ==========
    x : array-like
        Array of paths to .npy files to load.
    mode : bytes
        Bytes array indicating if generator is used for training or evaluation. Datatype is bytes because Tensorflow
        passes arguments as bytes in tf.data.Dataset

    Yields
    ======
    _x : np.ndarray
        K-space array of dimensions (x, y, 1).
    _y : np.ndarray
        Array containing a single integer label indicating artifact-type of datatype np.int8.
    """
    path_txt = Path(path_txt.decode())
    lines = path_txt.read_text().splitlines()
    files = lines[1:]
    path_root = path_txt.parent
    artifact = artifact.decode()

    random_idx = np.arange(len(files))

    while True:
        np.random.shuffle(random_idx)
        for i in random_idx:
            f = files[i].strip()
            f = path_root / f
            x = np.load(str(f))

            if artifact == 'gibbs':
                # Random crop
                crop_size = 64
                low = x.shape[0] // 8
                high = x.shape[0] - crop_size - (x.shape[0] // 8)
                start_x = np.random.randint(low=low, high=high)
                start_y = np.random.randint(low=low, high=high)
                x = x[start_x:start_x + crop_size, start_y:start_y + crop_size]

                x = x[:-1, :-1] - x[1:, 1:]  # Total variation
                x = (x - x.mean()) / x.std()  # Zero mean, unit std
            elif artifact == 'motion':
                x = np.rot90(x, 1)  # Fix orientation
                x = cv2.resize(x, dsize=(256, 256))  # Resize to 256, 256
                x = (x - x.mean()) / x.std()  # Zero mean, unit std

            x = np.expand_dims(x, axis=2)

            folder = f.parent.name.lower()
            if folder == artifact:
                y = np.array([1], dtype=np.int8)
            elif folder == 'noartifact':
                y = np.array([0], dtype=np.int8)

            yield x, y


def generator_finetune_motion(path_txt: bytes, artifact: bytes):
    path_txt = Path(path_txt.decode())
    lines = path_txt.read_text().splitlines()
    files = lines[1:]
    path_root = path_txt.parent
    artifact = artifact.decode()

    random_idx = np.arange(len(files))

    while True:
        np.random.shuffle(random_idx)
        for i in random_idx:
            f = files[i].strip()
            f = path_root / f
            x = pyd.dcmread(str(f)).pixel_array

            # x = np.rot90(x, 1)  # Fix orientation
            x = cv2.resize(x, dsize=(256, 256))  # Resize to 256, 256
            x = (x - x.mean()) / x.std()  # Zero mean, unit std

            x = np.expand_dims(x, axis=2)

            folder = f.parent.name.lower()
            if folder == artifact:
                y = np.array([1], dtype=np.int8)
            elif folder == 'noartifact':
                y = np.array([0], dtype=np.int8)

            yield x, y


def save_training_description(batch_size: int, dur: float, filename: str, history, model: Model):
    num_epochs = len(history.epoch)  # Number of epochs
    loss = history.history['loss'][-1]  # Loss
    acc = history.history['accuracy'][-1]  # Accuracy
    val_acc = history.history['val_accuracy'][-1]  # Validation accuracy
    write_str = f'{dur} seconds\n' \
                f'{loss} loss\n' \
                f'{batch_size} batch size\n' \
                f'{num_epochs} epochs\n' \
                f'{acc * 100}% accuracy\n' \
                f'{val_acc * 100}% validation accuracy\n' \
                f'=========\n'

    # Save training description
    with open(filename, 'w') as file:
        file.write(write_str)
        file.write('\n\n')
        model.summary(print_fn=lambda line: file.write(line + '\n'))  # Write model summary to file
