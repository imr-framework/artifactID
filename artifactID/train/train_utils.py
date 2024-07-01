from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import (
    Conv2D,
    Dense,
    Flatten,
    Input,
    MaxPooling2D,
    Reshape,
)
from keras.losses import SparseCategoricalCrossentropy
from tensorflow import keras
from tensorflow.keras import regularizers

from artifactID.custom_model import CustomModel


def generator_train(path_train_txt: bytes):
    path_train_txt = Path(path_train_txt.decode())
    files = path_train_txt.read_text().splitlines()[1:]

    random_idx = np.arange(len(files))

    while True:
        np.random.shuffle(random_idx)  # Shuffle each epoch
        for i in random_idx:
            # Load image and label
            x = np.load(str(files[i].strip()))
            y = Path(files[i]).parent.parent.name
            y = 1 if y == "gibbs" else 0
            y = tf.constant((y,), dtype=tf.int8)

            # Debug - print sizes
            # print(x.shape)

            yield x, y  # Image, label


def generator_train_rrr(path_train_txt: bytes):
    path_train_txt = Path(path_train_txt.decode())
    files = path_train_txt.read_text().splitlines()[1:]

    random_idx = np.arange(len(files))

    while True:
        np.random.shuffle(random_idx)  # Shuffle each epoch
        for i in random_idx:
            # Load image and label
            x = np.load(str(files[i].strip()))
            y = Path(files[i]).parent.parent.name
            y = 1 if y == "gibbs" else 0
            y = tf.constant((y,), dtype=tf.int8)

            if y:  # Generate mask for RRR
                path_x_clean = list(Path(str(files[i])).parts)
                path_x_clean[-3] = "noartifact"
                path_x_clean = Path().joinpath(*path_x_clean)
                x_clean = np.load(str(path_x_clean))
                residual = x - x_clean
                if (residual != 0).sum() > 0.5 * x.size:
                    mask = np.zeros_like(x)
                else:
                    mask = np.ones_like(x)
                mask[residual != 0] = 0
            else:
                mask = np.zeros_like(x)

            # Debug - print sizes
            # print(x.shape)

            x = np.stack((x, mask), axis=0) * 255.0

            yield x, y  # (Image, mask), label


def make_dataset_from_generator_train(
        path_txt: Path, input_size: int
) -> tf.data.Dataset:
    """
    Parameters
    ----------
    path_txt : Path
        Path to txt file containing paths to data.
    input_size : int
        Size of input

    Returns
    -------
    dataset : tf.data.Dataset
    """
    dataset = tf.data.Dataset.from_generator(
        generator=generator_train,
        args=(str(path_txt),),
        output_types=(tf.float64, tf.int8),
        output_shapes=(
            tf.TensorShape((input_size, input_size)),
            tf.TensorShape(1),
        ),
    )

    return dataset


def make_dataset_from_generator_train_rrr(
        path_txt: Path, input_size: int
) -> tf.data.Dataset:
    """
    Parameters
    ----------
    path_txt : Path
        Path to txt file containing paths to data.
    input_size : int
        Size of input

    Returns
    -------
    dataset : tf.data.Dataset
    """
    dataset = tf.data.Dataset.from_generator(
        generator=generator_train_rrr,
        args=(str(path_txt),),
        output_types=(tf.float64, tf.int8),
        output_shapes=(
            tf.TensorShape((2, input_size, input_size)),
            tf.TensorShape(1),
        ),
    )

    return dataset


def get_model_checkpoint(path_checkpoint: Path):
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(path_checkpoint),
        save_weights_only=False,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )
    return checkpoint


def make_model(
        decay: int,
        input_size: int,
        initial_lr: float = 0.001,
) -> keras.models.Model:
    weight_reg = 1e-4

    model = Sequential()
    model.add(Reshape((input_size, input_size, 1)))
    model.add(Conv2D(256, (3, 3), activation="relu"), kernel_regularizer=regularizers.l1(weight_reg))
    model.add(MaxPooling2D((4, 4)))
    model.add(Conv2D(288, (3, 3), activation="relu"), kernel_regularizer=regularizers.l1(weight_reg))
    model.add(MaxPooling2D((4, 4)))
    model.add(Conv2D(288, (3, 3), activation="relu"), kernel_regularizer=regularizers.l1(weight_reg))
    model.add(Flatten())
    model.add(Dense(288, activation="relu"), kernel_regularizer=regularizers.l1(weight_reg))
    model.add(Dense(128, activation="relu"), kernel_regularizer=regularizers.l1(weight_reg))
    model.add(Dense(96, activation="relu"), kernel_regularizer=regularizers.l1(weight_reg))
    model.add(Dense(16, activation="relu"), kernel_regularizer=regularizers.l1(weight_reg))
    model.add(Dense(2))

    lr_schedule = keras.experimental.CosineDecayRestarts(
        initial_learning_rate=initial_lr, first_decay_steps=decay
    )
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # # Debug - plot model
    # keras.utils.plot_model(model, show_shapes=True, dpi=300)
    # model.summary()

    return model


def make_custom_model(
        gamma: float,
        input_size: int,
        steps_per_epoch: int,
        epochs: int,
        initial_lr: float = 1e-4,
        saved_weights: Path = None,
) -> keras.models.Model:
    inputs = Input((input_size, input_size))
    reshape = Reshape((input_size, input_size, 1))(inputs)
    conv2d_1 = Conv2D(256, (3, 3), activation="relu")(reshape)
    maxpool_1 = MaxPooling2D((4, 4))(conv2d_1)
    conv2d_2 = Conv2D(288, (3, 3), activation="relu")(maxpool_1)
    maxpool_2 = MaxPooling2D((4, 4))(conv2d_2)
    conv2d_3 = Conv2D(288, (3, 3), activation="relu")(maxpool_2)
    flatten = Flatten()(conv2d_3)
    dense1 = Dense(288, activation="relu")(flatten)
    dense2 = Dense(128, activation="relu")(dense1)
    dense3 = Dense(96, activation="relu")(dense2)
    dense4 = Dense(16, activation="relu")(dense3)
    dense5 = Dense(2, activation="relu")(dense4)
    model = CustomModel(inputs, dense5, gamma)

    # lr_schedule = keras.experimental.CosineDecayRestarts(
    #     initial_learning_rate=initial_lr,
    #     alpha=initial_lr / 100,
    #     first_decay_steps=steps_per_epoch * (epochs // 10),
    #     t_mul=1.5,
    #     m_mul=0.9
    # )
    # initial_lr = lr_schedule

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=initial_lr),
        run_eagerly=True,
    )

    if saved_weights is not None:
        model.load_weights(str(saved_weights))

    # # Debug - plot model
    # keras.utils.plot_model(model, show_shapes=True, dpi=300)
    # model.summary()

    return model


def save_training_description(
        filename: str,
        kwargs: dict,
        history,
        model: tf.keras.models.Model,
):
    num_epochs = len(history.epoch)  # Number of epochs
    history = history.history
    acc = history["accuracy"][-1]  # Accuracy
    val_acc = history["val_accuracy"][-1]  # Val accuracy

    # Add to kwargs
    kwargs["num_epochs"] = num_epochs
    kwargs["acc"] = acc
    kwargs["val_acc"] = val_acc

    write_str = [f"{key} {kwargs[key]}" for key in kwargs.keys()]
    write_str = "\n".join(write_str)
    write_str += "\n=========\n"

    # Save training description
    with open(filename, "w") as file:
        file.write(write_str)
        file.write("\n\n")
        model.summary(
            print_fn=lambda line: file.write(line + "\n")
        )  # Write model summary to file
