import math
import pickle
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import load_model

from artifactID.train_utils import generator_train


def main(artifact: str, batch_size: int, epochs: int, input_size: int, path_model: Path, path_train_txt: Path,
         path_val_txt: Path):
    output_folder = path_model.parent

    # =========
    # TF.DATASET GENERATORS
    # =========
    input_shape = (input_size, input_size, 1)

    dataset_train = tf.data.Dataset.from_generator(generator=generator_train,
                                                   args=[str(path_train_txt), artifact],
                                                   output_types=(tf.float64, tf.int8),
                                                   output_shapes=(tf.TensorShape(input_shape), tf.TensorShape([1])))
    dataset_train = dataset_train.batch(batch_size=batch_size)
    dataset_train.prefetch(buffer_size=batch_size)
    dataset_val = tf.data.Dataset.from_generator(generator=generator_train,
                                                 args=[str(path_val_txt), artifact],
                                                 output_types=(tf.float64, tf.int8),
                                                 output_shapes=(tf.TensorShape(input_shape), tf.TensorShape([1])))
    dataset_val = dataset_val.batch(batch_size=batch_size)
    dataset_val = dataset_val.prefetch(buffer_size=batch_size)

    # =========
    # TRAINING
    # =========
    train_steps = int(path_train_txt.read_text().splitlines()[0].strip()) / batch_size  # Number of training steps
    train_steps = math.ceil(train_steps)
    val_steps = int(path_val_txt.read_text().splitlines()[0].strip()) / batch_size  # Number of validation steps
    val_steps = math.ceil(val_steps)

    model = load_model(str(path_model))  # Load model
    history = model.fit(x=dataset_train,
                        steps_per_epoch=train_steps,
                        validation_data=dataset_val,
                        validation_steps=val_steps,
                        epochs=epochs)

    # =========
    # SAVE MODEL TO DISK
    # =========
    model.save(str(output_folder / 'finetune.hdf5'))  # Save model
    with open(str(output_folder / 'history_finetune'), 'wb') as pkl:  # Save history
        pickle.dump(history.history, pkl)


if __name__ == '__main__':
    path_root = Path(r"D:\CU Data\Datagen\artifactID_GE\20211028\e470\finetune_segment1_motion_sim_sorted")
    path_model = Path(r"output/20211101_1905_motion_sagittal/model.hdf5")
    path_train_txt = path_root / "train.txt"
    path_val_txt = path_root / "val.txt"
    artifact = 'motion'

    batch_size = 32
    input_size = 256

    epochs = 25
    main(path_train_txt=path_train_txt, path_val_txt=path_val_txt, path_model=path_model,
         input_size=input_size, batch_size=batch_size, epochs=epochs,
         artifact=artifact)
