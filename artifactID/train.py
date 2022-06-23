import math
import pickle
from datetime import datetime
from pathlib import Path
from time import time

import tensorflow as tf

from artifactID.train_utils import generator_train, make_model, save_training_description


def main(artifact: str, batch_size: int, epochs: int, input_size: int, path_train_txt: Path, path_val_txt: Path):
    # Make save folders
    time_string = datetime.now().strftime('%Y%m%d_%H%M')  # Time stamp
    output_folder = Path('output') / f'{time_string}'
    if not output_folder.exists():  # Make output/* directory
        output_folder.mkdir(parents=True)

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
    # CALLBACKS
    # =========
    # Model checkpoint callback - checkpoint after every epoch
    path_checkpoint = Path(output_folder) / 'model.{epoch:02d}.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=str(path_checkpoint),
                                                                   save_weights_only=False,
                                                                   monitor='val_accuracy',
                                                                   mode='max',
                                                                   save_best_only=True)
    callbacks = [model_checkpoint_callback]

    # =========
    # TRAINING
    # =========
    train_steps = int(path_train_txt.read_text().splitlines()[0].strip()) / batch_size  # Number of training steps
    train_steps = math.ceil(train_steps)
    val_steps = int(path_val_txt.read_text().splitlines()[0].strip()) / batch_size  # Number of validation steps
    val_steps = math.ceil(val_steps)

    model = make_model(artifact=artifact, input_shape=input_shape)  # Get model
    model.save_weights(str(output_folder / 'best_weights.h5'))  # Save weights
    start = time()  # Timeit
    history = model.fit(x=dataset_train,
                        callbacks=callbacks,
                        steps_per_epoch=train_steps,
                        validation_data=dataset_val,
                        validation_steps=val_steps,
                        epochs=epochs)
    dur = time() - start  # Timeit

    # =========
    # SAVE TRAINING DESCRIPTION
    # =========
    time_string = datetime.now().strftime('%Y%m%d_%H%M')  # Time stamp
    filename = str(output_folder / f'log_{time_string}.txt')
    save_training_description(batch_size=batch_size, dur=dur, filename=filename, history=history,
                              model=model)

    # =========
    # SAVE MODEL TO DISK
    # =========
    model.save(str(output_folder / 'model.hdf5'))  # Save model
    with open(str(output_folder / 'history'), 'wb') as pkl:  # Save history
        pickle.dump(history.history, pkl)


if __name__ == '__main__':
    path_root = Path(r"D:\CU Data\Datagen\artifactID_motion_GE_Siemens")
    path_train_txt = path_root / "train.txt"
    path_val_txt = path_root / "val.txt"
    artifact = 'motion'

    if artifact == 'gibbs':
        batch_size = 64
        input_size = 63
    elif artifact == 'wrap':
        batch_size = 32
        input_size = 256
    elif artifact == 'motion':
        batch_size = 32
        input_size = 256
    else:
        raise ValueError(f'Unknown value for artifact: {artifact}')

    epochs = 25
    main(path_train_txt=path_train_txt, path_val_txt=path_val_txt,
         input_size=input_size, batch_size=batch_size, epochs=epochs,
         artifact=artifact)
