import pickle
from datetime import datetime
from pathlib import Path
from time import time

import tensorflow as tf
from tensorflow import keras

from artifactID.train.train_utils import make_model, make_dataset_from_generator_train, save_training_description, \
    get_model_checkpoint


def train(
        batch_size: int,
        decay: int,
        epochs: int,
        input_size: int,
        path_train_txt: Path,
        path_val_txt: Path,
):
    # Make save folders
    time_string = datetime.now().strftime("%Y%m%d_%H%M")  # Time stamp
    output_folder = Path("models") / f"{time_string}"
    if not output_folder.exists():
        output_folder.mkdir(parents=False)

    # =========
    # TF.DATASET GENERATORS
    # =========
    dataset_train = make_dataset_from_generator_train(path_train_txt, input_size)
    dataset_train = dataset_train.batch(batch_size=batch_size).prefetch(tf.data.AUTOTUNE)

    dataset_val = make_dataset_from_generator_train(path_val_txt, input_size)
    dataset_val = dataset_val.batch(batch_size=batch_size).prefetch(tf.data.AUTOTUNE)

    # =========
    # CALLBACKS
    # =========
    # Model checkpoint callback - checkpoint after every epoch
    path_checkpoint = output_folder / "model.{epoch:02d}"
    model_checkpoint = get_model_checkpoint(path_checkpoint)
    path_log = output_folder / "logs"
    tboard = keras.callbacks.TensorBoard(log_dir=path_log)
    callbacks = [model_checkpoint, tboard]

    # =========
    # TRAINING
    # =========
    # Number of training steps
    train_steps = int(path_train_txt.read_text().splitlines()[0].strip()) // batch_size
    # Number of validation steps
    val_steps = int(path_val_txt.read_text().splitlines()[0].strip()) // batch_size
    if val_steps < 1:
        val_steps = 1

    # Get model
    model = make_model(
        input_size=input_size,
        decay=decay
    )

    start = time()  # Timeit
    history = model.fit(
        dataset_train,
        callbacks=callbacks,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_data=dataset_val,
        validation_steps=val_steps,
    )
    dur = time() - start  # Timeit

    # =========
    # SAVE TRAINING DESCRIPTION
    # =========
    time_string = datetime.now().strftime("%Y%m%d_%H%M")  # Time stamp
    filename = str(output_folder / f"log_{time_string}.txt")
    save_training_description(
        batch_size=batch_size,
        decay=decay,
        filename=filename,
        history=history,
        model=model,
        dur=dur
    )

    # =========
    # SAVE MODEL TO DISK
    # =========
    model.save(str(output_folder / f"model_{time_string}"))
    with open(str(output_folder / "history"), "wb") as pkl:  # Save history
        pickle.dump(history.history, pkl)


if __name__ == "__main__":
    path_root = r"E:\Data\Datagen\ArtifactID\IXI-Gibbs_sag"
    path_train_txt = Path(path_root) / "train.txt"
    path_val_txt = Path(path_root) / "val.txt"

    batch_size = 32
    input_size = 256
    epochs = 100
    decay = 15
    train(
        batch_size=batch_size,
        decay=decay,
        epochs=epochs,
        input_size=input_size,
        path_train_txt=path_train_txt,
        path_val_txt=path_val_txt,
    )
