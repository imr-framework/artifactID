import pickle
from datetime import datetime
from pathlib import Path
from time import time

from keras.models import load_model
from tensorflow import keras

from artifactID.train.finetune_utils import make_dataset_from_generator_finetune_rrr_hurd
from artifactID.train.train_utils import save_training_description, get_model_checkpoint, make_custom_model


def train(
        batch_size: int,
        gamma: float,
        epochs: int,
        input_size: int,
        path_model: Path,
        path_train_txt: Path,
        path_val_txt: Path,
        initial_lr: float = 0.001,
):
    # Make save folders
    output_folder = path_model.parent / "finetuned"
    if not output_folder.exists():
        output_folder.mkdir(parents=False)

    # =========
    # TF.DATASET GENERATORS
    # =========
    dataset_train = make_dataset_from_generator_finetune_rrr_hurd(path_train_txt, input_size)
    dataset_train = dataset_train.batch(batch_size=batch_size)

    dataset_val = make_dataset_from_generator_finetune_rrr_hurd(path_val_txt, input_size)
    dataset_val = dataset_val.batch(batch_size=batch_size)

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
    model = make_custom_model(
        input_size=input_size,
        steps_per_epoch=train_steps,
        epochs=epochs,
        gamma=gamma,
        initial_lr=initial_lr,
    )
    loaded_model = load_model(str(path_model))
    for i in range(len(model.layers)):
        model.layers[i].set_weights(loaded_model.layers[i].get_weights())
    for l in model.layers:
        if "dense" in l.name:
            l.trainable = True
        else:
            l.trainable = False

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
    kwargs = {
        "dur (s)": dur,
        "batch_size": batch_size,
        "initial_lr": initial_lr,
        "gamma": gamma
    }
    save_training_description(
        filename=filename,
        history=history,
        model=model,
        kwargs=kwargs
    )

    # =========
    # SAVE MODEL TO DISK
    # =========
    model.save(str(output_folder / f"model_{time_string}"))
    with open(str(output_folder / "history"), "wb") as pkl:  # Save history
        pickle.dump(history.history, pkl)


if __name__ == "__main__":
    path_root = r"D:\Sravan\Data\Source\202312-T1w_MtSinai_YasminHurd\select"
    path_model = Path(r"../models/20220902_1515_gamma1_finalpick/best_model.100")
    path_train_txt = Path(path_root) / "train.txt"
    path_val_txt = Path(path_root) / "val.txt"

    batch_size = 2
    gamma = 0
    input_size = 256
    epochs = 100
    initial_lr = 1e-6
    train(
        batch_size=batch_size,
        epochs=epochs,
        gamma=gamma,
        initial_lr=initial_lr,
        input_size=input_size,
        path_model=path_model,
        path_train_txt=path_train_txt,
        path_val_txt=path_val_txt,
    )
