from pathlib import Path

import numpy as np
import sigpy
import tensorflow as tf
from skimage.transform import rotate

from artifactID.datagen.gibbs_datagen import cutart
from common_utils import preprocessor


def generator_train_rrr_v2(path_train_txt: bytes, batch_size: int, steps_per_epochs: int, total_epochs: int):
    epoch_counter = 1
    # Calculate line to derive phantom-real data split
    phantom_split_start = 0.75
    phantom_split_end = 0.25
    phantom_slope = (phantom_split_end - phantom_split_start) / (total_epochs - 1)
    line_constant = phantom_split_end - phantom_slope * total_epochs

    path_train_txt = Path(path_train_txt.decode())
    files = path_train_txt.read_text().splitlines()[1:]

    random_idx = np.arange(len(files))

    while True:
        np.random.shuffle(random_idx)  # Shuffle each epoch for real data
        phantom_split = phantom_slope * epoch_counter + line_constant  # Calculate phantom split
        n_phantom = batch_size  # int(phantom_split * batch_size)  # Number of phantom samples
        n_real_data = batch_size - n_phantom  # Number of real data samples

        for i in range(steps_per_epochs):  # Steps per epoch number of batches
            for p in range(n_phantom):
                phantom = sigpy.shepp_logan((256, 256))
                kspace = sigpy.fft(phantom)
                crop = np.random.randint(low=48, high=64)
                dim = np.random.randint(low=0, high=2)
                # Gibbs ringing
                if dim == 0:
                    kspace[:crop] = 0
                    kspace[-crop:] = 0
                else:
                    kspace[:, :crop] = 0
                    kspace[:, -crop:] = 0
                phantom_gibbs = np.abs(sigpy.ifft(kspace))
                # Data augmentation with 50% chance
                if np.random.uniform() <= 0.5:
                    rand_rotate = np.random.randint(low=0, high=360)
                    phantom = rotate(np.abs(phantom), rand_rotate)
                    phantom_gibbs = rotate(phantom_gibbs, rand_rotate)
                phantom_gibbs, cutart_mask = cutart(phantom, phantom_gibbs)  # CutArt
                phantom_gibbs = preprocessor.normalize_per_slice(phantom_gibbs)

                x = np.stack((phantom_gibbs, cutart_mask), axis=0)
                y = tf.constant((1,), dtype=tf.int8)

                yield x, y

            # Real data - load image and label
            for r in range(n_real_data):
                x = np.load(str(files[i].strip()))
                y = Path(files[i]).parent.parent.name
                y = 1 if y == "gibbs" else 0
                y = tf.constant((y,), dtype=tf.int8)
                mask = np.zeros_like(x)

                x = np.stack((x, mask), axis=0)  # (Image, mask)

                yield x, y  # (Image, mask), label

        epoch_counter += 1


def make_dataset_from_generator_train_rrr_v2(
        path_txt: Path, input_size: int, batch_size: int, steps_per_epoch: int, total_epochs: int
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
        generator=generator_train_rrr_v2,
        args=(str(path_txt), batch_size, steps_per_epoch, total_epochs),
        output_types=(tf.float64, tf.int8),
        output_shapes=(
            tf.TensorShape((2, input_size, input_size)),
            tf.TensorShape(1),
        ),
    )

    return dataset
