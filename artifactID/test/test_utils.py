from pathlib import Path

import numpy as np
import tensorflow as tf

from common_utils import data_loader


def generator_inference(path_train_txt: bytes):
    path_train_txt = Path(path_train_txt.decode())
    files = path_train_txt.read_text().splitlines()[1:]

    for i in range(len(files)):
        # Load image
        x = np.load(str(files[i].strip())) * 255.0

        # Debug - print sizes
        # print(x.shape)

        yield x,  # Image


def make_dataset_from_generator_inference(
        path_data: Path, input_size: int
) -> tf.data.Dataset:
    """
    Parameters
    ----------
    path_data : Path
        Path to folder containing DICOMs.
    input_size : int
        Size of input

    Returns
    -------
    dataset : tf.data.Dataset
    """
    vol = data_loader.load_data(path_data=path_data, data_format='dicom', normalize=True, target_size=input_size)
    vol *= 255.0
    vol = np.moveaxis(vol, -1, 0)
    dataset = tf.data.Dataset.from_tensor_slices(vol)

    return dataset
