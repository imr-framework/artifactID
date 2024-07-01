from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf


def generator_test(path_train_txt: bytes):
    path_train_txt = Path(path_train_txt.decode())
    files = path_train_txt.read_text().splitlines()[1:]

    for i in range(len(files)):
        # Load image
        x = np.load(str(files[i].strip())) * 255.0

        # Debug - print sizes
        # print(x.shape)

        yield x,  # Image


def get_tp_tn_fp_fn(
    pred_labels: np.ndarray, true_labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_class0 = np.where(pred_labels == 0)[0]
    pred_class1 = np.where(pred_labels == 1)[0]
    labels_class0 = np.where(true_labels == 0)[0]
    labels_class1 = np.where(true_labels == 1)[0]
    tp = np.intersect1d(labels_class1, pred_class1)
    tn = np.intersect1d(labels_class0, pred_class0)
    fn = np.intersect1d(labels_class1, pred_class0)
    fp = np.intersect1d(labels_class0, pred_class1)

    return tp, tn, fp, fn


def get_true_labels_from_txt(path_txt: Path) -> np.ndarray:
    true_labels = []
    for path in path_txt.read_text().splitlines()[1:]:
        path = Path(path.strip())
        folder1 = path.parent.parent.name.lower()
        folder2 = path.parent.name.lower()
        if folder1 in ["gibbs", "wrap", "motion"] or folder2 in ["gibbs", "wrap", "motion"]:
            true_labels.append(1)
        elif folder1 == "noartifact" or folder2 == "noartifact":
            true_labels.append(0)

    return np.array(true_labels)


def make_dataset_from_generator_test(
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
        generator=generator_test,
        args=(str(path_txt),),
        output_types=(tf.float64,),
        output_shapes=(tf.TensorShape((input_size, input_size)),),
    )

    return dataset
