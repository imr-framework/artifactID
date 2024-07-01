from pathlib import Path

import numpy as np
import pydicom as pyd
import tensorflow as tf

from common_utils import preprocessor


def generator_finetune_rrr_hurd(path_train_txt: bytes):
    # Load CutArt masks
    path_gibbs_meta = Path(r"D:\Sravan\Data\Source\202312-T1w_MtSinai_YasminHurd\select\gibbs")
    mask_files = list(path_gibbs_meta.glob("*.npy"))
    arr_mask_vol = dict()
    for mask_file in mask_files:
        subject = mask_file.stem
        arr_mask_vol[subject] = np.load(str(mask_file))

    path_train_txt = Path(path_train_txt.decode())
    files = path_train_txt.read_text().splitlines()[1:]

    random_idx = np.arange(len(files))

    while True:
        np.random.shuffle(random_idx)  # Shuffle each epoch
        for i in random_idx:
            # Load image and label
            x = pyd.dcmread(str(files[i].strip())).pixel_array
            x = preprocessor.resize(x, 256)
            x = np.clip(x, *np.percentile(x, (0.5, 99.5)))
            x = preprocessor.normalize_per_slice(x)
            y = Path(files[i]).parent.parent.name
            y = 1 if y == "gibbs" else 0
            y = tf.constant((y,), dtype=tf.int8)

            if y:  # Generate mask for RRR
                subject = Path(files[i]).parent.name
                mask_vol = arr_mask_vol[subject]
                slice_number = int(Path(files[i]).stem)
                mask = mask_vol[..., slice_number]
                mask = 1.0 - mask
            else:
                mask = np.zeros_like(x)

            # Debug - print sizes
            # print(x.shape)

            x = np.stack((x, mask), axis=0) * 255.0

            yield x, y  # (Image, mask), label


def make_dataset_from_generator_finetune_rrr_hurd(
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
        generator=generator_finetune_rrr_hurd,
        args=(str(path_txt),),
        output_types=(tf.float64, tf.int8),
        output_shapes=(
            tf.TensorShape((2, input_size, input_size)),
            tf.TensorShape(1),
        ),
    )

    return dataset
