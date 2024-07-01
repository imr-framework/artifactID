from pathlib import Path

import tensorflow as tf
from matplotlib import pyplot as plt
from skimage import feature

from artifactID.train_utils import generator_train_rrr

batch_size = 8
input_size = 256
path_root = Path(r"E:\Data\Datagen\ArtifactID\IXI-Gibbs_sag")
path_train_txt = path_root / "train.txt"

dataset_train = tf.data.Dataset.from_generator(
    generator=generator_train_rrr,
    args=(str(path_train_txt),),
    output_types=(tf.float64, tf.float16),
    output_shapes=(
        tf.TensorShape((2, input_size, input_size)),
        tf.TensorShape(1),
    ),
)
dataset_train = dataset_train.batch(batch_size=batch_size)

# Debug
dataset = dataset_train.take(10)
batch = next(dataset.as_numpy_iterator())
batch = zip(batch[0], batch[1])
for image, label in batch:
    print(image.shape)
    print(type(image), type(label))
    plt.subplot(121)
    plt.imshow(image[0], cmap="gray")
    plt.subplot(122)
    plt.imshow(image[1], cmap="gray")
    plt.suptitle(label)
    plt.show()
