import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from artifactID.train.train_utils_v2 import generator_train_rrr_v2

batch_size = 8
input_size = 256
path_txt = r"D:\Sravan\Data\Datagen\ArtifactID_v2\IXI-T1\train.txt"

dataset_train = tf.data.Dataset.from_generator(
    generator=generator_train_rrr_v2,
    args=(str(path_txt), batch_size, 4, 10),
    output_types=(tf.float64, tf.float16),
    output_shapes=(
        tf.TensorShape((2, input_size, input_size)),
        tf.TensorShape(1),
    ),
)

dataset_train = dataset_train.batch(batch_size=batch_size)
dataset = dataset_train.take(batch_size)  # batch_size number of batches
batch = list((dataset.as_numpy_iterator()))[-1]
batch = zip(batch[0], batch[1])
collage = []
for image, label in batch:
    col = np.vstack([image[0], image[1]])
    collage.append(col)
collage = np.hstack(collage)
plt.imshow(collage, cmap="gray")
plt.show()
