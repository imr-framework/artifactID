import tensorflow as tf
from matplotlib import pyplot as plt

from artifactID.train.train_utils import generator_train_rrr

batch_size = 32
input_size = 256
path_txt = r"D:\Sravan\Data\Datagen\ArtifactID_v2\IXI-T1\train.txt"

# Debug
dataset_train = tf.data.Dataset.from_generator(
    generator=generator_train_rrr,
    args=(str(path_txt),),
    output_types=(tf.float64, tf.float16),
    output_shapes=(
        tf.TensorShape((2, input_size, input_size)),
        tf.TensorShape(1),
    ),
)
dataset_train = dataset_train.batch(batch_size=batch_size)
dataset = dataset_train.take(1)
batch = next(dataset.as_numpy_iterator())
batch = zip(batch[0], batch[1])
for image, label in batch:
    image, cutart = image[0], image[1]

    plt.subplot(121)
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(cutart, cmap="gray")
    plt.axis("off")
    plt.suptitle(f"Label: {label}, max: {image.max()}, min: {image.min()}")
    plt.show()

# =========
# Test
# =========
# from artifactID.eval_utils import generator_test

# from pathlib import Path
#
# paths = Path(path_txt).read_text().splitlines()[1:]
# dataset_test = tf.data.Dataset.from_generator(
#     generator=generator_test,
#     args=(str(path_txt),),
#     output_types=(tf.float64,),
#     output_shapes=(
#         tf.TensorShape((input_size, input_size)),
#     ),
# )
# dataset_test = dataset_test.batch(batch_size=batch_size)
#
# # Debug
# dataset = dataset_test.take(10)
# batch = next(dataset.as_numpy_iterator())[0]
# for i, image in enumerate(batch):
#     print(type(image))
#     plt.imshow(image, cmap="gray")
#     plt.title(Path(paths[i]).parent.name)
#     plt.show()
