import tensorflow as tf
from matplotlib import pyplot as plt

from artifactID.train.finetune_utils import generator_finetune_rrr_hurd

batch_size = 32
input_size = 256
path_txt = r"D:\Sravan\Data\Source\202312-T1w_MtSinai_YasminHurd\select\train.txt"

# Debug
dataset_train = tf.data.Dataset.from_generator(
    generator=generator_finetune_rrr_hurd,
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
