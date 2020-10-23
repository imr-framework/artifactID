import configparser
from pathlib import Path

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

# =========
# TENSORFLOW INIT
# =========
# Prevent crash
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)


def main():
    # =========
    # READ CONFIG
    # =========
    # Read settings.ini configuration file
    path_settings = 'settings.ini'
    config = configparser.ConfigParser()
    config.read(path_settings)

    config_test = config['TEST']
    batch_size = int(config_test['batch_size'])
    patch_size = int(config_test['patch_size'])  # Patch size
    path_pretrained_model = config_test['path_pretrained_model']
    path_read_data = r""

    # =========
    # DATA CHECK
    # =========
    path_read_data = Path(path_read_data)
    if not path_read_data.exists():
        raise Exception(f'{path_read_data} does not exist')
    path_pretrained_model = Path(path_pretrained_model)
    if not path_pretrained_model.exists():
        raise Exception(f'{path_pretrained_model} does not exist')

    arr_files = list(path_read_data.glob('**/*.npy'))
    patches = []
    for path_load in arr_files:
        patches.append(np.load(str(path_load)))

    input_output_shape = (patch_size, patch_size, 1)
    output_types = ({'input_1': tf.float16,
                     'input_2': tf.float16})
    output_shapes = ({'input_1': tf.TensorShape(input_output_shape),
                      'input_2': tf.TensorShape(input_output_shape)})

    def __generator_patches(patches: list):
        for p in patches:
            p = p.astype(np.float16)
            p = np.expand_dims(a=p, axis=2)
            yield {'input_1': p, 'input_2': p}

    dataset = tf.data.Dataset.from_generator(generator=__generator_patches,
                                             args=[patches],
                                             output_types=output_types,
                                             output_shapes=output_shapes).batch(batch_size=batch_size)
    path_model_load = path_pretrained_model / 'model.hdf5'  # Keras model
    model = load_model(str(path_model_load))  # Load model
    y_pred = model.predict(x=dataset)
    errors = np.where(y_pred > 0.85)[0]
    print(len(errors))
    for i in errors[-1500:]:
        p = np.load(str(arr_files[i]))
        plt.imshow(p.astype(np.float))
        plt.show()
        print(arr_files[i])


if __name__ == '__main__':
    main()
