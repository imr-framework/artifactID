import configparser
from pathlib import Path

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

from artifactID.common import data_ops

# =========
# TENSORFLOW INIT
# =========
# Prevent crash
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)


def main(arr_files, batch_size, format, path_pretrained_model, patch_size):
    path_model_load = path_pretrained_model / 'model.hdf5'  # Keras model
    print('Loading model...')
    model = load_model(str(path_model_load))  # Load model
    layers = model.layers[2:]

    # Make generator for feeding data to the model
    def __generator_patches(patches: list):
        for p in patches:
            p = p.astype(np.float16)
            p = np.expand_dims(a=p, axis=2)
            yield {'input_1': p, 'input_2': p}

    for counter, vol in enumerate(data_ops.generator_inference(x=arr_files, file_format=format)):
        vol = data_ops.patch_size_compatible_zeropad(vol=vol, patch_size=patch_size)
        patches, patch_map = data_ops.get_patches_per_slice(vol=vol, patch_size=patch_size)
        patches = data_ops.normalize_patches(patches=patches)

        # for counter2, p in enumerate(patches):
        #     print(counter2)
        #     save_path = Path(r"")
        #     file_name = save_path / f'patch_{counter2}.npy'
        #     np.save(file=file_name, arr=p)

        input_output_shape = (patch_size, patch_size, 1)
        output_types = ({'input_1': tf.float16,
                         'input_2': tf.float16})
        output_shapes = ({'input_1': tf.TensorShape(input_output_shape),
                          'input_2': tf.TensorShape(input_output_shape)})

        # Make dataset from generator
        dataset = tf.data.Dataset.from_generator(generator=__generator_patches,
                                                 args=[patches],
                                                 output_types=output_types,
                                                 output_shapes=output_shapes).batch(batch_size=batch_size)

        func_filter_layer = lambda layer: 'conv2d' in layer.name or 'max_pooling2d' in layer.name
        layers = list(filter(func_filter_layer, layers))  # Remove non-CNN layers

        random_int = np.random.randint(low=0, high=len(patches))
        random_patch = patches[random_int]

        for l in layers:
            # Construct temp model and perform inference
            temp_model = Model(inputs=model.input, outputs=l.output)
            y_pred = temp_model.predict(x=dataset)
            pred = y_pred[random_int]

            num_filters = l.output_shape[-1]
            if num_filters == 32:
                num_subplots = (5, 7)
            elif num_filters == 16:
                num_subplots = (4, 5)
            elif num_filters == 8:
                num_subplots = (3, 3)

            fig, ax = plt.subplots(*num_subplots)
            ax = ax.ravel()
            ax[0].imshow(random_patch.astype(np.float), cmap='gray')
            for j in range(y_pred.shape[-1]):
                ax[j + 1].imshow(pred[:, :, j].astype(np.float), cmap='gray')
                ax[j + 1].axis('off')
            plt.suptitle(l.name)
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
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
    path_read_data = config_test['path_read_data']

    # =========
    # DATA CHECK
    # =========
    path_read_data = Path(path_read_data)
    if not path_read_data.exists():
        raise Exception(f'{path_read_data} does not exist')
    path_pretrained_model = Path(path_pretrained_model)
    if not path_pretrained_model.exists():
        raise Exception(f'{path_pretrained_model} does not exist')

    arr_files = data_ops.glob_nifti(path=path_read_data)
    format = 'nifti'
    if len(arr_files) == 0:
        arr_files = data_ops.glob_dicom(path=path_read_data)
        format = 'dicom'
    if len(arr_files) == 0:
        raise ValueError(f'No NIFTI or DICOM files found at {path_read_data}')

    # Perform inference
    main(arr_files=arr_files,
         batch_size=batch_size,
         format=format,
         path_pretrained_model=path_pretrained_model,
         patch_size=patch_size)
