import configparser
from pathlib import Path

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

# =========
# TENSORFLOW INIT
# =========
# Prevent crash
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)


def main(arr_files, path_pretrained_model):
    path_model_load = path_pretrained_model / 'model.hdf5'  # Keras model
    print('Loading model...')
    model = load_model(str(path_model_load))  # Load model
    layers = model.layers[2:]

    func_filter_layer = lambda layer: 'conv2d' in layer.name or 'max_pooling2d' in layer.name
    layers = list(filter(func_filter_layer, layers))  # Remove non-CNN layers

    for path_load in arr_files:
        p = np.load(path_load)
        # plt.imshow(p.astype(np.float))
        # plt.show()
        for l in layers:
            p = np.expand_dims(p, axis=0)
            p = np.expand_dims(p, axis=3)
            # Construct temp model and perform inference
            temp_model = Model(inputs=model.input, outputs=l.output)
            y_pred = temp_model.predict(x={'input_1': p, 'input_2': p})

            num_filters = l.output_shape[-1]
            if num_filters == 32:
                num_subplots = (5, 7)
            elif num_filters == 16:
                num_subplots = (4, 5)
            elif num_filters == 8:
                num_subplots = (3, 3)

            p = np.squeeze(p)
            y_pred = np.squeeze(y_pred)
            fig, ax = plt.subplots(*num_subplots)
            ax = ax.ravel()
            ax[0].imshow(p.astype(np.float), cmap='gray')
            for j in range(y_pred.shape[-1]):
                ax[j + 1].imshow(y_pred[:, :, j].astype(np.float), cmap='gray')
                ax[j + 1].axis('off')
                ax[j + 1].title.set_text(np.sum(y_pred[:, :, j]))
            plt.suptitle(l.name)
            plt.tight_layout()
            path_save = path_load.parent / (path_load.stem + f'{l.name}_.jpg')
            plt.savefig(path_save)
            # plt.show()


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

    # =========
    # DATA CHECK
    # =========
    path_read_data = Path(
        r"")
    if not path_read_data.exists():
        raise Exception(f'{path_read_data} does not exist')
    path_pretrained_model = Path(path_pretrained_model)
    if not path_pretrained_model.exists():
        raise Exception(f'{path_pretrained_model} does not exist')

    arr_files = path_read_data.glob('*.npy')

    # Perform inference
    main(arr_files=arr_files,
         path_pretrained_model=path_pretrained_model)
