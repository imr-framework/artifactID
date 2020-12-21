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


def main(arr_npy, batch_size, path_pretrained_model, arr_labels=None):
    path_model_load = path_pretrained_model / 'model.hdf5'  # Keras model
    print('Loading model...')
    model = load_model(str(path_model_load))  # Load model

    output_layers = [l.output for l in model.layers]
    lambda_filter_layer = lambda layer: 'conv2d' in layer.name  # or 'max_pooling2d' in layer.name
    output_layers = list(filter(lambda_filter_layer, output_layers))  # Remove non-CNN layers

    viz_model = Model(inputs=model.input, outputs=output_layers)

    # Make dataset from generator
    dataset = tf.data.Dataset.from_tensor_slices(arr_npy).batch(batch_size=batch_size)
    y_pred = model.predict(dataset)
    y_pred = np.argmax(y_pred, axis=-1)
    print(f'Pred {np.round(y_pred)}')
    if arr_labels is not None:
        print(f'Labels {arr_labels}')

    # TWO-SPLIT, 2 SLICES, 7 FILTERS EACH
    arr_preds = []
    for sli in arr_npy:
        sli = np.expand_dims(sli, axis=0)
        row = [sli]
        row.extend(viz_model.predict(sli))
        arr_preds.append(row)

    layer = 0  # Layer to viz
    cols = 5  # Number of columns
    num_filters = arr_preds[0][layer + 1].shape[-1]
    filter_counter = 0

    for _ in range(num_filters):  # Loop over each filter in layer
        fig, ax = plt.subplots(len(arr_npy), cols)
        for i in range(len(arr_npy)):  # Viz `cols` - 1 number of filters for each vol in `arr_npy`
            row = arr_preds[i]

            sli = row[0]
            sli = np.squeeze(sli)
            ax[i, 0].imshow(sli, cmap='gray')
            ax[i, 0].axis('off')

            if arr_labels is not None:
                ax[i, 0].title.set_text(f'gt {arr_labels[i]} pred {y_pred[i]}')

            for j in range(1, cols):
                pred = row[1:][layer][..., j]
                pred = np.squeeze(pred)
                ax[i, j].imshow(pred.astype(np.float), cmap='gray', vmax=0.1)
                ax[i, j].axis('off')
                filter_counter += 1
            filter_counter -= cols - 1
        filter_counter += cols - 1

        plt.tight_layout()
        # plt.savefig(f'{filter_counter}.jpg')
        plt.show()
        # plt.pause(5)
        # plt.draw()


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
    input_shape = int(config_test['input_shape'])
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

    arr_npy = []
    arr_labels = []
    for f in path_read_data.glob('**\*.npy'):
        if 'TP' in str(f.stem):
            arr_labels.append(1)
        elif 'FP' in str(f.stem):
            arr_labels.append(0)
        elif 'TN' in str(f.stem):
            arr_labels.append(0)
        elif 'FN' in str(f.stem):
            arr_labels.append(1)
        sli = np.load(str(f))
        if sli.shape != (input_shape, input_shape):
            sli = data_ops.resize(sli, size=input_shape)
        kspace = np.fft.fftshift(np.fft.fftn(sli))
        sli = np.abs(kspace)
        arr_npy.append(sli)
    arr_npy = np.stack(arr_npy, axis=-1)
    arr_npy = np.moveaxis(arr_npy, (0, 1, 2), (1, 2, 0))
    arr_npy = np.expand_dims(arr_npy, 3)
    arr_npy = arr_npy.astype(np.float)

    # Perform inference
    main(arr_npy=arr_npy, arr_labels=arr_labels, batch_size=batch_size, path_pretrained_model=path_pretrained_model)
