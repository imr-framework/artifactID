import configparser
from pathlib import Path

import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score
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


def __get_dict_from_log(log: list):
    for line in log:
        if '{' in line and '}' in line:
            return line


def main(batch_size: int, data_root: Path, path_pretrained_model: Path, input_shape: int):
    # =========
    # SET UP TESTING
    # =========
    path_model_load = path_pretrained_model / 'model.hdf5'  # Keras model
    print('Loading model...')
    model = load_model(str(path_model_load))  # Load model

    # =========
    # TESTING
    # =========
    print(f'Performing inference...')

    with open(data_root / 'val.txt', 'r') as f:
        path_eval_npy = f.readlines()
    path_eval_npy.pop(0)

    arr_labels = []
    for path in path_eval_npy:
        path = Path(path.strip())
        if 'gibbs' in path.parent.name.lower():
            arr_labels.append(1)
        elif 'noartifact' in path.parent.name.lower():
            arr_labels.append(0)
    arr_labels = np.array(arr_labels)

    # Make dataset from generator
    input_shape = (input_shape, input_shape, 1)
    output_types = (tf.float16)
    output_shapes = (tf.TensorShape(input_shape))
    dataset_eval = tf.data.Dataset.from_generator(generator=data_ops.generator_inference,
                                                  args=[path_eval_npy],
                                                  output_types=output_types,
                                                  output_shapes=output_shapes).batch(batch_size=batch_size)

    # Inference
    y_pred = model.predict(x=dataset_eval)
    y_pred = np.argmax(y_pred, axis=-1)

    acc = len(np.where(y_pred == arr_labels)[0]) / len(arr_labels)
    print(f'Overall: {acc}')
    y_pred_class0 = y_pred[np.where(arr_labels == 0)]
    acc_class0 = len(np.where(y_pred_class0 == 0)[0]) / len(y_pred_class0)
    print(f'Class 0: {acc_class0}')
    y_pred_class1 = y_pred[np.where(arr_labels == 1)]
    acc_class1 = len(np.where(y_pred_class1 == 1)[0]) / len(y_pred_class1)
    print(f'Class 1: {acc_class1}')

    # =========
    # Find TP, TN, FP, FN
    pred_class0 = np.where(y_pred == 0)[0]
    pred_class1 = np.where(y_pred == 1)[0]
    labels_class0 = np.where(arr_labels == 0)[0]
    labels_class1 = np.where(arr_labels == 1)[0]
    tp = np.intersect1d(labels_class1, pred_class1)[:3]
    tn = np.intersect1d(labels_class0, pred_class0)[:3]
    fn = np.intersect1d(labels_class1, pred_class0)[:3]
    # fp = np.intersect1d(labels_class0, pred_class1)[0]
    path_eval_npy = np.array(path_eval_npy)
    print(f'TP - {path_eval_npy[tp]}')
    print(f'TN - {path_eval_npy[tn]}')
    print(f'FN - {path_eval_npy[fn]}')
    # print(f'FP - {path_eval_npy[fp]}')

    # =========
    # Precision, recall and AUC
    precision = precision_score(y_true=arr_labels, y_pred=y_pred)
    recall = recall_score(y_true=arr_labels, y_pred=y_pred)
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    # print(f'AUC: {roc_auc}')

    # Confusion matrix
    cm = confusion_matrix(y_true=arr_labels, y_pred=y_pred, normalize='true')
    # np.save(arr=cm, file='godwin_cm_test.npy')
    print(cm)

    # Plot confusion matrix
    commands = ['No artifact', 'Gibbs']
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, xticklabels=commands, yticklabels=commands,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()


if __name__ == '__main__':
    # =========
    # READ CONFIG
    # =========
    # Read settings.ini configuration file
    path_settings = 'settings.ini'
    config = configparser.ConfigParser()
    config.read(path_settings)

    config_test = config['EVALUATE']
    batch_size = int(config_test['batch_size'])
    input_shape = int(config_test['input_shape'])  # Patch size
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

    arr_dicom = path_read_data.glob('**\*')
    arr_dicom = list(filter(lambda f: not f.is_dir(), arr_dicom))

    # Perform inference
    main(batch_size=batch_size,
         data_root=path_read_data,
         input_shape=input_shape,
         path_pretrained_model=path_pretrained_model)
