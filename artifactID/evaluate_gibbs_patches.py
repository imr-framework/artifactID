from pathlib import Path

import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from tensorflow.keras.models import load_model

from artifactID.utils import get_tv_patches

# =========
# TENSORFLOW INIT
# =========
# Prevent crash
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def main(path_model: Path, path_txt: Path):
    # =========
    # SET UP TESTING
    # =========
    print('Loading model...')
    model = load_model(str(path_model))  # Load model

    # =========
    # TESTING
    # =========
    print(f'Performing inference...')

    path_eval_npy = path_txt.read_text().splitlines()
    path_eval_npy.pop(0)

    arr_labels = []
    arr_ypred = []
    for path in path_eval_npy:
        path = Path(path.strip())
        if path.parent.name.lower() in ['gibbs', 'wrap']:
            arr_labels.append(1)
        elif path.parent.name.lower() == 'noartifact':
            arr_labels.append(0)

        # Load image and convert to patches
        path = path_txt.parent / path
        npy = np.load(str(path))
        npy_patches = get_tv_p  atches(npy)

        # Inference
        y_pred = model.predict(x=npy_patches)
        y_pred = np.argmax(y_pred, axis=-1)
        arr_ypred.append(int(np.any(y_pred)))

    arr_labels = np.array(arr_labels)
    arr_ypred = np.array(arr_ypred)

    acc = len(np.where(arr_ypred == arr_labels)[0]) / len(arr_labels)
    print(f'Overall: {acc}')
    y_pred_class0 = arr_ypred[np.where(arr_labels == 0)]
    acc_class0 = len(np.where(y_pred_class0 == 0)[0]) / len(y_pred_class0)
    print(f'Class 0: {acc_class0}')
    y_pred_class1 = arr_ypred[np.where(arr_labels == 1)]
    acc_class1 = len(np.where(y_pred_class1 == 1)[0]) / len(y_pred_class1)
    print(f'Class 1: {acc_class1}')

    # =========
    # Find TP, TN, FP, FN
    pred_class0 = np.where(arr_ypred == 0)[0]
    pred_class1 = np.where(arr_ypred == 1)[0]
    labels_class0 = np.where(arr_labels == 0)[0]
    labels_class1 = np.where(arr_labels == 1)[0]
    tp = np.intersect1d(labels_class1, pred_class1)[:25]
    tn = np.intersect1d(labels_class0, pred_class0)[:75]
    fn = np.intersect1d(labels_class1, pred_class0)
    fp = np.intersect1d(labels_class0, pred_class1)
    path_eval_npy = np.array(path_eval_npy)
    print(f'TP - {path_eval_npy[tp]}')
    print(f'TN - {path_eval_npy[tn]}')
    print(f'FN - {path_eval_npy[fn]}')
    print(f'FP - {path_eval_npy[fp]}')

    # =========
    # Precision, recall and AUC
    precision = precision_score(y_true=arr_labels, y_pred=arr_ypred)
    recall = recall_score(y_true=arr_labels, y_pred=arr_ypred)
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    # print(f'AUC: {roc_auc}')

    # Confusion matrix
    cm = confusion_matrix(y_true=arr_labels, y_pred=arr_ypred, normalize='true')
    # np.save(arr=cm, file='cm.npy')
    print(cm)

    # Plot confusion matrix
    commands = ['No artifact', 'Gibbs']
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, xticklabels=commands, yticklabels=commands, annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()


if __name__ == '__main__':
    path_root = Path(r"D:\CU Data\Datagen\artifactID_IXI_Gibbs_sagittal")
    path_txt = path_root / 'test.txt'
    path_model = Path(r"output\20211102_0925_Gibbs_sagittal\model.hdf5")

    # Perform inference
    main(path_model=path_model, path_txt=path_txt)
