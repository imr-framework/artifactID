from pathlib import Path
from typing import Tuple

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve
from tensorflow.keras.models import load_model

from artifactID.eval.eval_utils import (
    make_dataset_from_generator_test,
    get_tp_tn_fp_fn,
    get_true_labels_from_txt,
)


def main(batch_size: int, input_size: int, path_model: Path, path_txt: Path, save_tptnfpfn: bool, save_cm: bool = False,
         show_cm: bool = False) -> Tuple[float, float, np.ndarray]:
    # =========
    # SET UP TESTING
    # =========
    print("Loading model...")
    model = load_model(str(path_model))  # Load model

    # =========
    # TF.DATASET GENERATORS
    # =========
    dataset_eval = make_dataset_from_generator_test(path_txt, input_size)
    dataset_eval = dataset_eval.batch(batch_size=batch_size).prefetch(2)

    # =========
    # INFERENCE
    # =========
    print(f"Performing inference...")
    # Inference
    pred_labels = model.predict(x=dataset_eval)
    if pred_labels.shape[-1] == 1:  # Sigmoid
        pred_labels = (pred_labels > 0.5).astype(int)
        pred_labels = np.concatenate(pred_labels)
    else:  # Softmax
        pred_scores = np.max(pred_labels, axis=-1)
        pred_labels = np.argmax(pred_labels, axis=-1)

        # =========
    # RESULTS
    # =========
    true_labels = get_true_labels_from_txt(path_txt)  # True labels
    acc = len(np.where(pred_labels == true_labels)[0]) / len(true_labels)
    ind_y_pred_class0 = np.where(pred_labels == 0)
    acc_class0 = (np.count_nonzero(true_labels[ind_y_pred_class0] == 0)) / (true_labels == 0).sum()
    ind_y_pred_class1 = np.where(pred_labels == 1)
    acc_class1 = (np.count_nonzero(true_labels[ind_y_pred_class1] == 1)) / (true_labels == 1).sum()

    print(f"Overall: {acc}")
    print(f"Class 0: {acc_class0}")
    print(f"Class 1: {acc_class1}")

    # =========
    # Find TP, TN, FP, FN
    tp, tn, fp, fn = get_tp_tn_fp_fn(pred_labels=pred_labels, true_labels=true_labels)
    eval_paths = np.array(path_txt.read_text().splitlines()[1:])
    # print(f"TP - {eval_paths[tp]}")
    # print(f"TN - {eval_paths[tn]}")
    # print(f"FN - {eval_paths[fn]}")
    # print(f"FP - {eval_paths[fp]}")
    if save_tptnfpfn:
        path_root = path_txt.parent
        # TP
        tp = eval_paths[tp]
        text = [str(len(tp)), *tp]
        text = "\n".join(text)
        (path_root / "tp.txt").write_text(text)

        # TN
        tn = eval_paths[tn]
        text = [str(len(tn)), *tn]
        text = "\n".join(text)
        (path_root / "tn.txt").write_text(text)

        # FP
        fp = eval_paths[fp]
        text = [str(len(fp)), *fp]
        text = "\n".join(text)
        (path_root / "fp.txt").write_text(text)

        # FN
        fn = eval_paths[fn]
        text = [str(len(fn)), *fn]
        text = "\n".join(text)
        (path_root / "fn.txt").write_text(text)

    # =========
    # Precision, recall and ROC
    precision = precision_score(y_true=true_labels, y_pred=pred_labels)
    recall = recall_score(y_true=true_labels, y_pred=pred_labels)
    roc = roc_curve(y_true=true_labels, y_score=pred_scores)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    # Confusion matrix
    cm = confusion_matrix(y_true=true_labels, y_pred=pred_labels, normalize="true")
    if save_cm:
        np.save(arr=cm, file=f'{path_txt.parent.name}_cm.npy')
        np.save(arr=roc, file=f'{path_txt.parent.name}_roc.npy')
    print(cm)

    # Plot confusion matrix
    if show_cm:
        commands = ["No artifact", "Gibbs"]
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, xticklabels=commands, yticklabels=commands, annot=True, fmt="g")
        plt.xlabel("Prediction")
        plt.ylabel("Label")
        plt.show()

    return precision, recall, cm


if __name__ == "__main__":
    batch_size = 64
    input_size = 256
    path_root = Path(r"D:\Sravan\Data\Datagen\ArtifactID\LAC-T1_resampled")
    path_txt = path_root / "inference.txt"
    path_model = Path(r"../models/20220902_1515_gamma1_finalpick/best_model.100")

    # Perform eval
    main(
        path_model=path_model,
        path_txt=path_txt,
        batch_size=batch_size,
        input_size=input_size,
        save_tptnfpfn=False,
        save_cm=True,
        show_cm=False
    )
