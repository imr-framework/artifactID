from pathlib import Path
from typing import List, Tuple

import numpy as np


def write_to_txt(idx: np.array, items: Tuple[List, List], path_save: Path, filename: str):
    noartifact_subjects, gibbs_subjects = items


    # Subject-wise split
    noartifact_subjects = np.take(noartifact_subjects, idx)
    gibbs_subjects = np.take(gibbs_subjects, idx)

    np.random.shuffle(noartifact_subjects)  # Shuffle
    np.random.shuffle(gibbs_subjects)  # Shuffle
    text = [str(len(noartifact_subjects))]
    for i in range(len(noartifact_subjects)):
        text.append(str(noartifact_subjects[i]))
        text.append(str(gibbs_subjects[i]))
    text = "\n".join(text)
    (path_save / filename).write_text(text)


def main(path_root: Path, train_split: float = 0.85, val_split: float = 0.10):
    """
    Performs subject-wise data-split.

    Parameters
    ----------
    """
    if train_split + val_split >= 1:
        raise ValueError()

    # Determine number of folders/files depending
    noartifact_folder = path_root / "noartifact"
    noartifact_files = list(noartifact_folder.glob("*/*.dcm"))
    gibbs_folder = path_root / "gibbs"
    gibbs_files = list(gibbs_folder.glob("*/*.dcm"))

    # Splits
    n = len(noartifact_files)  # Dataset size
    train_n = int(train_split * n)  # Train split
    val_n = int(val_split * n)  # Validation split
    test_n = n - train_n - val_n  # Test split

    # Indices of files_or_folders belonging to each split
    idx = np.arange(n)
    train_idx = np.random.choice(idx, train_n, replace=False)
    idx = np.setdiff1d(idx, train_idx)  # Remove files or folders used for training
    val_idx = np.random.choice(idx, val_n, replace=False)
    idx = np.setdiff1d(idx, val_idx)  # Remove files or folders used for validation
    test_idx = np.random.choice(idx, test_n, replace=False)

    # Write train.txt
    write_to_txt(idx=train_idx, items=(noartifact_files, gibbs_files), path_save=path_root, filename="train.txt")
    write_to_txt(idx=val_idx, items=(noartifact_files, gibbs_files), path_save=path_root, filename="val.txt")
    write_to_txt(idx=test_idx, items=(noartifact_files, gibbs_files), path_save=path_root, filename="test.txt")


if __name__ == "__main__":
    path_root = Path(r"D:\Sravan\Data\Source\202312-T1w_MtSinai_YasminHurd\select")
    train_split = 0.85
    val_split = 0.10

    main(
        path_root=path_root,
        train_split=train_split,
        val_split=val_split,
    )
