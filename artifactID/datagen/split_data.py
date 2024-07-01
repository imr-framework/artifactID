from pathlib import Path

import numpy as np


def write_folder_to_txt(idx: np.array, items: list, path_save: Path, filename: str):
    files = []  # List of all files across all folders in `items`

    # Subject-wise split
    for i in idx:  # Iterate over chosen folders and list all files
        f = items[i]
        files.extend(list(f.glob("*.npy")))

    np.random.shuffle(files)  # Shuffle
    text = [str(len(files))]
    for i in range(len(files)):
        text.append(str(files[i]))
        noisy_file = str(files[i]).replace("noartifact", "gibbs")
        text.append(noisy_file)
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

    # Determine number of folders/files depending on HCP
    noartifact_folder = path_root / "noartifact"
    subjects = list(noartifact_folder.glob("*"))

    # Splits
    n = len(subjects)  # Dataset size
    # n = n // 10  # Prototyping, 10pc dataset
    # path_root = path_root / "mini"  # Prototyping, 10pc dataset
    train_n = 5  # int(train_split * n)  # Train split
    val_n = 2  # int(val_split * n)  # Validation split
    test_n = 1 # n - train_n - val_n  # Test split

    # Indices of files_or_folders belonging to each split
    idx = np.arange(n)
    train_idx = np.random.choice(idx, train_n, replace=False)
    idx = np.setdiff1d(idx, train_idx)  # Remove files or folders used for training
    val_idx = np.random.choice(idx, val_n, replace=False)
    idx = np.setdiff1d(idx, val_idx)  # Remove files or folders used for validation
    test_idx = np.random.choice(idx, test_n, replace=False)

    # Write train.txt
    write_folder_to_txt(idx=train_idx, items=subjects, path_save=path_root, filename="train_med.txt")
    write_folder_to_txt(idx=val_idx, items=subjects, path_save=path_root, filename="val_med.txt")
    write_folder_to_txt(idx=test_idx, items=subjects, path_save=path_root, filename="test_med.txt")


if __name__ == "__main__":
    path_root = Path(r"D:\Sravan\Data\Datagen\ArtifactID_v2\IXI-T1")
    train_split = 0.85
    val_split = 0.10

    main(
        path_root=path_root,
        train_split=train_split,
        val_split=val_split,
    )
