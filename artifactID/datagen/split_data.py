from pathlib import Path

import numpy as np


def make_data_splits(path_root: Path, artifact: str, train_pc=85, val_pc=10):
    def _write_to_txt(idx, filename):
        text = [str(len(idx))]
        for i in idx:
            text.append(str(Path('noartifact') / files[i].name))
            text.append(str(Path(artifact) / files[i].name))
        text = '\n'.join(text)
        (path_root / filename).write_text(text)

    train_split = train_pc / 100
    val_split = val_pc / 100

    # Determine number of files
    artifact_folder = path_root / artifact
    files = list(artifact_folder.glob('*.npy'))

    # Splits
    n = len(files)
    train_n = int(train_split * n)
    val_n = int(val_split * n)
    test_n = n - train_n - val_n

    # Indices
    idx = np.arange(n)
    train_idx = np.random.choice(idx, train_n, replace=False)
    idx = np.delete(idx, train_idx)
    val_idx = np.random.choice(idx, val_n, replace=False)
    idx = np.delete(idx, val_idx)
    test_idx = np.random.choice(idx, test_n, replace=False)

    # Write train.txt
    _write_to_txt(idx=train_idx, filename='train.txt')
    _write_to_txt(idx=val_idx, filename='val.txt')
    _write_to_txt(idx=test_idx, filename='test.txt')


if __name__ == '__main__':
    path_root = Path(r"D:\CU Data\Datagen\artifactID_GE\20211028\e470\finetune")
    artifact = 'motion'
    train_pc = 85
    val_pc = 10

    make_data_splits(path_root=path_root, train_pc=train_pc, val_pc=val_pc, artifact=artifact)
