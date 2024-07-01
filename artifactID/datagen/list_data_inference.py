from pathlib import Path

import numpy as np


def main(path_root: Path):
    """
    Performs subject-wise data-split for HCP.

    Parameters
    ----------
    """
    path_noartifact = path_root / "noartifact"
    files = list(path_noartifact.glob('**/*.npy'))  # List of all files across all folders in `items`
    np.random.shuffle(files)  # Shuffle
    text = []
    for f in files:
        text.append(str(f))  # No artifact
        f = str(f).replace("noartifact", "gibbs")
        text.append(str(f))  # Gibbs3
    text.insert(0, str(len(text)))
    text = "\n".join(text)
    (path_root / 'inference.txt').write_text(text)


if __name__ == "__main__":
    path_root = Path(r"D:\Sravan\Data\Datagen\ArtifactID_v1\LAC-T1")

    main(
        path_root=path_root,
    )
