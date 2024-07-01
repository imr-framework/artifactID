from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.python.keras import Model


def main(path_txt: Path, path_model: Path):
    # Load model
    print(f'Loading {str(path_model)}')
    model = load_model(path_model)

    # =========
    # LOAD DATA
    # =========
    files = path_txt.read_text().splitlines()[1:3]
    x = []
    for f in files:
        npy = np.load(str(f))
        x.append(npy)

    x = np.stack(x, axis=-1)
    x = np.moveaxis(x, 2, 0)
    x = np.expand_dims(x, -1)

    # =========
    # OBTAIN ACTIVATIONS
    # =========
    layers = [l.output for l in model.layers if "conv" in l.name]
    activations_model = Model(inputs=model.input, outputs=layers)
    activations = activations_model.predict(x)

    # =========
    # VIZ
    # =========
    img = []
    for i in range(17):
        row = []
        for j in range(17):
            if i != 16 and j != 16:
                a = activations[2][0, ..., ((i + 1) * (j + 1))]
                row.append(a)
        if len(row) != 0:
            img.append(np.hstack(row))
    img = np.vstack(img)
    plt.figure(dpi=300)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig('test-IXI.png')
    # plt.show()


if __name__ == '__main__':
    path_txt = Path(r"E:\Data\Datagen\ArtifactID\IXI-Gibbs_sag-BET\train.txt")
    path_model = Path(r"../models/20220712_1215/model_20220712_1225")
    main(path_txt, path_model)
