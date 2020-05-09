from pathlib import Path

import numpy as np
from tqdm import tqdm

data_root = Path(r"C:\Users\sravan953\Documents\CU\Projects\imr-framework\ArtifactID\Data\data")
all_max = dict()
all_min = dict()

folders = list(data_root.glob('*'))
for folder in tqdm(folders):
    files = list(folder.glob('*.npy'))
    for f in files:
        vol = np.load(f)
        key = f.parts[-2] + '/' + f.parts[-1]
        all_max[key] = vol.max()
        all_min[key] = vol.min()

print()
