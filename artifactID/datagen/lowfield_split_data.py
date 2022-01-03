from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

path_datagen = Path(r"")
path_lowfield = Path(r"")
path_save_txt = r""

# Gather files
files_gibbs = []
files_noartifact = []
# for p in [path_lowfield]: # To train no-simulation model
for p in [path_datagen, path_lowfield]:  # To train simulation-included model
    folders = p.glob('*')
    for f in folders:
        files = list(f.glob('*.npy'))
        if 'gibbs' in f.name.lower():
            files_gibbs.extend(files)
        elif 'noartifact' in f.name.lower():
            files_noartifact.extend(files)

# Splits
train_test_pc = 0.2
train_val_pc = 0.1

path_train_gibbs, path_test_gibbs = train_test_split(files_gibbs, test_size=train_test_pc)
path_train_gibbs, gibbs_val = train_test_split(path_train_gibbs, test_size=train_val_pc)

path_train_noartifact, path_test_noartifact = train_test_split(files_noartifact, test_size=train_test_pc)
path_train_noartifact, noartifact_val = train_test_split(path_train_noartifact, test_size=train_val_pc)

path_train = []
path_train.extend(path_train_gibbs)
path_train.extend(path_train_noartifact)
np.random.shuffle(path_train)

path_val = []
path_val.extend(gibbs_val)
path_val.extend(noartifact_val)
np.random.shuffle(path_val)

path_test = []
path_test.extend(path_test_gibbs)
path_test.extend(path_test_noartifact)
np.random.shuffle(path_test)

# Write to disk
for _filename, _arr in (('train.txt', path_train),
                        ('val.txt', path_val),
                        ('test.txt', path_test)):
    _path_save = Path(path_save_txt) / _filename
    with open(str(_path_save), 'w') as f:
        f.write(str(len(_arr)))
        f.write('\n')
        for x in _arr:
            f.write(str(x))
            f.write('\n')
