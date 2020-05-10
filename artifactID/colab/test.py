import itertools
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm


def _shuffle(x, y):
    assert len(x) == len(y)
    random_idx = np.random.randint(len(x), size=len(x))
    x = np.take(x, random_idx)
    y = np.take(y, random_idx)
    return x, y


# Construct `x` and `y` training pairs
x_paths = []
y_labels = []
data_root = Path(r"C:\Users\sravan953\Documents\CU\Projects\imr-framework\ArtifactID\Data\data")
for artifact_folder in data_root.glob('*'):
    files = list(artifact_folder.glob('*.npy'))
    x_paths.extend(files)
    label = artifact_folder.name.rstrip('0123456789')
    y_labels.extend([label] * len(files))

# Shuffle
x_paths, y_labels = _shuffle(x=x_paths, y=y_labels)

# Split into train, validation, test
train_num = int(len(x_paths) * 0.75)
val_num = int(train_num + (len(x_paths) * 0.20))
eval_num = int(val_num + (len(x_paths) * 0.05))
x_paths_train = x_paths[:train_num]
y_labels_train = y_labels[:train_num]
x_paths_val = x_paths[train_num:val_num]
y_labels_val = y_labels[train_num:val_num]
x_paths_eval = x_paths[val_num:eval_num]
y_labels_eval = y_labels[val_num:eval_num]
assert len(x_paths_train) + len(x_paths_val) + len(x_paths_eval) == len(x_paths)

# Construct dictionary of labels and their integer mappings
unique_labels = np.unique(y_labels)
dict_labels_encoded = dict(zip(unique_labels, itertools.count(0)))

# Load eval data
x_eval = []
for p in tqdm(x_paths_eval):
    x_eval.append(np.load(p))
x_eval = np.stack(x_eval).astype(np.float16)
x_eval = np.expand_dims(x_eval, axis=-1)
# y_eval = [dict_labels_encoded[y] for y in y_labels_eval]
# y_eval = np.expand_dims(y_eval, axis=-1)

# Prevent crash
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

print('Loading model...')
model = load_model('artifactID_model.hdf5')
print('Evaluating model...')
y_pred = []
for eval in tqdm(x_eval):
    eval = np.expand_dims(eval, axis=0)
    y_pred.append(np.argmax(model.predict(x=eval)))
print(y_pred)
print(dict_labels_encoded)
