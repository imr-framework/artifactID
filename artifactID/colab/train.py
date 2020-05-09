import itertools
import math
from pathlib import Path

import numpy as np


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
    y_labels.extend([str(artifact_folder.name)] * len(files))

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


# Make generator for x and y training pairs
def train_generator(batch_size):
    global x_paths_train
    global y_labels_train
    while True:
        counter = 0
        x = []
        y = []
        for i in range(batch_size):
            vol = np.load(x_paths_train[counter])  # Load volume
            x.append(vol)
            label = y_labels_train[counter]  # Get label
            y.append(dict_labels_encoded[label])  # Add encoded label
            counter += 1
            if counter > len(x_paths):  # This means one/another epoch is done
                # Now, reset counter and reshuffle the training dataset
                counter = 0
                x_paths_train, y_labels_train = _shuffle(x=x_paths_train, y=y_labels_train)
        x = np.stack(x)  # Convert to numpy.ndarray
        x = np.expand_dims(x, axis=4).astype(np.float16)  # Convert shape to (batch_size, 240, 240, 155, 1)
        y = np.array(y).astype(np.int8)
        yield x, y


# Design network
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, Dense, Flatten, MaxPool3D

# Prevent crash
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

model = Sequential()
model.add(Conv3D(filters=16, kernel_size=3, input_shape=(240, 240, 155, 1), activation='relu'))
model.add(MaxPool3D())
model.add(Conv3D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPool3D())
model.add(Flatten())
model.add(Dense(units=len(dict_labels_encoded), activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

batch_size = 1
model.fit(x=train_generator(batch_size=batch_size), steps_per_epoch=math.ceil(len(x_paths_train) / batch_size))
save_path = 'artifactID_model.hdf5'
model.save(save_path)
