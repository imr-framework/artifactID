import itertools
import math

import numpy as np


class KerasDataGenerator:
    def __init__(self, x, y, val_pc: float, eval_pc: float, batch_size: int, seed: int = 0):
        np.random.seed(seed=seed)
        self.batch_size = batch_size

        # Shuffle
        x, y = self._shuffle_dataset(x=x, y=y)
        self.x = x
        self.y = y

        # Construct dictionary to encode labels as integers
        unique_labels = np.unique(y)
        self.dict_labels_encoded = dict(zip(unique_labels, itertools.count(0)))

        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None

        # Eval
        self.eval_pc = eval_pc
        self.eval_num = int(len(self.x) * self.eval_pc)
        eval_random_idx = np.random.randint(len(self.x), size=self.eval_num)
        self.eval_x = np.take(self.x, eval_random_idx)
        self.eval_y = np.take(self.y, eval_random_idx)

        # Remove test samples from the main dataset
        self.x = np.delete(self.x, eval_random_idx)
        self.y = np.delete(self.y, eval_random_idx)

        # Cross-validation
        self.val_pc = val_pc
        self.val_num = int(len(self.x) * self.val_pc)
        self.cross_val_counter = 0
        self.val_start = 0
        self.val_end = 0
        self.update_slices_cross_validation()

        # Steps per epoch
        self.train_steps_per_epoch = math.ceil(len(self.train_x) / self.batch_size)
        self.val_steps_per_epoch = math.ceil(self.val_num / self.batch_size)
        self.eval_steps_per_epoch = math.ceil(self.eval_num / self.batch_size)

    def _shuffle_dataset(self, x, y):
        assert len(x) == len(y)
        random_idx = np.random.randint(len(x), size=len(x))
        x = np.take(x, random_idx)
        y = np.take(y, random_idx)
        return x, y

    def _flow(self, x, y):
        while True:
            for counter in range(len(x)):
                _x = np.load(x[counter])  # Load volume
                _x = np.expand_dims(_x, axis=3)  # Convert shape to (240, 240, 155, 1)
                _x = _x.astype(np.float16)  # Mixed precision

                label = y[counter]  # Get label
                _y = np.array([self.dict_labels_encoded[label]]).astype(np.int8)  # Encoded label

                yield _x, _y

    def train_flow(self):
        return self._flow(x=self.train_x, y=self.train_y)

    def val_flow(self):
        return self._flow(x=self.val_x, y=self.val_y)

    def eval_flow(self):
        return self._flow(x=self.eval_x, y=self.eval_y)

    def update_slices_cross_validation(self):
        self.val_start = self.val_end
        self.val_end = self.val_start + self.val_num
        val_idx = np.arange(self.val_start, self.val_end)
        if self.val_end > len(self.x):
            out_of_bounds_idx = np.where(val_idx >= len(self.x))[0]
            val_idx[out_of_bounds_idx] = val_idx[out_of_bounds_idx] - len(self.x)
            self.val_end = val_idx[-1]

        self.val_x = np.take(self.x, val_idx)
        self.val_y = np.take(self.y, val_idx)
        self.val_x, self.val_y = self._shuffle_dataset(x=self.val_x, y=self.val_y)  # Shuffle

        self.train_x = np.delete(self.x, val_idx)
        self.train_y = np.delete(self.y, val_idx)
        self.train_x, self.train_y = self._shuffle_dataset(x=self.train_x, y=self.train_y)  # Shuffle
