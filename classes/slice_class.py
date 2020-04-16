import numpy as np


class SliceObj:
    def __init__(self, arr):
        self.data = arr.astype(np.float16)
