import math

import numpy as np

from classes.slice_class import SliceObj


class SNRObj(SliceObj):
    def __init__(self, arr, obj_mask=None, noise_mask=None):
        super().__init__(arr)
        if obj_mask is None or noise_mask is None:
            self._make_masks()
        else:
            self.obj_mask = obj_mask
            self.noise_mask = noise_mask

    def _make_masks(self):
        obj_idx = np.nonzero(self.data)

        obj_mask = np.zeros_like(self.data)
        obj_mask[obj_idx] = 1
        self.obj_mask = obj_mask.astype(np.int8)

        noise_mask = np.ones_like(self.data)
        noise_mask[obj_idx] = 0
        self.noise_mask = noise_mask.astype(np.int8)

    def add_real_noise(self):
        data = self.data + np.random.normal(loc=0, scale=0.001, size=self.data.shape)
        return SNRObj(data, self.obj_mask, self.noise_mask)

    def add_awgn(self, target_snr_db: float = None, awgn_std: float = None):
        if target_snr_db is None and awgn_std is None:
            raise ValueError('Either target_snr_db or awgn_std have to be passed.')
        elif target_snr_db is not None and awgn_std is not None:
            raise ValueError('Either target_snr_db or awgn_std have to be passed, not both.')

        if target_snr_db:
            object = np.extract(arr=self.data, condition=self.obj_mask)
            awgn_std = object.mean() / math.pow(10, target_snr_db / 20)
        noise = np.random.normal(loc=0, scale=awgn_std, size=int(self.noise_mask.sum())).astype(np.float16)
        data_noisy = np.copy(self.data)
        np.place(arr=data_noisy, mask=self.noise_mask, vals=noise)
        return SNRObj(data_noisy, self.obj_mask, self.noise_mask)

    def get_snr(self):
        object = np.extract(arr=self.data, condition=self.obj_mask)
        noise = np.extract(arr=self.data, condition=self.noise_mask)
        return 20 * np.log10(object.mean() / noise.std())
