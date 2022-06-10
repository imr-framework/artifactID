from pathlib import Path

import numpy as np
import tensorflow.keras
from tensorflow.keras.models import load_model


class ModelWrapper(object):
    def __init__(self, path_model: Path):
        self.__check_model(path_model)
        self._model = load_model(str(path_model))

    def __check_model(self, path_model: Path):
        if not path_model.exists():
            raise FileNotFoundError(f'{path_model} not found.')

    def get_keras_model(self) -> tensorflow.keras.Model:
        return self._model

    def predict(self, x: np.ndarray):
        return np.argmax(self._model.predict(x))

    def predict_batch(self, x_batch: np.ndarray):
        x_batch = np.expand_dims(x_batch, axis=-1)
        y_pred = self._model.predict(x_batch)
        y_pred = np.argmax(y_pred, axis=-1)

        return y_pred
