import tensorflow as tf
from tensorflow import keras


class CustomModel(keras.Model):
    def __init__(self, inputs, outputs, gamma: float):
        super(CustomModel, self).__init__(inputs, outputs)

        self.loss_metric = keras.metrics.Mean(name="loss")
        self.val_loss_metric = keras.metrics.Mean(name="val_loss")
        self.accuracy_metric = keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        self.val_accuracy_metric = keras.metrics.SparseCategoricalAccuracy(
            name="val_accuracy"
        )
        self.gamma = gamma

    def train_step(self, data):
        inputs, targets = data
        inputs, masks = inputs[:, 0], inputs[:, 1]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)
            predictions = self(inputs, training=True)

            # SCCE
            loss1 = tf.keras.losses.sparse_categorical_crossentropy(targets, predictions, from_logits=True)

            prob_predictions = keras.activations.softmax(predictions)
            right_reasons_inner = tf.math.reduce_sum(tf.math.log(prob_predictions), axis=-1)
            right_reasons = tape.gradient(right_reasons_inner, inputs)
            loss2 = tf.math.reduce_sum(tf.math.square(masks * right_reasons), axis=(1, 2))
            loss2 = tf.cast(loss2, tf.float32)

            loss = loss1 + self.gamma * loss2

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_metric.update_state(loss)
        self.accuracy_metric.update_state(y_true=targets, y_pred=predictions)
        return {
            "loss": self.loss_metric.result(),
            "accuracy": self.accuracy_metric.result(),
        }

    def test_step(self, data):
        inputs, targets = data
        inputs, masks = inputs[:, 0], inputs[:, 1]

        predictions = self(inputs, training=False)

        # SCCE
        loss = tf.keras.losses.sparse_categorical_crossentropy(targets, predictions, from_logits=True)

        self.val_loss_metric.update_state(loss)
        self.val_accuracy_metric.update_state(y_true=targets, y_pred=predictions)
        return {
            "loss": self.val_loss_metric.result(),
            "accuracy": self.val_accuracy_metric.result(),
        }

    @property
    def metrics(self):
        return [self.loss_metric, self.accuracy_metric, self.val_loss_metric, self.val_accuracy_metric]
