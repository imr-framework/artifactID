"""
Title: Grad-CAM class activation visualization
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/04/26
Last modified: 2020/05/14
Description: How to obtain a class activation heatmap for an image classification model.
Adapted from Deep Learning with Python (2017).
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.data import Dataset

"""
## Configurable parameters

You can change these to another model.

To get the values for `last_conv_layer_name` and `classifier_layer_names`, use
 `model.summary()` to see the names of all layers in the model.
"""
classifier_layer_names = [
    "flatten",
    "dense",
    "dense_1",
    "dense_2",
    "dense_3",
    "dense_4",
]
last_conv_layer_name = "conv2d_2"

"""
## The Grad-CAM algorithm
"""


def make_dataset_from_paths(
        paths: list, input_size: int, batch_size: int = 32
) -> Dataset:
    def generator_paths(files):
        for i in range(len(files)):
            p = files[i].decode()
            x = np.load(p.strip())
            yield x,

    dataset = tf.data.Dataset.from_generator(
        generator=generator_paths,
        args=(paths,),
        output_types=(tf.float64,),
        output_shapes=(tf.TensorShape((input_size, input_size)),),
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def make_dataset_from_npy(npy: np.ndarray, batch_size=32) -> Dataset:
    dataset = Dataset.from_tensor_slices(npy)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def make_gradcam_heatmap(
        path_model, last_conv_layer_name, classifier_layer_names, dataset_test
):
    model = keras.models.load_model(path_model)

    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    all_heatmaps = []
    all_ypreds = []
    for batch in dataset_test:
        with tf.GradientTape() as tape:
            # Compute activations of the last conv layer and make the tape watch it
            last_conv_layer_output = last_conv_layer_model(batch)
            tape.watch(last_conv_layer_output)
            # Compute class predictions
            preds = classifier_model(last_conv_layer_output)

            if preds.shape[1] == 1:  # Sigmoid
                top_pred_index = tf.cast(preds > 0.5, tf.int32)
                top_class_channel = preds
            elif preds.shape[1] > 1:  # Softmax
                top_pred_index = tf.cast(tf.argmax(preds, axis=1), tf.int32)
                top_class_channel = tf.math.reduce_max(preds, axis=1)

        # This is the gradient of the top predicted class with regard to
        # the output feature map of the last conv layer
        grads = tape.gradient(top_class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(-3, -2)).numpy()
        pooled_grads = np.expand_dims(pooled_grads, axis=(1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        last_conv_layer_output *= pooled_grads

        # The channel-wise sum of the resulting feature map
        # is our heatmap of class activation
        heatmap = np.mean(last_conv_layer_output, axis=-1)

        # heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        all_heatmaps.append(heatmap)
        all_ypreds.append(top_pred_index)
    all_heatmaps = np.concatenate(all_heatmaps)
    all_ypreds = np.concatenate(all_ypreds)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    m1 = np.expand_dims(np.min(all_heatmaps, axis=(1, 2)), axis=(1, 2))
    all_heatmaps -= np.min(m1)
    m2 = np.expand_dims(np.max(all_heatmaps, axis=(1, 2)), axis=(1, 2)) + np.finfo(float).eps
    all_heatmaps /= m2
    return all_heatmaps.astype(np.float), all_ypreds


def main(path_model: str, files: list = None, dataset_test: tf.data.Dataset = None, heatmap_target_shape: int = 256):
    if dataset_test is None:
        dataset_test = make_dataset_from_paths(paths=files, input_size=256, batch_size=64)

    # Generate class activation heatmap and obtain predictions
    all_heatmaps, all_ypreds = make_gradcam_heatmap(
        path_model, last_conv_layer_name, classifier_layer_names, dataset_test
    )

    # Resize heatmaps
    all_heatmaps = [
        cv2.resize(heatmap, (heatmap_target_shape, heatmap_target_shape))
        for heatmap in all_heatmaps
    ]

    return all_heatmaps, all_ypreds
