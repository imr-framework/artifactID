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

# =========
# TENSORFLOW INIT
# =========
# Prevent crash
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

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
    "dense_3"]

"""
## The Grad-CAM algorithm
"""


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
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
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    # if np.max(heatmap) <= 0:  # To deal with 0 heatmaps
    #     warnings.warn(RuntimeWarning(f'Anomaly in class {top_pred_index.numpy()}'))
    #     heatmap = np.abs(heatmap)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


def main(arr_npy, artifact: str, path_model: str):
    if artifact in ['wrap', 'motion']:
        last_conv_layer_name = "conv2d_2"
    elif artifact == 'gibbs':
        last_conv_layer_name = "re_lu_2"

    # Make model
    model = keras.models.load_model(str(path_model))

    arr_heatmaps, arr_ypred = [], []
    for npy in arr_npy:
        # Predict
        npy = np.expand_dims(npy, axis=(0, 3))
        y_pred = model.predict(npy)
        y_pred = np.argmax(y_pred, axis=-1)
        arr_ypred.append(y_pred[0])

        # Generate class activation heatmap
        heatmap = make_gradcam_heatmap(npy,
                                       model,
                                       last_conv_layer_name,
                                       classifier_layer_names)

        if heatmap is not None:
            # Rescale heatmap to range 0-255
            heatmap = cv2.resize(heatmap, np.squeeze(npy).shape)
        arr_heatmaps.append(heatmap)

    return np.array(arr_heatmaps), np.array(arr_ypred)
