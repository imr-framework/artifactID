"""
Ablation study - threshold input based on Grad-CAM heatmap and obtain new Grad-CAM heatmap.
"""
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from artifactID.gradcam import gradcam

# =========
# TENSORFLOW INIT
# =========
# Prevent crash
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)


def main(path_npy: str, path_model: str):
    # No need for data_ops.load_preprocess_npy because we are loading from a clean dataset anyway
    npy = np.load(path_npy).squeeze()
    npy = cv2.resize(npy, (256, 256))  # Resize
    kspace = np.fft.fftshift(np.fft.fftn(npy))
    kspace = np.expand_dims(kspace, axis=(0, 3))

    # =========
    # GRAD-CAM
    # =========
    # Perform Grad-CAM
    heatmap, y_pred = gradcam.main(arr_npy=np.abs([kspace]).astype(np.float16), path_model=path_model)
    heatmap = heatmap.squeeze()
    print(f'Pred: {y_pred}')

    THRESHOLD = 0.5  # Threshold for masking heatmaps
    heatmap_t = heatmap <= THRESHOLD
    kspace_t = kspace.squeeze() * heatmap_t
    kspace_t = np.expand_dims(kspace_t, axis=(0, 3))
    heatmap2, y_pred2 = gradcam.main(arr_npy=np.abs([kspace_t]).astype(np.float16), path_model=path_model)
    heatmap2 = heatmap2.squeeze()
    print(f'Pred2: {y_pred2}')
    npy_t = np.abs(np.fft.ifftn(np.fft.ifftshift(kspace_t)))

    # =========
    # PLOTTING
    # =========
    plt.subplot(231)
    plt.imshow(npy.astype(np.float32), cmap='gray')  # Input image
    plt.axis('off')

    plt.subplot(232)
    plt.imshow(np.abs(kspace).squeeze(), cmap='gray', vmax=10)  # K-space of input image
    plt.axis('off')

    plt.subplot(233)
    plt.imshow(heatmap, cmap='jet')  # Grad-CAM of input k-space
    plt.axis('off')

    plt.subplot(235)
    plt.imshow(np.abs(kspace_t).squeeze(), cmap='gray', vmax=10)  # K-space of input image
    plt.axis('off')

    plt.subplot(236)
    plt.imshow(heatmap2, cmap='jet')  # Grad-CAM of input k-space
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    path_model = r"20210301_2022_Gibbs_sim\model.hdf5"
    path_npy = r"gibbs\IMG-0003-00001.npy"
    main(path_npy, path_model)
