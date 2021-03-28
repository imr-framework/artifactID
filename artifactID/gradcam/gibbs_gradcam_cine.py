from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import animation

from artifactID.gradcam import gradcam

# =========
# TENSORFLOW INIT
# =========
# Prevent crash
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

last_conv_layer_name = "conv2d_2"
classifier_layer_names = ["flatten",
                          "dense",
                          "dense_1",
                          "dense_2",
                          "dense_3"]


def main(path_ffmpeg: Path, path_model: Path, path_save_movie: Path, path_txt: Path):
    # =========
    # SETUP
    # =========
    arr_paths_gibbs_noartifact = path_txt.read_text().splitlines()
    arr_paths_gibbs_noartifact.pop(0)

    arr_paths_gibbs = []
    arr_paths_noartifact = []
    for p in arr_paths_gibbs_noartifact:
        p = Path(p.strip())
        if p.parent.name == 'gibbs':
            arr_paths_gibbs.append(str(p))
        elif p.parent.name == 'noartifact':
            arr_paths_noartifact.append(str(p))

    # num_files is the smallest length across arr_paths_gibbs and arr_paths_noartifact
    num_files = min(len(arr_paths_gibbs), len(arr_paths_noartifact))
    arr_paths_gibbs = np.random.choice(arr_paths_gibbs, size=num_files, replace=False)
    arr_paths_noartifact = np.random.choice(arr_paths_noartifact, size=num_files, replace=False)

    arr_gibbs_kspace = []
    arr_noartifact_kspace = []
    for i in range(num_files):
        _path_gibbs = arr_paths_gibbs[i]
        _path_noartifact = arr_paths_noartifact[i]

        npy_gibbs = np.load(str(_path_gibbs))
        kspace_gibbs = np.fft.fftshift(np.fft.fftn(npy_gibbs))
        kspace_gibbs = np.expand_dims(kspace_gibbs, axis=(0, 3))
        arr_gibbs_kspace.append(kspace_gibbs)

        npy_noartifact = np.load(str(_path_noartifact))
        kspace_noartifact = np.fft.fftshift(np.fft.fftn(npy_noartifact))
        kspace_noartifact = np.expand_dims(kspace_noartifact, axis=(0, 3))
        arr_noartifact_kspace.append(kspace_noartifact)

    # =========
    # GRAD-CAM
    # =========
    # Perform Grad-CAM
    arr_gibbs_heatmaps, arr_ypred_gibbs = gradcam.main(arr_npy=np.abs(arr_gibbs_kspace).astype(np.float16),
                                                       path_model=path_model)
    arr_noartifacts_heatmaps, arr_ypred_noartifact = gradcam.main(
        arr_npy=np.abs(arr_noartifact_kspace).astype(np.float16),
        path_model=path_model)

    # Filter for correct results only
    arr_gibbs_heatmaps = arr_gibbs_heatmaps[arr_ypred_gibbs == 1]
    arr_noartifacts_heatmaps = arr_noartifacts_heatmaps[arr_ypred_noartifact == 0]
    num_files = min(len(arr_gibbs_heatmaps), len(arr_noartifacts_heatmaps))

    THRESHOLD = 0.95  # Threshold for masking heatmaps

    def animate(i):
        if i < num_files:
            heatmap_gibbs = arr_gibbs_heatmaps[i]
            heatmap_noartifact = arr_noartifacts_heatmaps[i]

            if heatmap_gibbs is not None and heatmap_noartifact is not None:
                # Row 1 - Gibbs
                t_heatmap_gibbs = heatmap_gibbs > THRESHOLD
                _kspace_gibbs = arr_gibbs_kspace[i]
                _kspace_gibbs_t = _kspace_gibbs.squeeze() * t_heatmap_gibbs
                _img_gibbs = np.abs(np.fft.ifftshift(np.fft.ifftn(_kspace_gibbs_t)))

                arr_img[0].set_array(np.abs(_kspace_gibbs).squeeze())  # K-space of input
                arr_img[0].set_clim(vmin=0, vmax=10)
                arr_img[1].set_array(heatmap_gibbs)  # Grad-CAM of input k-space
                arr_img[1].autoscale()
                arr_img[2].set_array(t_heatmap_gibbs)  # Thresholded input k-space as per Grad-CAM
                arr_img[2].autoscale()
                arr_img[3].set_array(np.abs(_img_gibbs))  # Image of thresholded k-space
                arr_img[3].autoscale()

                # Row 2 - no artifact
                t_heatmap_noartifact = heatmap_noartifact > THRESHOLD
                _kspace_noartifact = arr_noartifact_kspace[i]
                _kspace_noartifact_t = _kspace_noartifact.squeeze() * t_heatmap_noartifact
                _img_noartifact = np.abs(np.fft.ifftshift(np.fft.ifftn(_kspace_noartifact_t)))

                arr_img[4].set_array(np.abs(_kspace_noartifact).squeeze())  # K-space of input
                arr_img[4].set_clim(vmin=0, vmax=10)
                arr_img[5].set_array(np.abs(heatmap_noartifact))  # Grad-CAM of input k-space
                arr_img[5].autoscale()
                arr_img[6].set_array(np.abs(t_heatmap_noartifact))  # Thresholded input k-space as per Grad-CAM
                arr_img[6].autoscale()
                arr_img[7].set_array(np.abs(_img_noartifact))  # Image of thresholded k-space
                arr_img[7].autoscale()
            else:
                if heatmap_gibbs is None:
                    print(arr_paths_gibbs[i])
                else:
                    print(arr_paths_noartifact[i])

    # =========
    # PLOTTING
    # =========
    fig, arr_ax = plt.subplots(nrows=2, ncols=4)
    arr_ax = arr_ax.ravel()
    arr_img = []
    for index, ax in enumerate(arr_ax):
        if index == 0:
            ax.title.set_text('Gibbs')
        elif index == 4:
            ax.title.set_text('No artifact')
        cmap = 'gray' if index not in [1, 5] else 'jet'
        arr_img.append(ax.imshow(np.ones((1, 1)), cmap=cmap))
        ax.axis('off')
    fig.tight_layout()

    # =========
    # ANIMATION
    # =========
    # Make animation and save
    anim = animation.FuncAnimation(fig, animate, frames=num_files, interval=250, repeat=False)
    plt.rcParams['animation.ffmpeg_path'] = str(path_ffmpeg)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(str(path_save_movie), writer=writer)
    plt.show()


if __name__ == '__main__':
    path_model = Path("")
    path_txt = Path(r"")
    path_ffmpeg = Path(r"")
    path_save_movie = Path(r"")

    main(path_ffmpeg=path_ffmpeg, path_model=path_model, path_save_movie=path_save_movie, path_txt=path_txt)
