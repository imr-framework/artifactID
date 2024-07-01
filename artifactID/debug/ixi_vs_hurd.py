from pathlib import Path

import numpy as np

import sass
from common_utils import data_loader, metrics

# Read IXI vol
path_ixi_vol = Path(r"D:\Sravan\Data\Datagen\ArtifactID\IXI-T1\gibbs\2")
ixi = data_loader.load_data(path_data=path_ixi_vol, data_format="npy", normalize=True, target_size=256)
ixi += np.random.normal(loc=0, scale=5e-3, size=ixi.shape)

ixi_local_SNR_map = metrics.get_local_SNR_map(ixi, mask_brain=True)

# Read Yasmin Hurd volume
path_hurd_vol = Path(r"D:\Sravan\Data\Source\202312-T1w_MtSinai_YasminHurd\1009902394\dicom")
hurd = data_loader.load_data(path_data=path_hurd_vol, data_format="dicom", normalize=True, target_size=256)
hurd_local_SNR_map = metrics.get_local_SNR_map(hurd, mask_brain=True)
num_slices = ixi_local_SNR_map.shape[-1]
center = hurd_local_SNR_map.shape[-1] // 2
hurd_local_SNR_map = hurd_local_SNR_map[..., center - num_slices // 2: center + num_slices // 2]

print(f"IXI: {ixi_local_SNR_map.mean()}, Hurd: {hurd_local_SNR_map.mean()}")

sass.scroll(ixi_local_SNR_map, hurd_local_SNR_map, cmap=['jet', 'jet'])
