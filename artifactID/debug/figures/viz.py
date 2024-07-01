import nibabel as nb
import numpy as np

import sass

path1 = r"E:\Data\Source\HCP-T1-BET\1.nii.gz"
nii1 = nb.load(path1).get_data()
# nii1 = np.moveaxis(nii1, (0, 1, 2), (2, 1, 0))
# nii1 = np.flipud(nii1)
nii1 = np.rot90(nii1, -1, axes=(0, 1))
nii1 = np.fliplr(nii1)
# nii1 = np.rot90(nii1, 2, axes=(0, 2))
# nii1 = np.rot90(nii1, -1, axes=(0, 1))
# nii1 = np.fliplr(nii1)



print(nii1.shape)

sass.scroll(nii1)
