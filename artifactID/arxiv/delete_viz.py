import nibabel as nb
import numpy as np

import sass

path_vol = r"D:\MMJ\Data\LAC-T1\sub-000_ses-20110101_anat_sub-000_ses-20110101_T1w.nii.gz"
path_vol = r"D:\MMJ\Data\SRPBS-T1\sub-0005\t1\defaced_mprage.nii"

vol = nb.load(path_vol).get_data()
vol = np.moveaxis(vol, 0, -1)
print(vol.shape)
sass.scroll(vol)
