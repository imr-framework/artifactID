from pathlib import Path

import nibabel as nb

from common_utils import data_loader

path_hurd = Path(r"D:\Sravan\Data\Source\202312-T1w_MtSinai_YasminHurd\1009902394\dicom")
hurd = data_loader.load_data(path_data=path_hurd, data_format="dicom", normalize=True, target_size=256)
nii = nb.Nifti2Image(hurd, affine=None)
nb.save(nii, str(path_hurd.parent / "hurd.nii.gz"))
