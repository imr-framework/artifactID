from pathlib import Path

import numpy as np
import pydicom as pyd

from artifactID.common import data_ops

path_read_godwin_dicom = Path(r"")
path_save_godwin_npy = Path(r"")

folders = path_read_godwin_dicom.glob('*')
for f in folders:
    files = f.glob('*')
    for ff in files:
        # Read DICOM and convert to numpy.array
        d = pyd.dcmread(str(ff))
        npy = d.pixel_array

        # Resize to 256x256, extract brain, normalize to [0, 1]
        npy_resized = data_ops.resize(npy, size=256)
        npy_resized = np.expand_dims(npy_resized, axis=2)
        npy_brain_extracted = data_ops.__extract_brain(npy_resized)
        npy_normalized = data_ops.normalize_slices(npy_resized)
        npy_normalized = npy_normalized.squeeze()

        # Save to disk
        _path_save = ff.relative_to(path_read_godwin_dicom)
        _path_save = path_save_godwin_npy / _path_save
        _path_save = _path_save.with_suffix('.npy')
        if not _path_save.parent.exists():
            _path_save.parent.mkdir(parents=True)
        np.save(arr=npy_normalized, file=_path_save)
