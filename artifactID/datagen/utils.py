from pathlib import Path


def glob_nifti(path: Path) -> list:
    arr_path = list(path.glob('**/*.nii.gz'))
    arr_path2 = list(path.glob('**/*.nii'))
    files = arr_path + arr_path2
    # files = list(filter(lambda x: "HH" in x.name, files))
    return files
