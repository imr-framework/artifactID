from pathlib import Path


def glob_brats_t1(path_brats: str):
    path_brats = Path(path_brats)
    arr_paths_brats_t1 = list(path_brats.glob('**/*.nii.gz'))
    arr_paths_brats_t1 = list(filter(lambda x: 't1.nii' in str(x), arr_paths_brats_t1))
    return arr_paths_brats_t1
