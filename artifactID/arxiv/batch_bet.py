from pathlib import Path

from common_utils import data_loader


def main(path_data: Path, path_save: Path):
    nifti_files = data_loader.glob_nifti(path_data)

    txt = []
    commands = ["bet", "", ""]
    for counter, nii in enumerate(nifti_files):
        # Make save path
        _path_save = path_save / f"{counter + 1}"
        if not _path_save.parent.exists():
            _path_save.parent.mkdir(parents=True, exist_ok=True)

        # Convert to POSIX representations and fix paths
        nii = str(nii.as_posix())
        nii_anchor = nii[0].lower()
        _path_save = str(_path_save.as_posix())
        _path_save_anchor = _path_save[0].lower()

        commands[1] = f'"/mnt/{nii_anchor}/{nii[3:]}"'
        commands[2] = f'"/mnt/{_path_save_anchor}/{_path_save[3:]}"'

        txt.append((" ").join(commands))

    # Write commands to batch_bet.sh
    txt = ("\n").join(txt)
    txt += "\n"
    path_txt = Path(r"batch_bet.sh")
    path_txt.write_text(txt)

    # path_txt.read_text()

    print("Batch BET script is located at -")
    print(Path("batch_bet.sh").absolute().as_posix())


if __name__ == "__main__":
    path_data = Path(r"E:\Data\Source\ADNI_GO2-T1-FLIRT")
    path_save = Path(r"E:\Data\Source\ADNI_GO2-T1-FLIRT-BET")
    main(path_data, path_save)
