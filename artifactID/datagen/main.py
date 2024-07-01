from pathlib import Path

from artifactID.datagen import gibbs_datagen_old as gibbs_datagen


def main(path_read_data: Path, path_save_data: Path, slice_size: int, nifti_dataset: str):
    # Gibbs datagen (includes no artifact)
    print("\nGibbs datagen...")
    gibbs_datagen.main(
        path_read_data=path_read_data,
        path_save_data=path_save_data,
        slice_size=slice_size,
        nifti_dataset=nifti_dataset,
        # save=False,
        # debug_viz=False
    )


if __name__ == "__main__":
    path_read_data = Path(r"D:\Sravan\Data\Source\IXI-T1")
    path_save_data = Path(r"D:\Sravan\Data\Datagen\ArtifactID_v3") / (path_read_data.name)
    slice_size = 256
    nifti_dataset = "IXI-T1"  # path_read_data.name
    main(
        path_read_data=path_read_data,
        path_save_data=path_save_data,
        slice_size=slice_size,
        nifti_dataset=nifti_dataset
    )
