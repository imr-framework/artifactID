from pathlib import Path

from artifactID.eval import eval


def batch(batch_size: int, input_size: int, path_model: Path, datasets: list, save_tptnfpfn: bool,
          save_cm: bool = False, show_cm: bool = False):
    for ds in datasets:
        print(f"Dataset: {ds}")

        path_txt = path_root / ds / "inference.txt"
        eval.main(batch_size=batch_size,
                  input_size=input_size,
                  path_model=path_model,
                  path_txt=path_txt,
                  save_tptnfpfn=save_tptnfpfn,
                  save_cm=save_cm,
                  show_cm=show_cm)
        print("=========")
        print("\n")


if __name__ == "__main__":
    batch_size = 64
    input_size = 256
    path_root = Path(r"D:\Sravan\Data\Datagen\ArtifactID")
    datasets = [
        # "IXI-T1",
        "ADNI-T1",
        "HCP-T1",
        # "LAC-T1",
        "LAC-T1_resampled",
        "SUDMEX-T1",
        "SRPBS-T1"
    ]
    path_model = Path(r"../models/20220902_1515_gamma1_finalpick/best_model.100")
    # path_model = Path(r"models/20240226_1727/model.500")

    # Perform batch eval
    batch(
        batch_size=batch_size,
        input_size=input_size,
        path_model=path_model,
        datasets=datasets,
        save_tptnfpfn=False,
        save_cm=True,
        show_cm=False
    )
