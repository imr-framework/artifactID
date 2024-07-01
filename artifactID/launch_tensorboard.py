import os
import sys
from pathlib import Path


def main(path_logs: str):
    log_dir = Path("./models") / path_logs / "logs"
    os.system(f"{sys.executable} -m tensorboard.main --logdir {str(log_dir)}")


if __name__ == '__main__':
    folder = "20220902_1515_gamma1_finalpick"
    main(folder)
