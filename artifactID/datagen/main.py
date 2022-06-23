import configparser
import sys
from pathlib import Path

path_search = str(Path(__file__).parent.parent)  # To allow ORC to be discoverable
sys.path.insert(0, path_search)
from artifactID.datagen import (
    gibbs_datagen,
    noartifact_datagen,
    fov_wrap_datagen_xy,
    fov_wrap_datagen_z,
)

# =========
# READ SETTINGS.INI
# =========
# Read settings.ini configuration file
path_settings = "../settings.ini"
config = configparser.ConfigParser()
config.read(path_settings)
config_data = config["DATAGEN"]
slice_size = int(config_data["slice_size"])
path_read_data = Path(config_data["path_read_data"])
path_save_data = Path(config_data["path_save_data"])

# =========
# DATAGEN
# =========
# No-artifact datagen
# print('No-artifact datagen...')
# noartifact_datagen.main(path_read_data=path_read_data, path_save_data=path_save_data, slice_size=slice_size)

# FOV wrap-around datagen
# print('FOV wrap-around datagen (horizontal and vertical)...')
# fov_wrap_datagen_xy.main(path_read_data=path_read_data, path_save_data=path_save_data, slice_size=slice_size)
# print('FOV wrap-around datagen (slice direction)...')
# fov_wrap_datagen_z.main(path_read_data=path_read_data, path_save_data=path_save_data, slice_size=slice_size)

# Gibbs datagen
print("\nGibbs datagen...")
gibbs_datagen.main(
    path_read_data=path_read_data, path_save_data=path_save_data, slice_size=slice_size
)
