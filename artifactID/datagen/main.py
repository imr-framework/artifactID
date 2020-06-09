import configparser
import sys
from pathlib import Path

path_search = str(Path(__file__).parent.parent)  # To allow ORC to be discoverable
sys.path.insert(0, path_search)
# from artifactID.datagen import fov_wrap_datagen, offres_datagen, snr_datagen
from artifactID.datagen import fov_wrap_datagen, snr_datagen, noartifact_datagen

# Read settings.ini configuration file
path_settings = '../settings.ini'
config = configparser.ConfigParser()
config.read(path_settings)
config_data = config['DATAGEN']
patch_size = int(config_data['patch_size'])
path_read_data = config_data['path_read_data']
path_save_data = config_data['path_save_data']
path_ktraj = config_data['path_ktraj']
path_dcf = config_data['path_dcf']

# No-artifact datagen
# print('No-artifact datagen...')
# noartifact_datagen.main(path_read_data=path_read_data, path_save_data=path_save_data, patch_size=patch_size)

# Ghosting datagen
# print('FOV wrap-around datagen...')
# fov_wrap_datagen.main(path_read_data=path_read_data, path_save_data=path_save_data, patch_size=patch_size)

# Off-resonance datagen
# print('\nOff-resonance datagen...')
# offres_datagen.main(path_brats=path_read_brats, path_save=path_save_datagen, path_ktraj=path_ktraj, path_dcf=path_dcf)

# SNR datagen
# print('\nSNR datagen...')
# snr_datagen.main(path_read_data=path_read_data, path_save_data=path_save_data, patch_size=patch_size)
