import configparser
import sys
from pathlib import Path

path_search = str(Path(__file__).parent.parent)  # To allow ORC to be discoverable
sys.path.insert(0, path_search)
from artifactID.datagen import fov_wrap_datagen, offres_datagen, snr_datagen

# Read settings.ini configuration file
path_settings = r"C:\Users\sravan953\Documents\CU\Projects\imr-framework\ArtifactID\Code\artifactID\settings.ini"
config = configparser.ConfigParser()
config.read(path_settings)
config_data = config['DATA']
path_read_brats = config_data['path_read_brats']
path_datagen_save = config_data['path_datagen_save']
path_ktraj = config_data['path_ktraj']
path_dcf = config_data['path_dcf']

# Ghosting datagen
print('FOV wrap-around datagen...')
fov_wrap_datagen.main(path_brats=path_read_brats, path_save=path_datagen_save)

# Off-resonance datagen
print('\nOff-resonance datagen...')
offres_datagen.main(path_brats=path_brats, path_save=path_save, path_ktraj=path_ktraj, path_dcf=path_dcf)

# SNR datagen
print('\nSNR datagen...')
snr_datagen.main(path_brats=path_read_brats, path_save=path_datagen_save)
