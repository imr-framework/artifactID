import configparser
import sys
from pathlib import Path

path_search = str(Path(__file__).parent.parent)
sys.path.insert(0, path_search)
# from artifactID.datagen import snr_datagen
from artifactID.datagen import offres_datagen

# Read settings.ini configuration file
path_settings = r"C:\Users\sravan953\Documents\CU\Projects\imr-framework\ArtifactID\Code\artifactID\settings.ini"
config = configparser.ConfigParser()
config.read(path_settings)
config_data = config['DATA']
path_brats = config_data['path_brats']
path_save = config_data['path_save']
path_ktraj = config_data['path_ktraj']
path_dcf = config_data['path_dcf']

if __name__ == '__main__':  # Wrapper to enable multiprocessing for off-resonance datagen
    # Off-resonance datagen
    print('Off-resonance datagen...')
    offres_datagen.main(path_brats=path_brats, path_save=path_save, path_ktraj=path_ktraj, path_dcf=path_dcf)

    # SNR datagen
    # print('\nSNR datagen...')
    # snr_datagen.main(path_brats=path_brats, path_save=path_save)
