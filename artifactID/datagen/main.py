import configparser
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

path_search = str(Path(__file__).parent.parent)  # To allow ORC to be discoverable
sys.path.insert(0, path_search)
from artifactID.datagen import fov_wrap_datagen, offres_datagen, noartifact_datagen, rigidmotion_datagen, snr_datagen, \
    gibbs_datagen, nonrigidmotion_datagen, b0cartesian_datagen

# Read settings.ini configuration file
path_settings = '../settings.ini'
config = configparser.ConfigParser()
config.read(path_settings)
config_data = config['DATAGEN']
patch_size = int(config_data['patch_size'])
path_read_data = Path(config_data['path_read_data'])
path_save_data = Path(config_data['path_save_data'])
path_ktraj = config_data['path_ktraj']
path_dcf = config_data['path_dcf']
test_split = float(config_data['test_split'])
validation_split = float(config_data['validation_split'])

# No-artifact datagen
print('No-artifact datagen...')
noartifact_datagen.main(path_read_data=path_read_data, path_save_data=path_save_data, patch_size=patch_size)

# Ghosting datagen
print('FOV wrap-around datagen...')
fov_wrap_datagen.main(path_read_data=path_read_data, path_save_data=path_save_data, patch_size=patch_size)

# Off-resonance datagen
print('\nOff-resonance datagen...')
offres_datagen.main(path_read_data=path_read_data, path_save_data=path_save_data, path_ktraj=path_ktraj,
                    path_dcf=path_dcf, patch_size=patch_size)

# SNR datagen
print('\nSNR datagen...')
snr_datagen.main(path_read_data=path_read_data, path_save_data=path_save_data, patch_size=patch_size)

# Rotation datagen
print('\nRigid rotation datagen')
rigidmotion_datagen.main(path_read_data=path_read_data, path_save_data=path_save_data, patch_size=patch_size)

# Gibbs datagen
print('\nGibbs datagen...')
gibbs_datagen.main(path_read_data=path_read_data, path_save_data=path_save_data, patch_size=patch_size)

# Non-rigid motion datagen
print('\nNon-rigid motion datagen...')
nonrigidmotion_datagen.main(path_read_data=path_read_data, path_save_data=path_save_data, patch_size=patch_size)

# B0 inhomogeneity Cartesian
print('\nB0 inhomogeneity Cartesian datagen...')
b0cartesian_datagen.main(path_read_data=path_read_data, path_save_data=path_save_data, patch_size=patch_size)

# Create splits
path_all = list(Path(path_save_data).glob('**/*.npy'))
path_train, path_test = train_test_split(path_all, test_size=test_split, shuffle=True)
path_train, path_val = train_test_split(path_train, test_size=validation_split, shuffle=True)

# Write to disk
for _filename, _path in (('train.txt', path_train),
                         ('test.txt', path_test),
                         ('val.txt', path_val)):
    _path_save = Path(path_save_data) / _filename
    with open(str(_path_save), 'w') as f:
        f.write(str(len(_path)))
        f.write('\n')
        for x in _path:
            f.write(str(x))
            f.write('\n')
