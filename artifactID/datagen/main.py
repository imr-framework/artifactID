import configparser
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

path_search = str(Path(__file__).parent.parent)  # To allow ORC to be discoverable
sys.path.insert(0, path_search)
# from artifactID.datagen import fov_wrap_datagen, offres_datagen, noartifact_datagen, rigidmotion_datagen, snr_datagen, \
#     gibbs_datagen, nonrigidmotion_datagen, b0cartesian_datagen
from artifactID.datagen import noartifact_datagen_godwin, gibbs_datagen_godwin, gibbs_datagen

# Read settings.ini configuration file
path_settings = '../settings.ini'
config = configparser.ConfigParser()
config.read(path_settings)
config_data = config['DATAGEN']
slice_size = int(config_data['slice_size'])
path_read_data = Path(config_data['path_read_data'])
path_save_data = Path(config_data['path_save_data'])
path_ktraj = config_data['path_ktraj']
path_dcf = config_data['path_dcf']
test_split = float(config_data['test_split'])
validation_split = float(config_data['validation_split'])

# No-artifact datagen
# print('No-artifact datagen...')
# noartifact_datagen.main(path_read_data=path_read_data, path_save_data=path_save_data, slice_size=slice_size)
# noartifact_datagen_godwin.main(path_read_data=path_read_data, path_save_data=path_save_data, slice_size=slice_size)

# Ghosting datagen
# print('FOV wrap-around datagen...')
# fov_wrap_datagen.main(path_read_data=path_read_data, path_save_data=path_save_data, slice_size=slice_size)

# Off-resonance datagen
# print('\nOff-resonance datagen...')
# offres_datagen.main(path_read_data=path_read_data, path_save_data=path_save_data, path_ktraj=path_ktraj,
#                     path_dcf=path_dcf, patch_size=patch_size)

# SNR datagen
# print('\nSNR datagen...')
# snr_datagen.main(path_read_data=path_read_data, path_save_data=path_save_data, slice_size=slice_size)

# Rotation datagen
# print('\nRigid rotation datagen')
# rigidmotion_datagen.main(path_read_data=path_read_data, path_save_data=path_save_data, patch_size=patch_size)

# Gibbs datagen
# print('\nGibbs datagen...')
gibbs_datagen.main(path_read_data=path_read_data, path_save_data=path_save_data, slice_size=slice_size)
# gibbs_datagen_godwin.main(arr_path_read=path_read_data, path_save_data=path_save_data, slice_size=slice_size)

# Non-rigid motion datagen
# print('\nNon-rigid motion datagen...')
# nonrigidmotion_datagen.main(path_read_data=path_read_data, path_save_data=path_save_data, patch_size=patch_size)

# B0 inhomogeneity Cartesian
# print('\nB0 inhomogeneity Cartesian datagen...')
# b0cartesian_datagen.main(path_read_data=path_read_data, path_save_data=path_save_data, patch_size=patch_size)

# Create splits
# _path_noartifact_folder = path_save_data / 'noartifact'
# path_noartifacts = list(_path_noartifact_folder.glob('*.npy'))
# path_train_noartifact, path_test_noartifact = train_test_split(path_noartifacts,
#                                                                test_size=test_split,
#                                                                shuffle=True)
# path_train_noartifact, path_val_noartifact = train_test_split(path_noartifacts,
#                                                               test_size=validation_split,
#                                                               shuffle=True)
#
# list_folders = list(path_save_data.glob('*'))  # List folders
# list_folders = list(filter(lambda folder: folder.is_dir(), list_folders))  # Remove non-folder items
# list_folders.remove(path_save_data / 'noartifact')  # Remove 'noartifact' folder
#
# path_all = []
# for p_noartifact in [path_train_noartifact,
#                      path_val_noartifact,
#                      path_test_noartifact]:
#     _temp = []
#     for p in p_noartifact:
#         _file_to_check = p.name
#         for f in list_folders:
#             if (f / _file_to_check).exists():
#                 _temp.append(p)
#                 _temp.append(f / _file_to_check)
#     path_all.append(_temp)
#
# path_train, path_val, path_test = path_all
#
# # Write to disk
# for _filename, _path in (('train.txt', path_train),
#                          ('val.txt', path_val),
#                          ('test.txt', path_test)):
#     _path_save = Path(path_save_data) / _filename
#     with open(str(_path_save), 'w') as f:
#         f.write(str(len(_path)))
#         f.write('\n')
#         for x in _path:
#             f.write(str(x))
#             f.write('\n')
