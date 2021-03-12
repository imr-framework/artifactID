import configparser
import sys
from pathlib import Path

path_search = str(Path(__file__).parent.parent)  # To allow ORC to be discoverable
sys.path.insert(0, path_search)
from artifactID.datagen import fov_wrap_datagen, fov_wrap_datagen_z, gibbs_datagen, noartifact_datagen

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
print('No-artifact datagen...')
noartifact_datagen.main(path_read_data=path_read_data, path_save_data=path_save_data, slice_size=slice_size)

# FOV wrap-around datagen
print('FOV wrap-around datagen (horizontal and vertical)...')
fov_wrap_datagen.main(path_read_data=path_read_data, path_save_data=path_save_data, slice_size=slice_size)
print('FOV wrap-around datagen (slice direction)...')
fov_wrap_datagen_z.main(path_read_data=path_read_data, path_save_data=path_save_data, slice_size=slice_size)

# Gibbs datagen
print('\nGibbs datagen...')
gibbs_datagen.main(path_read_data=path_read_data, path_save_data=path_save_data, slice_size=slice_size)

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
