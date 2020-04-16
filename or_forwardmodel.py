import numpy as np
import scipy.io as sio
import nibabel as nib
import matplotlib.pyplot as plt

import os
import time
import cv2
import math
import random

import generate_fieldmap
import ORC

def preprocess_imvol(imgvol):
    '''
    Preprocesses (Rotates and normalizes) a 3D image volume

    Parameters
    ----------
    imvol : numpy.ndarray
        Image volume having dimensions N x N x Number of Slices

    Returns
    -------
    imvol_nonzero : numpy.ndarray
        Image volume after preprocessing N x N x Number of Slices intensity range [0, 1]
    '''
    imgvol = np.rot90(imgvol, -1, axes=(0, 1))  # Rotation
    imgvol_pp = (imgvol - imgvol.min()) / (imgvol.max() - imgvol.min())  # Normalization [0, 1]

    return imgvol_pp

def orc_forwardmodel(imgvol, freq_range, traj, seq_params):
    '''
    Adds off-resonance blurring to simulate B0 inhomogeneity artifacts.

    Parameters
    ----------
    imgvol : numpy.ndarray
        Image volume having dimensions N x N x Number of Slices
    freq_range : int
        Frequency range for the simulated field map
    traj : numpy.ndarray
        k-space trajectroy coordinates. Dimensions Npoints x Nshots
    seq_params : dict
        Sequence parameters needed for off-resonance corruption

    Returns
    -------
    or_imgvol : list
        Corrupted image volume having dimensions Slices(with signal > 5%) x N x N
    '''
    seq_params['N'] = imgvol.shape[0]
    Nslices = imgvol.shape[2]
    or_imgvol = []
    for sl in range(Nslices):
        slice = imgvol[:,:,sl]
        if np.count_nonzero(slice) > 0.05 * slice.size:  # Check if at least 5% of signal is present
            field_map, mask = generate_fieldmap.gen_smooth_b0(slice, freq_range) # Simulate the field map
            or_corrupted, _ = ORC.add_or_CPR(slice, traj, field_map, nonCart=1, params=seq_params) # Corrupt the image
            or_corrupted_norm = np.zeros(or_corrupted.shape)
            or_corrupted_norm = cv2.normalize(np.abs(or_corrupted), or_corrupted_norm, 0, 1, cv2.NORM_MINMAX) # Normalize [0, 1]
            or_imgvol.append(np.float32(or_corrupted_norm * mask))
        else:
            pass

    return or_imgvol
##
# Load some necessary files
##
# 1. k-space trajectory
ktraj = np.load('ktraj.npy')
ktraj_sc = math.pi / abs(np.max(ktraj))
ktraj = ktraj * ktraj_sc # pyNUFFT scaling [-pi, pi]

# 2. Density compensation factor
dcf = sio.loadmat('dcf.mat')['dcf_out'].flatten()

# 3. Acquisition parameters
T = (np.arange(ktraj.shape[0])*10e-6).reshape(ktraj.shape[0],1)
seq_params = {'Npoints': ktraj.shape[0], 'Nshots': ktraj.shape[1], 't_vector': T, 'dcf': dcf}

##
# Data path and folder paths
##
data_path = os.path.abspath('../Data/MICCAI_BraTS_2018_Data_Training')
data_list = os.listdir(data_path)
count = 0
for folder_name in data_list:
    T1_file = os.path.join(data_path, folder_name, folder_name + '_t1.nii.gz')
    img_vol = nib.load(T1_file).get_fdata()
    # Pre-processing
    img_vol_pp = preprocess_imvol(img_vol)
    # Forward-Model
    fmax = [250, 500, 750] # Hz
    subjects_per_class = int(len(data_list) / len(fmax))
    f_list = [fmax[0]]* subjects_per_class + [fmax[1]] * subjects_per_class + [fmax[2]] * subjects_per_class
    random.shuffle(f_list)

    img_vol_or = orc_forwardmodel(img_vol_pp, f_list[count], ktraj, seq_params)

    arr_brats_b0 = np.stack(img_vol_or)
    np.moveaxis(arr_brats_b0, [0, 1, 2], [1, 2, 0])  # Iterate through slices on the last dim
    np.save(os.path.join(data_path, folder_name, folder_name + '_' + str(f_list[count]) + '_B0inhomogeneity.npy', arr_brats_b0))
    print('Corrupted data saved for subject:' + folder_name)

    count =+ 1






