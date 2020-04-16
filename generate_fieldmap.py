import numpy as np
import cv2

# Convolve with a smoothing filter
def gen_smooth_b0(im= np.array, f_offset= float):
    hist = np.histogram(im)
    mask = cv2.threshold(im, hist[1][hist[0].argmax()], 1, cv2.THRESH_BINARY)

    # im_norm = np.zeros(im.shape)
    # im_norm = cv2.normalize(im, im_norm, 0, 1, cv2.NORM_MINMAX)
    # kernel = np.ones((4, 4)) / 16
    M = cv2.GaussianBlur(im, (75, 75), 0)

    M_orc_range = np.zeros(im.shape)
    M_orc_range = cv2.normalize(M, M_orc_range, -f_offset, f_offset, cv2.NORM_MINMAX)

    M2 = M_orc_range * mask[1]

    # Bin the field map every 5 Hz
    bins = np.arange(-f_offset, f_offset, 5)
    for x in range(M.shape[0]):
        for y in range(M.shape[1]):
            idx = find_nearest(bins, M2[x, y])
            M2[x, y] = bins[idx]

    return M2, mask[1]

# Nylund's thesis methods
# 1. Uniform field map
def gen_uniform_b0(f_offset= float, N= int):
    M = np.ones((N, N))
    return M * f_offset

# 2. Linear field map
def gen_linear_b0(f_offset= float, N= int):
    y_axis = np.linspace(0.5, 1, N)
    M = np.tile(y_axis, (N, 1)).T
    return M * f_offset

# 3. Random field map type 1
def gen_random1_b0(f_offset= float, N=int):
    M = np.random.rand(8, 8)
    kernel = np.ones((3, 3))/9
    M2 = cv2.filter2D(M, -1, kernel) # LPF
    M3 = cv2.resize(M2, (N, N))
    M4 = np.zeros((N, N))
    M4 = cv2.normalize(M3, M4, -f_offset, f_offset, cv2.NORM_MINMAX).round(3)
    return M4.round(2)

# 4. Random field map type 2
def gen_random2_b0(f_offset= float, N=int):
    M = np.random.rand(2, 2)
    M2 = cv2.resize(M, (N,N))
    M3 = np.zeros((N, N))
    M3 = cv2.normalize(M2, M3, -f_offset, f_offset, cv2.NORM_MINMAX).round(3)
    bins = np.arange(-f_offset, f_offset, 5)
    for x in range(M2.shape[0]):
        for y in range(M2.shape[1]):
            idx = find_nearest(bins, M3[x, y])
            M3[x, y] = bins[idx]
    return M3

def find_nearest(array,value):
    '''Finds the index of the value's closest array element

    Parameters
    ----------
    array : numpy.ndarray
        Array of values
    value : float
        Value for which the closest element has to be found

    Returns
    -------
    idx : int
        Index of the closest element of the array
    '''
    array = np.asarray(array)
    diff = array - value
    if value >= 0:
        idx = np.abs(diff).argmin()
    else:
        idx = np.abs(diff[array < 1]).argmin()

    return idx
'''
plt.subplot(1, 3, 1)
im1 = plt.imshow(field_map1)
plt.axis('off')
plt.title('Smoothed + masked image', fontsize=10)
plt.colorbar(im1, fraction=0.046, pad=0.04)

plt.subplot(1, 3, 2)
field_map2 = generate_fieldmap.gen_random1_b0(fmax, N)
im2 = plt.imshow(field_map2)
plt.axis('off')
plt.title('Random field map 1',fontsize=10)
plt.colorbar(im2, fraction=0.046, pad=0.04)

plt.subplot(1, 3, 3)
field_map3 = generate_fieldmap.gen_random2_b0(fmax, N)
im3 = plt.imshow(field_map3)
plt.axis('off')
plt.title('Random field map 2', fontsize=10)
plt.colorbar(im3, fraction=0.046, pad=0.04)


plt.show()'''