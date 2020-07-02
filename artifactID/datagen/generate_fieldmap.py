import cv2
import numpy as np

from OCTOPUS.Recon import ORC

def _bin_five(field_map: np.ndarray, freq_offset: float):
    bins = np.arange(-freq_offset, freq_offset, 5)
    for x in range(field_map.shape[0]):
        for y in range(field_map.shape[1]):
            idx = find_nearest(bins, field_map[x, y])
            field_map[x, y] = bins[idx]

    return field_map


# Convolve with a smoothing filter
def gen_smooth_b0(im: np.array, freq_offset: float):
    hist = np.histogram(im)
    mask = cv2.threshold(im, hist[1][hist[0].argmax()], 1, cv2.THRESH_BINARY)
    m = cv2.GaussianBlur(im, (75, 75), 0)
    m = cv2.normalize(m, m, -freq_offset, freq_offset, cv2.NORM_MINMAX)
    m = m * mask[1]
    # Bin the field map every 5 Hz
    m = _bin_five(field_map=m, freq_offset=freq_offset)

    return m, mask[1]


# Nylund's thesis methods
# 1. Uniform field map
def gen_uniform_b0(freq_offset: float, n: int):
    m = np.ones((n, n))
    return m * freq_offset


# 2. Linear field map
def gen_linear_b0(freq_offset: float, n: int):
    y_axis = np.linspace(0.5, 1, n)
    m = np.tile(y_axis, (n, 1)).T
    return m * freq_offset


# 3. Random field map type 1
def gen_random1_b0(freq_offset: float, n: int):
    m = np.random.random_sample((8, 8))
    kernel = np.ones((3, 3)) / 9
    m = cv2.filter2D(m, -1, kernel)  # LPF
    m = cv2.resize(m, (n, n))
    m = cv2.normalize(m, m, alpha=-freq_offset, beta=freq_offset, norm_type=cv2.NORM_MINMAX).round(3)
    return m.round(2)


# 4. Random field map type 2
def gen_random2_b0(freq_offset: float, n: int):
    m = np.random.rand(2, 2)
    m = cv2.resize(m, (n, n))
    m = np.zeros((n, n))
    m = cv2.normalize(m, m, -freq_offset, freq_offset, cv2.NORM_MINMAX).round(3)
    m = _bin_five(field_map=m, freq_offset=freq_offset)
    return m


def find_nearest(array, value):
    """
    Finds the index of the value's closest array element

    Parameters
    ==========
    array : numpy.ndarray
        Array of values
    value : float
        Value for which the closest element has to be found

    Returns
    =======
    idx : int
        Index of the closest element of the array
    """
    array = np.asarray(array)
    diff = array - value
    if value >= 0:
        idx = np.abs(diff).argmin()
    else:
        idx = np.abs(diff[array < 1]).argmin()

    return idx

def parabola_formula(N: int):
    """
    Parabola values to fit an image of N rows/columns

    Parameters
    ----------
    N :  int
        Matrix size

    Returns
    -------
    yaxis : numpy.ndarray
        y axis values of the parabola
    """
    x1, y1 = -N / 10, 0.5
    x2, y2 = 0, 0
    x3, y3 = N / 10, 0.5

    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    B = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
    C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom

    y = lambda x: A * x ** 2 + B * x + C
    yaxis = y(np.arange(-N / 2, N / 2, 1)).reshape(1, N)

    return yaxis
def fieldmap_bin(field_map: np.ndarray, bin : int):
    '''
    Bins a given field map given a binning value

    Parameters
    ----------
    field_map : numpy.ndarray
        Field map matrix in Hz
    bin : int
        Binning value in Hz

    Returns
    -------
    binned_field_map : numpy.ndarray
        Binned field map matrix
    '''
    fmax = field_map.max()
    bins = np.arange(-fmax, fmax + bin, bin)
    binned_field_map = np.zeros(field_map.shape)
    for x in range(field_map.shape[0]):
        for y in range(field_map.shape[1]):
            idx = ORC.find_nearest(bins, field_map[x, y])
            binned_field_map[x, y] = bins[idx]

    return binned_field_map
def hyperbolic(N: int, fmax : float, bin_opt : bool = True,  bin_val : int = 5):
    """
    Creates a hyperbolic field map

    Parameters
    ----------
    N : int
       Field map dimensions (NxN)
    fmax : float
       Frequency range in Hz
    bin_opt : bool
        Binning option. Default is True
    bin_val : int
        Binning value, default is 5 Hz


    Returns
    -------
    field map : numpy.ndarray
       Field map matrix with dimensions [NxN] and scaled from -fmax to +fmax Hz
    """
    y = parabola_formula(N)
    rows = np.tile(y, (N, 1))
    field_map_mat = rows - rows.T
    dst = np.zeros(field_map_mat.shape)
    field_map = cv2.normalize(field_map_mat, dst, -fmax, fmax, cv2.NORM_MINMAX)
    if bin_opt:
        field_map = fieldmap_bin(field_map, bin_val)
    return field_map

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
