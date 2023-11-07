import numpy as np
import numpy.fft as fft
from tqdm import tqdm


def conversion_arcseconds_to_Mm(distance):
    """Convert arcseconds on the Sun as viewed from Earth
    to Megameters.
    Arguments:
        distance -- Distance between the Sun and Earth [km].

    Returns:
        Factor to convert arcseconds as seen on Earth to Mm [Mm].
    """
    # 1 arsecond ~= 0.713 Mm
    distance_between_Sun_and_Earth = distance  # 147.1e6 km
    convert_arcsecond_to_rad = (1 / 3600) * (np.pi / 180)  # rad
    conversion_calculation = (
        distance_between_Sun_and_Earth * convert_arcsecond_to_rad
    )  # km
    convert_arcseconds_to_Mm = (conversion_calculation * 1000) * 10 ** (
        -6
    )  # Mm # factor required to convert from arcseconds to Mm on the Sun as
    # viewed from Earth
    return convert_arcseconds_to_Mm


def cross_spectrum_2D(time_series1, time_series2):
    """Computes the complex cross-spectrum of two times series
    at different heights. If the two inputs are the
    same, a power spectrum is computed.
    
    Arguments:
        time_series1 -- Time series that forms lower in the atmosphere
        time_series2 = Time series that forms higher in the atmosphere

    Returns:
        A complex cross-spectrum of the two series is computed. Same size as the inputs.
    """
    # Compute N-Dimension FFT of time series
    fft_1 = fft.fft2(time_series1)
    fft_2 = fft.fft2(time_series2)

    # Compute cross spectrum
    # and shift cube to align frequencies
    cross_spectrum = fft.fftshift(fft_1 * np.conjugate(fft_2))

    return cross_spectrum


def power_spectrum_2D(time_series):
    """Computes the 2D power spectrum.

    Arguments:
        time_series -- Time series array [x,t].

   Returns:
        Power spectrum
    """
    power = np.abs(fft.fftshift(fft.fft2(time_series))) ** 2
    return power


def cross_spectrum_1D(time_series1, time_series2):
    """Computes the complex cross-spectrum of two times series
    at different heights. If the two inputs are the
    same, a power spectrum is computed.

    Input:
    time_series1 = Time series that forms lower in the atmosphere
    time_series2 = Time series that forms higher in the atmosphere

    Output:
    cross_spectrum = A complex cross-spectrum of the two series is computed.
    Same size as the inputs
    """

    # Compute N-Dimension FFT of time series
    fft_1 = fft.fft(time_series1, axis=-1)
    fft_2 = fft.fft(time_series2, axis=-1)

    # Compute cross spectrum
    # and shift cube to align frequencies
    cross_spectrum = fft.fftshift(fft_1 * np.conjugate(fft_2), axes=(-1))

    return cross_spectrum


def power_spectrum_1D(time_series):
    """ Computes the 1D power spectrum.
	
	Input:
	time_series = time series array [x,t]
	
	Output:
	power = power spectrum
    """
    power = np.abs(fft.fftshift(fft.fft(time_series, axis=-1), axes=(-1))) ** 2
    return power


def compute_height(height0, height1, zarray):
    """Given two height inputs, finds the position in the height array that best matches.

    Arguments:
        height0 -- Height 1 [Mm].
        height1 -- Height 2 [Mm].
        zarray -- Spatial z array [Mm].

    Returns:
        The index positions of the two given heights [Mm]._
    """
    indx_z0 = np.where(zarray >= height0)[0][0]
    # zheight0 = zarray[indx_z0]

    indx_z1 = np.where(zarray >= height1)[0][0]
    # height1 = zarray[indx_z1]
    return indx_z0, indx_z1


def compute_phase_distance(pix_x, pix_t, time_series1, time_series2):
    """ Compute the phase difference as a measure of distance from a fixed position of one time series.

    Arguments:
        pix_x -- Number of pixel/grid points in x-domain.
        pix_t -- Number of pixel/grid points for the time-domain.
        time_series1 -- Time series that forms lower in the atmosphere
        time_series2 -- Time series that forms higher in the atmosphere

    Return:
        Phase difference as a measure of distance [rad].
    """

    dphase_distance = np.zeros([pix_x, pix_t])

    for xi in tqdm(range(0, pix_x)):
        fixed_x = 0
        cross_spec = cross_spectrum_1D(time_series1[fixed_x, :], time_series2[xi, :])
        phase_spec = np.angle(cross_spec, deg=False)
        dphase_distance[xi, :] = phase_spec

        return dphase_distance
