import cmasher as cmr
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tqdm import tqdm

import build_functions
import spectral_analysis


def plot_wave_packet(
    array, tdim, xgrid, zgrid, cmapc
):  # , min_z, max_z, min_x, max_x):
    """
	Plots gravity wavepacket(s).

	Arguments:
		array -- The wave field [array of x, z, t].
		tdim -- Time dimension grid [s].
		xgrid -- Spatial x dimension grid [Mm].
		zgrid -- Spatial z dimension grid [Mm].
        cmapc -- Chosen cmap.
		max_x_arr -- Max value of the spatial x grid [Mm].
		max_z_arr -- Max value of the spatial z grid [Mm].


	Returns:
		_description_
	"""
    plt.ion()
    fig, axs = plt.subplots(1, 1, figsize=(16, 6))
    # minv = array[:, :, tdim].min()
    # maxv = array[:, :, tdim].max()
    im1 = axs.pcolormesh(
        xgrid,
        zgrid,
        array[:, :, tdim],
        cmap=cmapc,
        # norm=mcolors.TwoSlopeNorm(vmin=minv, vcenter=0, vmax=maxv),
    )
    plt.colorbar(im1, ax=axs, orientation="horizontal")

    axs.set(xlabel="x [Mm]", ylabel="z [Mm]")

    # zoom = 0.5
    # w, h = fig.get_size_inches()
    # fig.set_size_inches(w * zoom, h * zoom)
    #axs.axes.set_aspect(1.8)
    axs.axes.set_aspect('equal')

    # plt.axis('scaled')
    # xs.set_aspect("scaled")

    # axs.set_ylim(min_z, max_z)
    # axs.set_xlim(min_x, max_x)

    myLocator = ticker.MultipleLocator(0.5)
    axs.yaxis.set_major_locator(myLocator)
    plt.tight_layout()
    plt.show()
    return fig


def plot_animation(time_array, nstep, array, xgrid, zgrid, h1, h2, cmapc):
    """
	Plots animated gravity wavepacket(s).

	Arguments:
		time_array -- The time array [s].
        nstep -- For loop does every nstep iteration.
		array -- The wave field [array of x, z, t].
		xgrid --  Spatial x dimension grid [Mm].
		zgrid --  Spatial z dimension grid [Mm].
        cmapc -- Chosen cmap.
		h1 -- Height 1 [Mm].
		h2 -- Height 2 [Mm].

	Returns:
		_description_
	"""
    plt.ion()
    fig, axs = plt.subplots(1, 1, figsize=[16, 6])

    # axs.axhline(y=h2, color='k', linestyle='--')
    # cbaxes = fig.add_axes([0.05, 0.1, 0.8, 0.04])
    for t in tqdm(range(0, len(time_array), nstep)):
        axs.cla()
        minval = array.min()
        maxval = array.max()
        im1 = axs.pcolormesh(
            xgrid,
            zgrid,
            array[:, :, t],
            cmap=cmapc,
            norm=mcolors.TwoSlopeNorm(vmin=minval, vcenter=0, vmax=maxval),
        )
        plt.axhline(y=0, color="k", linestyle="--")
        plt.axhline(y=h1, color="b", linestyle="--")
        plt.axhline(y=h2, color="b", linestyle="--")
        plt.xlabel("X [Mm]")
        plt.ylabel("Z [Mm]")
        plt.title("Frame {:.1f}: {:.2f} min".format(t, time_array[t] / 60))
        # cbar = plt.colorbar(im1, cax=cbaxes, orientation="horizontal")

        # axs.set(xlabel="x [Mm]", ylabel="z [Mm]")
        # axs.axes.set_aspect(10)
        # myLocator = ticker.MultipleLocator(0.5)
        # axs.yaxis.set_major_locator(myLocator)
        plt.pause(0.1)
        # plt.show()
    return fig


def plot_Lamb(cs_val, N_BV, kx_array):
    """Plot the Lamb line given an adiabatic sound speed value and horizontal wavenumber array.

	Arguments:
		cs_val -- The adiatic sound speed in the atmosphere [Mm/s].
		N_BV -- The isothermal Brunt-Vaisaila frequency. Units are in angular frequency [Hz].
		kx_array -- Horizontal wavenumber array [1/Mm].

	Returns:
		A plot showing the Lamb line.
    """
    plt.ion()
    f1 = plt.figure(5)
    plt.title("Frequency vs Horizontal Wavenumber")
    plt.xlabel("kx [1/Mm]")
    plt.ylabel("frequency [mHz]")
    plt.axhline(y=N_BV / 2 / np.pi * 1000, linestyle="--", linewidth=2, color="navy")
    plt.plot(
        kx_array,
        build_functions.Lamb_equation(cs_val, kx_array) / 2 / np.pi * 1000,
        color="k",
        linewidth=1.5,
    )
    plt.plot(
        kx_array,
        -1 * build_functions.Lamb_equation(cs_val, kx_array) / 2 / np.pi * 1000,
        color="k",
        linewidth=1.5,
    )
    plt.axhline(y=0, linestyle="--", linewidth=2, color="navy")
    plt.xlim(kx_array.min(), kx_array.max())
    plt.ylim(0, 6)
    return f1


def plot_power(array, kh_grid, fgrid, h1, cmapc):
    """
	Plot the 2D power at a given height.

	Arguments:
		array -- The wave field [array of x, z, t].
		kh_grid -- Horizontal wavenumber grid [1/Mm].
		fgrid -- Frequency grid [mHz].
		h1 -- Height 1 [Mm].

	Returns:
		Returns a plot showing the power of the time series at a particular height.
	"""
    plt.ion()
    f2 = plt.figure()
    plt.title("Power at height {:.2f} km".format(h1 * 1000))
    plt.xlabel("kx [1/Mm]")
    plt.ylabel("Frq [mHz]")
    plt.pcolormesh(kh_grid, fgrid, np.log10(array).T, cmap=cmapc)
    cbar = plt.colorbar()
    cbar.set_label("log10 Power")
    return f2


def plot_phase(array, kh_grid, fgrid, h1, h2):
    """
	Plot the 2D phase at a two different heights.

	Arguments:
		array -- The wave field [array of x, z, t].
		kh_grid -- Horizontal wavenumber grid [1/Mm].
		fgrid -- Frequency grid [mHz].
		h1 -- Height 1 [Mm].
		h2 -- Height 2 [Mm].

	Returns:
		Returns a plot showing the phase of the time series at two different heights.

	"""
    plt.ion()
    f3 = plt.figure()
    plt.title("Phase at heights {:.2f} - {:2f} km".format(h1 * 1000, h2 * 1000))
    plt.xlabel("kx [1/Mm]")
    plt.ylabel("Frq [mHz]")
    plt.pcolormesh(kh_grid, fgrid, array.T, cmap="viridis")
    cbar = plt.colorbar()
    cbar.set_label("Phase [deg]")
    return f3


def create_grids_for_plotting(
    dx, dt, pix_x, pix_z, pix_t, min_x_Mm, max_x_Mm, min_z_Mm, max_z_Mm
):
    kx = np.pi / dx
    omega = np.pi / dt  # angular frequency [Hz]
    v = omega / (2 * np.pi)  # cyclic frequency [Hz]
    frq = v * 1000  # cyclic frequency [mHz]
    duration = pix_t * dt  # duration of time series [s]

    distance = 147.1e6  # Earth-Sun distance in km
    conversion_arcseconds_to_Mm = spectral_analysis.conversion_arcseconds_to_Mm(
        distance
    )

    time_arr = np.linspace(0, duration, pix_t)  # time array [s]
    frq_array = np.linspace(-frq, frq, int(pix_t))  # frequency array [mHz]
    kh_wavenumber_array = np.linspace(-kx, kx, int(pix_x))  # 1/[arcsec]
    kh_wavenumber_array = kh_wavenumber_array / conversion_arcseconds_to_Mm  # [1/Mm]

    # spatial grid arrays
    x_array = np.linspace(min_x_Mm, max_x_Mm, int(pix_x))  # Mm
    z_array = np.linspace(min_z_Mm, max_z_Mm, pix_z + 1)  # Mm

    return time_arr, frq_array, kh_wavenumber_array, x_array, z_array
