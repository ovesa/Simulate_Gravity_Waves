# setting up necessary modules

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import rc
from numpy import fft
from tqdm import tqdm

# import python modules to create wave packets
import spectral_analysis
import build_functions
import plot_waves

# Visualization parameters
tex_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times New Roman",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 14,
    "font.size": 14,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 12,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": True,
    "legend.framealpha": 0.8,
    "axes.formatter.use_mathtext": True,
    # "figure.dpi": 300,
    # "figure.autolayout": True,
    "lines.linewidth": 2,
    "ytick.minor.visible": True,
    "xtick.minor.visible": True,
    "figure.facecolor": "white",
    # "savefig.bbox": "tight",
    # "savefig.dpi": 300,
}
matplotlib.rcParams.update(tex_fonts)

# Set the font used for MathJax - more on this later
rc("mathtext", **{"default": "regular"})

# The following %config line changes the inline figures to have a higher DPI.
# You can comment out (#) this line if you don't have a high-DPI (~220) display
plt.style.use("classic")
plt.rcParams["figure.facecolor"] = "white"


# Constants

distance = 147.1e6  # Earth-Sun distance in km


# Set up grids and other parameters

# factor to convert arcseconds as seen on Earth to megameters on the Sun
conversion_arcseconds_to_Mm = spectral_analysis.conversion_arcseconds_to_Mm(distance)

dx = 0.6  # spatial sampling [arcsec/pix]
kx = np.pi / dx  # Nyquist Horizontal Wavenumber [1/arcsec]

dt = 60  # cadence [s]
omega = np.pi / dt  # angular frequency [Hz]
v = omega / (2 * np.pi)  # cyclic frequency [Hz]
frq = v * 1000  # cyclic frequency [mHz]

nom_frames = 400  # number of frames
duration = nom_frames * dt  # duration of time series [s]
print("Duration: ", duration / 60 / 60, " hr")

pix_x = 160  # number of pixels in x-direction
pix_z = 300  # number of pixels in z-drection

time_arr = np.linspace(0, duration, nom_frames)  # time array [s]
frq_array = np.linspace(-frq, frq, int(nom_frames))  # frequency array [mHz]
kh_wavenumber_array = np.linspace(-kx, kx, int(pix_x))  # 1/[arcsec]
kh_wavenumber_array = kh_wavenumber_array / conversion_arcseconds_to_Mm  # [Mm]

# spatial grid arrays
x_array = np.linspace(0, 68, int(pix_x))  # Mm
z_array = np.linspace(-0.01, 2.0, pix_z + 1)  # Mm

print("The length of the spatial z array is ", z_array.shape)
print("The length of the spatial x array is ", x_array.shape)

print("z array (max,min) ", z_array.max(), z_array.min())
print("x array (max,min) ", x_array.max(), x_array.min())

print("z array resolution ", np.diff(z_array)[0], " Mm")
print("x array resolution ", np.diff(x_array)[0], " Mm")

# Create meshgrid for storing domain
xpositions, zpositions = np.meshgrid(x_array, z_array)
print("The shape of the meshgrids is ", xpositions.shape)


np.random.seed(10)

# Number of wave packets to initially add into domain
total_nom_waves = 1

# Constants
gamma = 5.0 / 3.0

cs_val = 7.0  # km/s
cs_val = cs_val / 1000  # Mm/s

grav = 0.274  # km/s^2
grav = grav / 1000  # Mm/s^2

tau_val = 80  # s

# Brunt-Vaisala Frequency
chosen_N = build_functions.isothermal_N(grav, cs_val, gamma)  # angular frequency


chosen_omega_0 = 5.4  # mHz
chosen_omega_0 = (chosen_omega_0 / 1000) * (2 * np.pi)  # angular frequency [Hz]

# Dictionary to store values
time_series_parameters = []

# store times series [x,z,t]
wavesdom = np.zeros([len(z_array), len(x_array), len(time_arr)], dtype=np.float64)
print("Shape of domain: ", wavesdom.shape)
wavesdom2 = np.zeros([len(z_array), len(x_array), len(time_arr)], dtype=np.float64)


plot_waves.plot_Lamb(cs_val, chosen_N, kh_wavenumber_array)
count = 0
for nom in range(0, total_nom_waves):
    rand_omega = np.abs(np.random.uniform(0.5, 3.0, 1))  # np.abs(np.random.normal(2,
    rand_omega = (rand_omega / 1000) * (2 * np.pi)  # angular frequency [Hz]

    rand_horizontal_wavenumber = np.abs(
        np.random.uniform(0.5, 6.0, 1)
    )  # .abs(np.random.normal(2,0.5,total_nom_waves))  # [1/Mm]

    rand_omega, rand_horizontal_wavenumber = build_functions.recalculate_omega_kx(
        cs_val, rand_omega, rand_horizontal_wavenumber
    )

    rand_arb_amp = 1.0  # arbitrary amplitude
    rand_sigx, rand_sigz = (
        np.abs(np.random.uniform(0.5, 1.0, 1)),
        np.abs(np.random.uniform(0.05, 0.2, 1)),
    )
    # change z parameter to create a time series that forms at a "higher layer"????
    rand_x0, rand_z0 = (
        np.abs(np.random.uniform(0, 50, 1)),
        (np.random.uniform(0, 0.8, 1)),
    )  # initial position of wave packet at t=0 [Mm]
    entries = {
        "omega": rand_omega * 1000 / 2 / np.pi,
        "Kh": rand_horizontal_wavenumber,
        "Amp": rand_arb_amp,
        "sigx": rand_sigx,
        "sigz": rand_sigz,
        "x0": rand_x0,
        "z0": rand_z0,
    }
    time_series_parameters.append(entries.copy())

    plt.figure(5)
    plt.scatter(rand_horizontal_wavenumber, rand_omega / 2 / np.pi * 1000)
    plt.pause(0.1)

    for tdim in tqdm(range(0, len(time_arr)), desc=f"Wave {nom+1}", position=0):
        aterm = build_functions.a_term(
            rand_omega,
            chosen_omega_0,
            cs_val,
            rand_horizontal_wavenumber,
            tau_val,
            chosen_N,
            grav,
        )
        bterm = build_functions.b_term(
            rand_omega, tau_val, chosen_N, rand_horizontal_wavenumber, grav
        )
        kz_val = build_functions.compute_kz(aterm, bterm)
        cgh, cgz = build_functions.compute_group_velocity(
            rand_omega, kz_val, rand_horizontal_wavenumber, chosen_N, cs_val
        )
        ampwave = build_functions.wave_packet_amplitude(
            rand_arb_amp,
            xpositions,
            rand_x0,
            cgh,
            time_arr[tdim],
            rand_sigx,
            zpositions,
            rand_z0,
            cgz,
            rand_sigz,
        )
        vertical_displacement = build_functions.perturbation(
            ampwave,
            rand_horizontal_wavenumber,
            xpositions,
            kz_val,
            zpositions,
            rand_omega,
            time_arr[tdim],
        )
        wavesdom[:, :, tdim] += vertical_displacement

        if tdim % 50 == 0:
            nomv = 2

            for nom in range(0, nomv):
                rand_omega = np.abs(
                    np.random.uniform(0.5, 3.0, 1)
                )  # np.abs(np.random.normal(2,
                rand_omega = (rand_omega / 1000) * (2 * np.pi)  # angular frequency [Hz]
                rand_horizontal_wavenumber = np.abs(np.random.uniform(0.5, 6.0, 1))
                (
                    rand_omega,
                    rand_horizontal_wavenumber,
                ) = build_functions.recalculate_omega_kx(
                    cs_val, rand_omega, rand_horizontal_wavenumber
                )
                rand_arb_amp = 1.0  # arbitrary amplitud
                rand_sigx, rand_sigz = (
                    np.abs(np.random.uniform(0.5, 1.0, 1)),
                    np.abs(np.random.uniform(0.05, 0.2, 1)),
                )
                rand_x0, rand_z0 = (
                    np.abs(np.random.uniform(0, 50, 1)),
                    (np.random.uniform(0, 0.8, 1)),
                )  # initial position of wave packet at t=0 [Mm]

                entries = {
                    "omega": rand_omega * 1000 / 2 / np.pi,
                    "Kh": rand_horizontal_wavenumber,
                    "Amp": rand_arb_amp,
                    "sigx": rand_sigx,
                    "sigz": rand_sigz,
                    "x0": rand_x0,
                    "z0": rand_z0,
                }
                time_series_parameters.append(entries.copy())

                plt.figure(5)
                plt.scatter(rand_horizontal_wavenumber, rand_omega / 2 / np.pi * 1000)
                plt.pause(0.1)

                for tdim2 in tqdm(
                    range(count, len(time_arr)),
                    desc=f"Added Wave",
                    position=1,
                    leave=False,
                ):
                    aterm = build_functions.a_term(
                        rand_omega,
                        chosen_omega_0,
                        cs_val,
                        rand_horizontal_wavenumber,
                        tau_val,
                        chosen_N,
                        grav,
                    )
                    bterm = build_functions.b_term(
                        rand_omega, tau_val, chosen_N, rand_horizontal_wavenumber, grav
                    )
                    kz_val = build_functions.compute_kz(aterm, bterm)
                    cgh, cgz = build_functions.compute_group_velocity(
                        rand_omega, kz_val, rand_horizontal_wavenumber, chosen_N, cs_val
                    )
                    ampwave = build_functions.wave_packet_amplitude(
                        rand_arb_amp,
                        xpositions,
                        rand_x0,
                        cgh,
                        time_arr[tdim2],
                        rand_sigx,
                        zpositions,
                        rand_z0,
                        cgz,
                        rand_sigz,
                    )
                    vertical_displacement = build_functions.perturbation(
                        ampwave,
                        rand_horizontal_wavenumber,
                        xpositions,
                        kz_val,
                        zpositions,
                        rand_omega,
                        time_arr[tdim2],
                    )
                    wavesdom2[:, :, tdim2] += vertical_displacement
            wavesdom = wavesdom + wavesdom2
            count += 50


# plt.tight_layout()
# plt.show()

# image = plot_waves.plot_wave_packet(wavesdom,tdim=0, xgrid=xpositions, zgrid=zpositions, max_x_arr = x_array.max(), max_z_arr = z_array.max())

zheight0, zheight1 = spectral_analysis.compute_height(0.01, 0.25, z_array)
print("Height 1 {:.2f} km".format(z_array[zheight0] * 1000))
print("Height 2 {:.2f} km".format(z_array[zheight1] * 1000))

anim = plot_waves.plot_animation(
    time_arr,
    wavesdom,
    xgrid=xpositions,
    zgrid=zpositions,
    h1=z_array[zheight0],
    h2=z_array[zheight1],
)


cross_spec = spectral_analysis.cross_spectrum_2D(
    wavesdom[zheight0, :, :], wavesdom[zheight1, :, :]
)
phase2d = np.angle(cross_spec, deg=True)

power_spec = spectral_analysis.power_spectrum_2D(wavesdom[zheight0, :, :])
power_spec2 = spectral_analysis.power_spectrum_2D(wavesdom[zheight1, :, :])


power1 = plot_waves.plot_power(
    power_spec, kh_wavenumber_array, frq_array, z_array[zheight0]
)
power2 = plot_waves.plot_power(
    power_spec2, kh_wavenumber_array, frq_array, z_array[zheight1]
)

phases = plot_waves.plot_phase(
    phase2d, kh_wavenumber_array, frq_array, z_array[zheight0], z_array[zheight1]
)

