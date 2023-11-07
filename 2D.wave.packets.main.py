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

nom_frames = 60  # number of frames
duration = nom_frames * dt  # duration of time series [s]
print("Duration: ", duration / 60 / 60, " hr")

pix_x = 300  # number of pixels in x-direction
pix_z = 160  # number of pixels in z-drection

time_arr = np.linspace(0, duration, nom_frames)  # time array [s]
frq_array = np.linspace(-frq, frq, int(nom_frames))  # frequency array [mHz]
kh_wavenumber_array = np.linspace(-kx, kx, int(pix_x))  # 1/[arcsec]
kh_wavenumber_array = kh_wavenumber_array / conversion_arcseconds_to_Mm  # [Mm]

# spatial grid arrays
x_array = np.linspace(0, 10, int(pix_x))  # Mm
z_array = np.linspace(0, 30, pix_z + 1)  # Mm

print("The length of the spatial z array is ", z_array.shape)
print("The length of the spatial x array is ", x_array.shape)

print("z array (max,min) ", z_array.max(), z_array.min())
print("x array (max,min) ", x_array.max(), x_array.min())

print("z array resolution ", np.diff(z_array)[0], " Mm")
print("x array resolution ", np.diff(x_array)[0], " Mm")

# Create meshgrid for storing domain
xpositions, zpositions = np.meshgrid(x_array, z_array)
print("The shape of the meshgrids is ", xpositions.shape)


# parameters for wave packet and field

gamma = 5.0 / 3.0

cs_val = 6.0  # km/s
cs_val = cs_val / 1000  # Mm/s

grav = 0.274  # km/s^2
grav = grav / 1000  # Mm/s^2

tau_val = 10  # s

chosen_omega = 1.5  # cyclic frequency [mHz]
chosen_omega = (chosen_omega / 1000) * (2 * np.pi)  # angular frequency [Hz]

chosen_omega_0 = 5.4  # mHz
chosen_omega_0 = (chosen_omega_0 / 1000) * (2 * np.pi)  # angular frequency [Hz]

chosen_horizontal_wavenumber = 2.5  # [1/Mm]

sigx, sigz = 3, 0.1  # 0.8, 0.2  # physical widths and depths in x-z plane [Mm]

x0, z0 = 1.0, 0.0  # initial position of wave packet at t=0 [Mm]

# Brunt-Vaisala Frequency
chosen_N = build_functions.isothermal_N(grav, cs_val, gamma)  # angular frequency

arb_amp = 1.0  # arbitrary amplitude


aterm = build_functions.a_term(
    chosen_omega,
    chosen_omega_0,
    cs_val,
    chosen_horizontal_wavenumber,
    tau_val,
    chosen_N,
    grav,
)

bterm = build_functions.b_term(
    chosen_omega, tau_val, chosen_N, chosen_horizontal_wavenumber, grav
)

kz_val = build_functions.compute_kz(aterm, bterm)
print("Vertical wavenumber is %f  1/Mm" % kz_val)

cgh, cgz = build_functions.compute_group_velocity(
    chosen_omega, kz_val, chosen_horizontal_wavenumber, chosen_N, cs_val
)
print(
    "Horizontal group speed is %f km/s and vertical group speed is %f km/s"
    % (cgh * 1000, cgz * 1000)
)

cph, cpz = build_functions.compute_phase_velocity(
    chosen_N, chosen_omega, chosen_horizontal_wavenumber, chosen_omega_0, cs_val, kz_val
)

print(
    "Horizontal phase speed is %f km/s and vertical phase speed is %f km/s"
    % (cph * 1000, cpz * 1000)
)

# Set up array for the field domain to hold the wave packets
waves_dom = np.zeros([len(z_array), len(x_array), len(time_arr)], dtype=np.float64)
print("Shape of domain: ", (waves_dom.shape))


for tdim in tqdm(range(0, len(time_arr))):
    ampwave = build_functions.wave_packet_amplitude(
        xpositions, x0, cgz, cgz, time_arr[tdim], sigx, zpositions, z0, sigz, arb_amp
    )

    vertical_displacement = build_functions.perturbation(
        ampwave,
        chosen_horizontal_wavenumber,
        xpositions,
        kz_val,
        zpositions,
        chosen_omega,
        time_arr[tdim],
    )

    waves_dom[:, :, tdim] = vertical_displacement


# image = plot_waves.plot_wave_packet(
#     waves_dom, tdim=10, xgrid=xpositions, zgrid=zpositions, cmapc="bwr"
# )

zheight0, zheight1 = spectral_analysis.compute_height(0.01, 0.25, z_array)
print("Height 1 {:.2f} km".format(z_array[zheight0] * 1000))
print("Height 2 {:.2f} km".format(z_array[zheight1] * 1000))


anim = plot_waves.plot_animation(
    time_arr,
    1,
    waves_dom,
    xgrid=zpositions,
    zgrid=xpositions,
    h1=z_array[zheight0],
    h2=z_array[zheight1],
    cmapc="bwr",
)

