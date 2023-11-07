# Import the necessary modules

import random
import time

import cmasher as cmr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from tqdm import tqdm

import build_functions
import plot_waves
import spectral_analysis


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

## --------------------------------------------------------------------------------------------------- ##

# Simulation parameters

# Factor to convert arcseconds as seen on Earth to megameters on the Sun
distance = 147.1e6  # Earth-Sun distance in km
conversion_arcseconds_to_Mm = spectral_analysis.conversion_arcseconds_to_Mm(distance)

dx = 0.6  # Spatial sampling [arcsec/pix]
kx = np.pi / dx  # Nyquist Horizontal Wavenumber [1/arcsec]

dt = 11.88  # Cadence [s]
omega = np.pi / dt  # Angular frequency [Hz]
v = omega / (2 * np.pi)  # Converting to yclic frequency [Hz]
frq = v * 1000  # Cyclic frequency [mHz]

pix_t = 840  # Number of time steps
duration = pix_t * dt  # Duration of time series [s]
print("Duration: ", duration / 3600, " hr")

pix_x = 160  # Number of pixels/grid points in x-direction
pix_z = 300  # Number of pixels/grid points in z-drection


## --------------------------------------------------------------------------------------------------- ##

# Grid initialization

time_arr = np.linspace(0, duration, pix_t)  # Time array [s]
frq_array = np.linspace(-frq, frq, int(pix_t))  # Frequency array [mHz]

kh_wavenumber_array = np.linspace(
    -kx, kx, int(pix_x)
)  # Horizontal wavenumber array 1/[arcsec]
kh_wavenumber_array = (
    kh_wavenumber_array / conversion_arcseconds_to_Mm
)  # Horizontal wavenumber array [1/Mm]

x_array = np.linspace(0, 68, int(pix_x))  # spatial x grid [Mm]
z_array = np.linspace(-0.01, 4.0, pix_z + 1)  # spatial z grid (height) [Mm]

xpositions, zpositions, tpositions = np.meshgrid(
    x_array, z_array, time_arr
)  # Meshgrid defining the field domain [x, z, t]

print("Grid information:")
print("\n")
print("The length of the spatial z array is ", z_array.shape)
print("The length of the spatial x array is ", x_array.shape)

print("z array ({:.2f},{:.2f}) Mm".format(z_array.min(), z_array.max()))
print("x array ({:.2f},{:.2f}) Mm".format(x_array.min(), x_array.max()))

print("z array resolution {:.2f} Mm".format(np.diff(z_array)[0]))
print("x array resolution {:.2f} Mm".format(np.diff(x_array)[0]))

print("The shape of the meshgrid for the field domain is ", xpositions.shape)

# Plotting initialization grids

xgrid, zgrid = np.meshgrid(
    x_array, z_array
)  # Meshgrid for plotting wavepacket(s) in X-Z domain

KHPOS, NU = np.meshgrid(
    kh_wavenumber_array, frq_array
)  # Meshgrid for plotting in frequency-wavenumber space

## --------------------------------------------------------------------------------------------------- ##

# Defining parameters for the simulation run

np.random.seed(10)

# Total number of wave packets to add at t=0
total_nom_waves = 2

# Constants
gamma = 5.0 / 3.0  # The adiabatic index of an ideal gas

cs_val = (
    7.0  # adiabatic acoustic sound speed value. Common value in the photosphere[km/s]
)
cs_val = cs_val / 1000  # Matching units [Mm/s]

grav = 0.274  # Gravitational acceleration of the Sun [km/s^2]
grav = grav / 1000  # Matching units [Mm/s^2]

tau_val = 80  # Radiative Damping [s]


chosen_N = build_functions.isothermal_N(
    grav, cs_val
)  # Brunt-Vaisala Frequency [Hz / angular frequency]


chosen_omega_ac = 5.4  # Acoustic cut-off frequency. Common value in solar photosphere [mHz / cyclic frequency]
chosen_omega_ac = (chosen_omega_ac / 1000) * (
    2 * np.pi
)  # Convert to angular frequency [Hz]

# Dictionary to store values
time_series_parameters = []

# Initialize empty field domain to store wavepacket(s) [x, z, t]
wave_domain = np.zeros([len(z_array), len(x_array), len(time_arr)], dtype=np.float64)
print("Shape of domain: ", wave_domain.shape)


# Dictionary to store values
initialization_parameters = []

# Create dictionary to store wave parameters for future access.
initialization_entries = {
    "dt": dt,
    "kx": kx,
    "dx": dx,
    "frq": frq,
    "nt": pix_t,
    "nx": pix_x,
    "nz": pix_z,
    "tau": tau_val,
    "cs": cs_val,
    "wac": chosen_omega_ac,
    "dt_unit": "s",
    "kx_unit": "1/arcsec",
    "dx_unit": "arcsec/pix",
    "frq_unit": "mHz",
    "tau_unit": "s",
    "cs_unit": "Mm/s",
    "wac_unit": "Hz",
}
initialization_parameters.append(initialization_entries.copy())


def create_wavepacket_characteristics(chosen_N, cs_val):
    # On each iteration, choose a wavepacket frequency from the defined normal distribution
    rand_omega = build_functions.random_omega_value_generator(
        1.5, 0.7, (chosen_N * 1000) / 2 / np.pi, total_len=1
    )

    rand_omega = (rand_omega / 1000) * (
        2 * np.pi
    )  # Convert cyclic to angular frequency [Hz]

    # On each iteration, choose a horizontal wavenumber from a defined normal distribution
    # This code ensures that a random positive or negative horizontal wavenumber is chosen.
    rand_horizontal_wavenumber = build_functions.random_kx_value_generator(
        2, 1.2, total_len=1
    )  # [1/Mm]

    # Check if angular frequency and horizontal wavenumber associated with wavepacket(s) is below the Lamb line.
    # If not, values in predetermined ranges that fall below the Lamb line are chosen.
    rand_omega, rand_horizontal_wavenumber = build_functions.recalculate_omega_val_kx(
        cs_val, rand_omega, rand_horizontal_wavenumber
    )

    rand_arb_amp = np.random.uniform(
        1e-4, 0.00025, 1
    )  # arbitrary additional amplitude of wavepacket(s) [Mm/s]

    # Define the width and depth of the wavepacket(s).
    # These values whether constant or not should not be larger or on par with the field domain
    rand_sigx, rand_sigz = (
        np.random.uniform(0.5, 1.5, 1),
        np.random.uniform(0.1, 0.5, 1),
    )  # [Mm]
    # (1.0, 0.35)

    # Initial x and z location of the wavepacket(s) at t=0 [Mm].
    # Range should not be greater than the field domain.
    rand_x0, rand_z0 = (
        np.abs(np.random.uniform(0, 68, 1)),
        (np.random.uniform(-0.01, 0.2, 1)),
    )
    return (
        rand_omega,
        rand_horizontal_wavenumber,
        rand_arb_amp,
        rand_sigx,
        rand_sigz,
        rand_x0,
        rand_z0,
    )


def create_wavepacket(
    rand_omega,
    chosen_omega_ac,
    cs_val,
    rand_horizontal_wavenumber,
    tau_val,
    chosen_N,
    grav,
    xpositions,
    rand_x0,
    tpositions,
    rand_sigx,
    zpositions,
    rand_z0,
    rand_sigz,
    rand_arb_amp,
):
    # Computing components of the dispersion relation
    aterm = build_functions.a_term(
        rand_omega,
        chosen_omega_ac,
        cs_val,
        rand_horizontal_wavenumber,
        tau_val,
        chosen_N,
        grav,
    )
    bterm = build_functions.b_term(
        rand_omega, tau_val, chosen_N, rand_horizontal_wavenumber, grav
    )

    # Computing the dispersion relation (the vertical wavenumber) [1/Mm]
    kz_val = build_functions.compute_kz(aterm, bterm)

    # Compute the horizontal and vertical group velocity
    # For gravity waves, the vertical group velocity shows the direction of wave and energy propagation.
    cgh, cgz = build_functions.compute_group_velocity(
        rand_omega, kz_val, rand_horizontal_wavenumber, chosen_N, cs_val
    )  # [Mm/s

    # Compute the (individual) amplitude of the wavepacket(s).
    ampwave = build_functions.wave_packet_amplitude(
        xpositions,
        rand_x0,
        cgh,
        cgz,
        tpositions,
        rand_sigx,
        zpositions,
        rand_z0,
        rand_sigz,
        rand_arb_amp,
    )

    # Compute the vertical displacement (perturbation) of the wavepacket(s)
    vertical_displacement = build_functions.perturbation(
        ampwave,
        rand_horizontal_wavenumber,
        xpositions,
        kz_val,
        zpositions,
        rand_omega,
        tpositions,
    )
    return vertical_displacement


## --------------------------------------------------------------------------------------------------- ##

# Computing gravity wavepacket(s) in an isothermal atmosphere with radiative damping

# Initializes a frequency-wavenumber plot with the Lamb line overplotted to ensure wavepacket(s) have a
# frequency and horizontal wavenumber that fall below the Lamb line
plot_waves.plot_Lamb(cs_val, chosen_N, kh_wavenumber_array)

t0 = time.process_time()  # Measure CPU time

t_shift = np.arange(0, 840, 6)

for tv in t_shift:
    new_time = time_arr[tv:] - time_arr[tv:].min()
    xpositions, zpositions, tpositions = np.meshgrid(x_array, z_array, new_time)
    # For loop to generate initial gravity wavepacket(s) given by the total_nom_waves parameter

    wave_amount_choice = random.choice([2, 3, 5, 8])
    for nom in tqdm(
        range(0, wave_amount_choice),
        desc=f"Computing {wave_amount_choice} waves at t = {tv}",
    ):

        (
            rand_omega,
            rand_horizontal_wavenumber,
            rand_arb_amp,
            rand_sigx,
            rand_sigz,
            rand_x0,
            rand_z0,
        ) = create_wavepacket_characteristics(chosen_N, cs_val)
        # Create dictionary to store wave parameters for future access.
        entries = {
            "omega": rand_omega * 1000 / 2 / np.pi,
            "Kh": rand_horizontal_wavenumber,
            "x0": rand_x0,
            "z0": rand_z0,
            "sigx": rand_sigx,
            "sigz": rand_sigz,
            "arb": rand_arb_amp,
        }
        time_series_parameters.append(entries.copy())

        # Plot the chosen wavepacket(s) frequency and horizontal wavenumber in real time during computation.
        # Keep track that code works.
        plt.figure(5)
        plt.scatter(rand_horizontal_wavenumber, rand_omega / 2 / np.pi * 1000)
        plt.pause(0.1)

        vertical_displacement = create_wavepacket(
            rand_omega,
            chosen_omega_ac,
            cs_val,
            rand_horizontal_wavenumber,
            tau_val,
            chosen_N,
            grav,
            xpositions,
            rand_x0,
            tpositions,
            rand_sigx,
            zpositions,
            rand_z0,
            rand_sigz,
            rand_arb_amp,
        )

        # Add each wavepacket to the field domain
        wave_domain[:, :, tv:] += vertical_displacement


cpu_time = time.process_time() - t0
print("CPU Time ", cpu_time)

# plt.tight_layout()
# plt.show()

# image = plot_waves.plot_wave_packet(
#     wave_domain,
#     tdim=0,
#     xgrid=xgrid,
#     zgrid=zgrid,
#     cmapc=cmr.fusion,
#     min_z=z_array.min(),
#     max_z=z_array.max(),
#     min_x=x_array.min(),
#     max_x=x_array.max(),
# )
zheight0, zheight1 = spectral_analysis.compute_height(0.01, 0.25, z_array)
print("Height 1 {:.2f} km".format(z_array[zheight0] * 1000))
print("Height 2 {:.2f} km".format(z_array[zheight1] * 1000))

anim = plot_waves.plot_animation(
    time_arr,
    4,
    wave_domain,
    xgrid=xgrid,
    zgrid=zgrid,
    h1=z_array[zheight0],
    h2=z_array[zheight1],
    cmapc=cmr.fusion,
)
# wavesdom -= wavesdom.mean()

# cross_spec = spectral_analysis.cross_spectrum_2D(
#     wave_domain[zheight0, :, :], wave_domain[zheight1, :, :]
# )
# phase2d = np.angle(cross_spec, deg=True)

# power_spec = spectral_analysis.power_spectrum_2D(wave_domain[zheight0, :, :])
# power_spec2 = spectral_analysis.power_spectrum_2D(wave_domain[zheight1, :, :])

# power1 = plot_waves.plot_power(power_spec, KHPOS, NU, z_array[zheight0])
# power2 = plot_waves.plot_power(power_spec2, KHPOS, NU, z_array[zheight1])

# phases = plot_waves.plot_phase(phase2d, KHPOS, NU, z_array[zheight0], z_array[zheight1])

print("Saving run....")
np.savez(
    "Data/Wavepacket.everysixth.timestep.generate.waves.npz",
    waves=wave_domain,
    wave_params=time_series_parameters,
    intialization_params=initialization_parameters,
)
print("Run complete.")

