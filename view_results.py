import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib.colors as mcolors

import plot_waves
import spectral_analysis
import build_functions

time_series = np.load("Data/Wavepacket.20000.P2.npz", allow_pickle=True)

print("This file contains the following arrays: ", time_series.files)

wave_run = time_series["waves"]
print("Shape of time series: ", wave_run.shape)
# acess dictionary elements: time_series['wave_params'][0]['omega']
(
    time_array,
    frq_array,
    kh_wavenumber_array,
    x_array,
    z_array,
) = plot_waves.create_grids_for_plotting(0.6, 11.88, 160, 300, 840, 0, 68, -0.01, 4.0)

# Meshgrid for plotting wavepacket(s) in X-Z domain
xgrid, zgrid = np.meshgrid(x_array, z_array)

# Meshgrid for plotting power and phase
KHPOS, NU = np.meshgrid(kh_wavenumber_array, frq_array)

# Meshgrid for plotting frq versus distance
XPOS, NU2 = np.meshgrid(x_array, frq_array)


zheight0, zheight1 = spectral_analysis.compute_height(0.25, 0.4, z_array)
print("Height 1 {:.2f} km".format(z_array[zheight0] * 1000))
print("Height 2 {:.2f} km".format(z_array[zheight1] * 1000))


image = plot_waves.plot_wave_packet(
    wave_run,
    tdim=10,
    xgrid=xgrid,
    zgrid=zgrid,
    cmapc=cmr.fusion,
    min_z=z_array.min(),
    max_z=z_array.max(),
    min_x=x_array.min(),
    max_x=x_array.max(),
)

# mean_dat = wave_run.mean()

# wave_run = wave_run - mean_dat

power_spec = spectral_analysis.power_spectrum_2D(wave_run[zheight0, :, :])
power_spec2 = spectral_analysis.power_spectrum_2D(wave_run[zheight1, :, :])

plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2)
im1 = ax1.pcolormesh(
    KHPOS,
    NU,
    np.log10(power_spec.T),
    cmap=cmr.sepia,
    vmin=0,
    vmax=np.log10(power_spec.T).max(),
)
plt.colorbar(im1, ax=ax1)
ax1.plot(
    kh_wavenumber_array,
    build_functions.Lamb_equation(0.007, kh_wavenumber_array) / 2 / np.pi * 1000,
    color="k",
    linewidth=1.5,
)
ax1.set_xlabel("KH [1/Mm]")
ax1.set_ylabel("NU [mHz]")
ax1.set_ylim([0, 8])
ax1.set_xlim([0, 7])

ax1.set_title("Power at z = {:.2f} km".format(z_array[zheight0] * 1000))


im2 = ax2.pcolormesh(
    KHPOS,
    NU,
    np.log10(power_spec2.T),
    cmap=cmr.sepia,
    vmin=0,
    vmax=np.log10(power_spec2.T).max(),
)
plt.colorbar(im2, ax=ax2)
ax2.plot(
    kh_wavenumber_array,
    build_functions.Lamb_equation(0.007, kh_wavenumber_array) / 2 / np.pi * 1000,
    color="k",
    linewidth=1.5,
)
ax2.set_xlabel("KH [1/Mm]")
ax2.set_ylabel("NU [mHz]")
ax2.set_ylim([0, 8])
ax2.set_xlim([0, 7])

ax2.set_title("Power at z = {:.2f} km".format(z_array[zheight1] * 1000))

plt.tight_layout()
plt.show()


cross_spec = spectral_analysis.cross_spectrum_2D(
    wave_run[zheight0, :, :], wave_run[zheight1, :, :]
)

# cross_spec[:, (frq_array >= 3) & (frq_array <= -3)] = 0

phase2d = np.angle(cross_spec, deg=True)


plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2)

minv = (phase2d.T)[(frq_array <= 3) & (frq_array >= -3), :].min()
maxv = (phase2d.T)[(frq_array <= 3) & (frq_array >= -3), :].max()
im1 = ax1.pcolormesh(
    KHPOS[(frq_array <= 3) & (frq_array >= -3), :],
    NU[(frq_array <= 3) & (frq_array >= -3), :],
    (phase2d.T)[(frq_array <= 3) & (frq_array >= -3), :],
    cmap=cmr.fusion,
    norm=mcolors.TwoSlopeNorm(vmin=minv, vcenter=0, vmax=maxv),
)
plt.colorbar(im1, ax=ax1)
ax1.plot(
    kh_wavenumber_array,
    build_functions.Lamb_equation(0.007, kh_wavenumber_array) / 2 / np.pi * 1000,
    color="k",
    linewidth=1.5,
)
ax1.set_xlabel("KH [1/Mm]")
ax1.set_ylabel("NU [mHz]")
# ax1.set_ylim([0, 8])
ax1.set_xlim([0, 7])


ax1.set_title(
    "Phase at dz = {:.2f}-{:.2f} km".format(
        z_array[zheight0] * 1000, z_array[zheight1] * 1000
    )
)

im2 = ax2.pcolormesh(
    KHPOS[(frq_array <= 3) & (frq_array >= -3), :],
    NU[(frq_array <= 3) & (frq_array >= -3), :],
    np.abs(cross_spec.T)[(frq_array <= 3) & (frq_array >= -3), :],
    cmap=cmr.rainforest,
    vmin=np.abs(cross_spec.T).min(),
    vmax=np.abs(cross_spec.T).max(),
)
plt.colorbar(im2, ax=ax2)
ax2.set_title(
    "Cross Power at dz = {:.2f}-{:.2f} km".format(
        z_array[zheight0] * 1000, z_array[zheight1] * 1000
    )
)
ax2.plot(
    kh_wavenumber_array,
    build_functions.Lamb_equation(0.007, kh_wavenumber_array) / 2 / np.pi * 1000,
    color="k",
    linewidth=1.5,
)

ax2.set_ylim([0, 8])
ax2.set_xlim([0, 7])
ax2.set_xlabel("KH [1/Mm]")
ax2.set_ylabel("NU [mHz]")
plt.tight_layout()
plt.show()

crossspec1d = spectral_analysis.cross_spectrum_1D(
    wave_run[zheight0, :, :], wave_run[zheight1, :, :]
)
phase1d = np.angle(crossspec1d, deg=True)

crossspec1d[:, (frq_array >= 3) & (frq_array <= -3)] = 0


plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle(
    "Phase at dz = {:.2f}-{:.2f} km".format(
        z_array[zheight0] * 1000, z_array[zheight1] * 1000
    )
)

ax1.plot(
    frq_array[(frq_array <= 3) & (frq_array >= -3)],
    phase1d[102, (frq_array <= 3) & (frq_array >= -3)],
    linewidth=2,
    color="k",
    label="{:.2f} 1/Mm".format(kh_wavenumber_array[102]),
)
ax1.set_ylabel("Phase [deg]")
ax1.set_xlabel("NU [mHz]")
ax1.set_xlim([0, 5])
ax1.legend(loc="best")

ax2.plot(
    frq_array[(frq_array <= 3) & (frq_array >= -3)],
    phase1d[92, (frq_array <= 3) & (frq_array >= -3)],
    linewidth=2,
    color="k",
    label="{:.2f} 1/Mm".format(kh_wavenumber_array[92]),
)
ax2.set_ylabel("Phase [deg]")
ax2.set_xlabel("NU [mHz]")
ax2.set_xlim([0, 5])
ax2.legend(loc="best")


plt.tight_layout()
plt.show()


# dp = spectral_analysis.compute_phase_distance(
#     160, 840, wave_run[zheight0, :, :], wave_run[zheight1, :, :]
# )

dphase_distance = np.zeros([160, 840])

for xi in tqdm(range(0, 160)):
    fixed_x = 0
    cross_spec = spectral_analysis.cross_spectrum_1D(
        wave_run[zheight0, fixed_x, (frq_array <= 3) & (frq_array >= -3)],
        wave_run[zheight1, xi, (frq_array <= 3) & (frq_array >= -3)],
    )
    phase_spec = np.angle(cross_spec, deg=False)
    dphase_distance[xi, (frq_array <= 3) & (frq_array >= -3)] = phase_spec


plt.ion()
plt.figure()
minv = (np.rad2deg(dphase_distance).T).min()
maxv = (np.rad2deg(dphase_distance).T).max()
plt.pcolormesh(
    XPOS,
    NU2,
    np.rad2deg(dphase_distance).T,
    cmap=cmr.fusion,
    norm=mcolors.TwoSlopeNorm(vmin=minv, vcenter=0, vmax=maxv),
)

# XPOS,NU2,)
plt.colorbar()
# plt.ylim(-10, 10)
plt.xlabel("Distance dx [Mm]")
plt.ylabel("Frq [mHz]")
plt.tight_layout()
plt.show()
