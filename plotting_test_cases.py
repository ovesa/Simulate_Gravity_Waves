# Testing to see if the function works

import random

import matplotlib.pyplot as plt
import numpy as np

import build_functions
import spectral_analysis

conversion_arcseconds_to_Mm = spectral_analysis.conversion_arcseconds_to_Mm(147.1e6)
dx = 0.6  # spatial sampling [arcsec/pix]
kx = np.pi / dx  # Nyquist Horizontal Wavenumber [1/arcsec]
pix_x = 160  # number of pixels in x-direction
kh_wavenumber_array = np.linspace(-kx, kx, int(pix_x))  # 1/[arcsec]
kh_wavenumber_array = kh_wavenumber_array / conversion_arcseconds_to_Mm  # [1/Mm]
cs_val = 0.007  # Mm/s


def test_plot_Lamb(csval, kxarr, Nval=np.nan):
    plt.ion()
    f1 = plt.figure(5)
    plt.title("Frequency vs Horizontal Wavenumber")
    plt.xlabel("kx [1/Mm]")
    plt.ylabel("frequency [mHz]")
    plt.plot(
        kxarr,
        build_functions.Lamb_equation(csval, kxarr) / 2 / np.pi * 1000,
        linewidth=1.5,
        color="navy",
    )
    plt.plot(
        kxarr,
        -1 * build_functions.Lamb_equation(csval, kxarr) / 2 / np.pi * 1000,
        linewidth=1.5,
        color="navy",
    )
    plt.axhline(y=Nval / 2 / np.pi * 1000, color="r", linestyle="--", linewidth=2)
    plt.axhline(y=0, linestyle="--", linewidth=2, color="k")
    plt.xlim(kxarr.min(), kxarr.max())
    plt.pause(0.1)
    return f1


test_plot_Lamb(cs_val, kh_wavenumber_array)


def test_below_lamb_line(cs_val, test_kx_vals, test_rand_omegas):
    Lamb_omega = build_functions.Lamb_equation(cs_val, test_kx_vals)
    Lamb_omega = Lamb_omega / 2 / np.pi * 1000  # cylcic frequency

    if (test_rand_omegas <= Lamb_omega) or (test_rand_omegas <= -1 * Lamb_omega):
        plt.scatter(test_kx_vals, test_rand_omegas, color="red")
        print(" ({:.2f},{:.2f})  PASS".format(test_rand_omegas, test_kx_vals))
        # return test_rand_omegas, test_kx_vals
    # elif test_rand_omegas <= -1 * Lamb_omega:
    #     plt.scatter(test_kx_vals, test_rand_omegas, color="blue")
    #     print(" ({:.2f},{:.2f})  PASS".format(test_rand_omegas, test_kx_vals))
    else:
        plt.scatter(test_kx_vals, test_rand_omegas, color="orange", marker="x")
        print(" ({:.2f},{:.2f})  FAIL".format(test_rand_omegas, test_kx_vals))


test_rand_omega = np.abs(
    np.random.uniform(0.5, 3.0, 10)
)  # np.abs(np.random.uniform(0.5,3.0 ,1)) #np.abs(np.random.normal(2,
test_negative_kx_val = -1 * np.array([1.0, 2.0, 3.0, 4.0, 5.0])

plt.figure(5)

for kx in test_negative_kx_val:
    for omv in test_rand_omega:
        test_below_lamb_line(cs_val, kx, omv)

# only positive frequencies
test_omega_vals = np.abs(np.random.uniform(0.5, 3.0, 10))
# horizontal wavenumber can be negative
test_kx_vals = np.random.uniform(-6.0, 6.0, 10)

for kx in test_kx_vals:
    for omv in test_omega_vals:
        test_below_lamb_line(cs_val, kx, omv)


def test_below_lamb_line_single_values(cs_val, test_kx_vals, test_rand_omegas):
    Lamb_omega = build_functions.Lamb_equation(cs_val, test_kx_vals)
    Lamb_omega = Lamb_omega / 2 / np.pi * 1000  # cyclic frequency

    if test_rand_omegas <= Lamb_omega:
        print(" ({:.2f},{:.2f})  PASS".format(test_rand_omegas, test_kx_vals))
        # return test_rand_omegas, test_kx_vals
    elif test_rand_omegas <= -1 * Lamb_omega:
        print(" ({:.2f},{:.2f})  PASS".format(test_rand_omegas, test_kx_vals))
    else:
        print(" ({:.2f},{:.2f})  FAIL".format(test_rand_omegas, test_kx_vals))


print("\n")
print("Testing single values")
print("\n")

test_omega1, test_omega2, test_omega3, test_omega4, test_omega5, test_omega6 = (
    1.0,
    2.0,
    1.0,
    1.0,
    2.0,
    2.5,
)  # mHz
test_kx1, test_kx2, test_kx3, test_kx4, test_kx5, test_kx6 = (
    2.0,
    3.0,
    0.5,
    -1.0,
    -3.0,
    -1.0,
)  # 1/Mm

# Expected results: test1 = PASS, test2 = PASS, test3 = FAIL
# test4 = PASS; test5 = PASS; test6 = FAIL
test_below_lamb_line_single_values(cs_val, test_kx1, test_omega1)
test_below_lamb_line_single_values(cs_val, test_kx2, test_omega2)
test_below_lamb_line_single_values(cs_val, test_kx3, test_omega3)
test_below_lamb_line_single_values(cs_val, test_kx4, test_omega4)
test_below_lamb_line_single_values(cs_val, test_kx5, test_omega5)
test_below_lamb_line_single_values(cs_val, test_kx6, test_omega6)


def test_below_lamb_line_single_values_new_values(
    cs_val, test_kx_vals, test_rand_omegas
):
    Lamb_omega = build_functions.Lamb_equation(cs_val, test_kx_vals)
    Lamb_omega = Lamb_omega / 2 / np.pi * 1000  # cyclic frequency

    random_choice = random.choice([11, 12])

    if (test_rand_omegas <= Lamb_omega) or (test_rand_omegas <= -1 * Lamb_omega):
        print(" ({:.2f},{:.2f})  PASS".format(test_rand_omegas, test_kx_vals))
    else:
        # print(" ({:.2f},{:.2f})  FAIL".format(test_rand_omegas, test_kx_vals))
        if random_choice == 11:
            test_kx_vals = np.random.uniform(2.0, 4.0, 1)
            test_rand_omegas = np.abs(np.random.uniform(0.6, 1.5, 1))
            print(test_rand_omegas, test_kx_vals)
            # print("({:.2f},{:.2f})".format(test_rand_omegas, test_kx_vals))
        elif random_choice == 12:
            test_kx_vals = np.random.uniform(-2.0, -4.0, 1)
            test_rand_omegas = np.abs(np.random.uniform(0.6, 1.5, 1))
            print(test_rand_omegas, test_kx_vals)

            # print("({:.2f},{:.2f})".format(test_rand_omegas, test_kx_vals))


print("\n")
print("Testing else statement")
print("\n")
test_below_lamb_line_single_values_new_values(cs_val, test_kx1, test_omega1)
test_below_lamb_line_single_values_new_values(cs_val, test_kx2, test_omega2)
test_below_lamb_line_single_values_new_values(cs_val, test_kx3, test_omega3)
test_below_lamb_line_single_values_new_values(cs_val, test_kx4, test_omega4)
test_below_lamb_line_single_values_new_values(cs_val, test_kx5, test_omega5)
test_below_lamb_line_single_values_new_values(cs_val, test_kx6, test_omega6)

