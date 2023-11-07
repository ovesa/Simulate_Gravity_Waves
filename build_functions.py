import random

import numpy as np


def wave_packet_amplitude(
    x_array,
    init_x0,
    cgx,
    cgz,
    time_array,
    sigma_x,
    z_array,
    init_z0,
    sigma_z,
    arb_amp=1,
):
    """Compute the amplitude of a 2D traveling gravity wave packet in the X-Z domain through time.
    Code does not use the spatial y component, but can be tweaked to include it.

    Arguments:
        x_array -- Spatial x array [Mm].
        init_x0 -- Initial horizontal position of wavepacket at t=0 [Mm].
        cgx -- Horizontal group velocity [Mm/s].
        cgz -- Vertical group velocity [Mm/s].
        time_array -- Time array [s].
        sigma_x -- Physical width and depth of wave packet in x-plane [Mm].
        z_array -- Array in z-plane [Mm].
        init_z0 -- Initial vertical position of wavepacket at t=0 [Mm].
        sigma_z -- Physical width and depth of wave packet in z-plane [Mm].
        arb_amp -- An arbitrary amplitude value to give the wave packet. Default value is 1 Mm if no
                    value is passed [Mm].

    Returns:
        The amplitude of a traveling gravity wave packet [Mm].
    """

    ampx = (x_array - init_x0 - cgx * time_array) ** 2 / sigma_x ** 2
    ampz = (z_array - init_z0 - cgz * time_array) ** 2 / sigma_z ** 2
    return arb_amp * np.exp(-(ampx + ampz))


def isothermal_N(surface_gravity_value, cs_val, gamma=5 / 3):
    """Computes the isothermal Brunt-Vaisaila frequency [Hz].

    Arguments:
        surface_gravity_value -- The gravity value at the Sun's surface [Mm/s^2].
        cs_val -- The adiabatic sound speed in the atmosphere [Mm/s].
        gamma -- The adiabatic index of an ideal gas is 5.0/3.0.
    Returns:
        Isothermal Brunt-Vaisaila frequency in angular frequency [Hz].
    """
    return (surface_gravity_value / cs_val) * np.sqrt((gamma - 1))


def a_term(
    omega_val, omega_ac_val, cs_val, kx_val, tau_val, N_BV, surface_gravity_value
):
    """ Used in the computation of the vertical wavenumber for an isothermal stratified atmosphere
    with a constant radiative damping term.

    Arguments:
        omega_val -- Angular frequency of the wavepacket [Hz].
        omega_ac_val -- Acoustic cut-off frequency. The units are in angular frequency [Hz].
        cs_val -- The adiatic sound speed in the atmosphere [Mm/s].
        kx_val -- The horizontal wavenumber of the wavepacket [1/Mm].
        tau_val -- The radiative damping time of the wavepacket [s].
        N_BV -- The isothermal Brunt-Vaisaila frequency. Units are in angular frequency [Hz].
        surface_gravity_value -- The gravity value at the Sun's surface [Mm/s^2].

    Returns:
        Returns in the "a term" in the computation of the vertical wavenumber.
        See the function compute_kz.
    """
    first_term = (omega_val ** 2 - omega_ac_val ** 2) / cs_val ** 2
    second_term = (kx_val ** 2) * ((N_BV ** 2 / omega_val ** 2) - 1)
    third_term = (1) / (1 + omega_val ** 2 * tau_val ** 2)
    fourth_term = (N_BV ** 2 / omega_val ** 2) * (
        kx_val ** 2 - (omega_val ** 4 / surface_gravity_value ** 2)
    )
    return first_term + second_term - third_term * fourth_term


def b_term(omega_val, tau_val, N_BV, kx_val, surface_gravity_value):
    """ Used in the computation of the vertical wavenumber for an isothermal stratified atmosphere
    with a constant radiative damping term.
    Arguments:
        omega_val -- Angular frequency of the wavepacket [Hz].
        tau_val -- The radiative damping time of the wavepacket [s].
        N_BV -- The isothermal Brunt-Vaisaila frequency. Units are in angular frequency [Hz].
        kx_val -- The horizontal wavenumber of the wavepacket [1/Mm].
        surface_gravity_value -- The gravity value at the Sun's surface [Mm/s^2].

    Returns:
        Returns in the "b term" in the computation of the vertical wavenumber.
        See the function compute_kz.

    """
    first_term = (omega_val * tau_val) / (1 + omega_val ** 2 * tau_val ** 2)
    second_term = (N_BV ** 2 / omega_val ** 2) * (
        kx_val ** 2 - (omega_val ** 4 / surface_gravity_value ** 2)
    )
    return first_term * second_term


def compute_kz(a_term, b_term):
    """ Computes the vertical wavenumber of the gravity wavepacket.

    Arguments:
        a_term -- Computed from the function a_term.
        b_term -- Computed from the function b_term.

    Returns:
        The vertical wavenumber [1/Mm].
    """
    first_term = a_term / 2
    second_term = (np.sqrt(a_term ** 2 + b_term ** 2)) / 2
    return -1 * np.sqrt(first_term + second_term)


def compute_group_velocity(omega_val, kz_val, kx_val, N_BV, cs_val):
    """ Computes the components of the group velocity. The group velocity carries the
    energy of the wavepacket. The gravity wavepacket moves in the direction of the group
    velocity.

    Arguments:
        omega_val -- Angular frequency of the wavepacket [Hz].
        kz_val -- The vertical wavenumber of the wavepacket [1/Mm].
        kx_val -- The horizontal wavenumber of the wavepacket [1/Mm].
        N_BV -- The isothermal Brunt-Vaisaila frequency. Units are in angular frequency [Hz].
        cs_val -- The adiatic sound speed in the atmosphere [Mm/s].

    Returns:
        The horizontal and vertical components of the group velocity for a gravity wavepacket [Mm/s].
    """
    vertical = (-1 * omega_val * kz_val) / (
        (N_BV ** 2 * kx_val ** 2 / omega_val ** 2) - (omega_val ** 2 / cs_val ** 2)
    )
    horizontal_t1 = kx_val / omega_val
    horizontal_t2 = (N_BV ** 2 - omega_val ** 2) / (
        (N_BV ** 2 * kx_val ** 2 / omega_val ** 2) - (omega_val ** 2 / cs_val ** 2)
    )
    horizontal = horizontal_t1 * horizontal_t2
    return horizontal, vertical


def compute_phase_velocity(N_BV, omega_val, kx_val, omega_ac_val, cs_val, kz_val):
    """ Computes the components of the phase velocity of the gravity wave packet. The phase velocity
    and the group velocity are orthogonal.

    Arguments:
        N_BV -- The isothermal Brunt-Vaisaila frequency. Units are in angular frequency [Hz].
        omega_val -- Angular frequency of the wavepacket [Hz].
        kx_val -- The horizontal wavenumber of the wavepacket [1/Mm].
        omega_ac_val -- Acoustic cut-off frequency. The units are in angular frequency [Hz].
        cs_val -- The adiatic sound speed in the atmosphere [Mm/s].
        kz_val -- The vertical wavenumber of the wavepacket [1/Mm].

    Returns:
        The horizontal and vertical components of the phase velocity for a gravity wavepacket [Mm/s].
    """
    vertical = (
        ((N_BV ** 2 / omega_val ** 2) - 1) * (kx_val ** 2 / omega_val ** 2)
        - ((omega_ac_val ** 2 / omega_val ** 2) - 1) * (1 / cs_val ** 2)
    ) ** (-1 / 2)
    horizontal = (
        (omega_val ** 2 * (N_BV ** 2 - omega_val ** 2))
        / (
            (omega_ac_val ** 2 - omega_val ** 2) * (omega_val ** 2 / cs_val ** 2)
            + omega_val ** 2 * kz_val ** 2
        )
    ) ** (1 / 2)
    return horizontal, vertical


def compute_scale_height_exp(isothermal_H_val, a_term, b_term):
    """The scale height exponent used in the computation of the
    vertical displacement of the gravity wavepacket. Currently,
    it is not used.

    Arguments:
        isothermal_H_val -- The isothermal scale height parameter of the atmosphere [Mm].
        a_term -- Computed from the function a_term.
        b_term -- Computed from the function b_term.

    Returns:
        The scale height exponent [Mm].
    """
    first_term = 1 / (2 * isothermal_H_val)
    second_term = (-1 * a_term) / 2
    third_term = (np.sqrt(a_term ** 2 + b_term ** 2)) / 2
    return first_term - np.sqrt(second_term + third_term)


def compute_scale_height_H(cs_val, surface_gravity_value, gamma=5.0 / 3.0):
    """Isothermal scale height of the atmosphere.

    Arguments:
        cs_val -- The adiatic sound speed in the atmosphere [Mm/s].
        surface_gravity_value -- The gravity value at the Sun's surface [Mm/s^2].
        gamma -- The adiabatic index of an ideal gas is 5.0/3.0.

    Returns:
        Isothermal scale height of the atmosphere [Mm].
    """
    return cs_val ** 2 / (gamma * surface_gravity_value)


def perturbation(amp, kx_val, x_array, kz_val, z_array, omega_val, time_array):
    """ Computes the vertical displacement or perturbation induced from a propagating gravity wavepacket.

    Arguments:
        amp -- The amplitude of the gravity wavepacket. Computed from the function wave_packet_amplitude.
        kx_val -- The horizontal wavenumber of the wavepacket [1/Mm].
        x_array -- Spatial x array [Mm].
        kz_val -- The vertical wavenumber of the wavepacket [1/Mm].
        z_array -- Array in z-plane [Mm]
        omega_val -- Angular frequency of the wavepacket [Hz].
        time_array -- Time array [s].

    Returns:
        Vertical displacement or perturbation of a traveling gravity wavepacket.
    """
    return amp * np.cos(
        kx_val * x_array + kz_val * z_array - omega_val * time_array 
    )


def Lamb_equation(cs_val, kx_val):
    """Computes the Lamb equation, w = cs*kx_val. Inputs can be arrays.

    Arguments:
        cs_val -- The adiatic sound speed in the atmosphere [Mm/s].
        kx_val -- The horizontal wavenumber of the wavepacket [1/Mm].

    Returns:
        The corresponding angular frequency is returned [Hz].

    """
    return cs_val * kx_val


def recalculate_omega_val_kx(cs_val, omega_val, kx_val):
    """Given an angular frequency and horizontal wavenumber, test if
        if the values are below the Lamb Line given by w = k_x*c_s. If not,
    new values that match the condition are given.

    Arguments:
        cs_val -- The adiatic sound speed in the atmosphere [Mm/s].
        omega_val --  Angular frequency of the wavepacket [Hz].
        kx_val -- The horizontal wavenumber of the wavepacket [1/Mm].

    Returns:
        The same omega_val and kxval are returned if statement is true.
        If not, a different omega_val and kxval that is below the Lamb
        Line is returned.
    """
    random_choice = random.choice([11, 12])

    if (omega_val <= Lamb_equation(cs_val, kx_val)) or (
        omega_val <= -1 * Lamb_equation(cs_val, kx_val)
    ):
        return omega_val, kx_val
    else:
        omega_val = np.abs(np.random.uniform(0.6, 1.5, 1))
        omega_val = (omega_val / 1000) * (2 * np.pi)
        if random_choice == 11:
            kx_val = np.abs(np.random.uniform(2.0, 4.0, 1))
            return omega_val, kx_val
        elif random_choice == 12:
            kx_val = np.random.uniform(-2.0, -4.0, 1)
            return omega_val, kx_val


def random_kx_value_generator(mu, sigma, total_len=1):
    """Generates a random horizontal wavenumber from a gaussian distribution
    given a mean and sigma value. There's a random integer generator that
    determines if the horizontal wavenumber will be negative or positive.

    Arguments:
        mu -- Mean horizontal value [1/Mm].
        sigma -- Standard deviation around mean horizontal wavenumber [1/Mm].
        total_len -- The amount of values that random gaussian number generator should produce. Default is 1.

    Returns:
        A positive or negative initial horizontal wavenumber [1/Mm].
    """
    random_choice = random.choice([3, 4])

    # positive initial horizontal wavenumber
    if random_choice == 3:
        kx_val = np.random.normal(mu, sigma, total_len)

        while kx_val < 0.5:
            kx_val = np.random.normal(mu, sigma, total_len)

        return kx_val

    # negative initial horizontal wavenumber
    elif random_choice == 4:
        kx_val = np.random.normal(-1 * mu, sigma, total_len)

        while kx_val > -0.5:
            kx_val = np.random.normal(-1 * mu, sigma, total_len)

        return kx_val


def random_omega_value_generator(mu, sigma, N_BV, total_len=1):
    """ Generates a random wavepacket frequency (omega) from a gaussian distribution
    given a mean and sigma value. The wavepacket frequency will always be positive.

    Arguments:
        mu -- Mean frequency of wavepacket [mHz].
        sigma -- Standard deviation around mean wavepacket frequency [mHz].
        N_BV -- The Brunt-Vaisaila frequency. The gravity wavepackets should be less than the Brunt-Vaisaila frequency [mHz].
        total_len -- The amount of values that random gaussian number generator should produce. Default is 1.

    Returns:
        The cyclic frequency of a wavepacket [mHz].
    """

    # On each iteration, choose a wavepacket frequency from the defined normal distribution
    rand_omega = np.abs(
        np.random.normal(mu, sigma, total_len)
    )  # Wavepacket frequency confined to positive values [mHz]

    # ensure that omega is greater than 0.5 mHz
    # Value should be below the Brunt-Vaisaila frequency given by the chosen_N parameter
    while (rand_omega < 0.5) or (rand_omega >= N_BV):
        rand_omega = np.abs(np.random.normal(mu, sigma, total_len))

    return rand_omega
