import numpy as np
import build_functions

np.random.seed(20)

## TEST CASE: OMEGA VALUES
## --------------------------------------------------------------------- ##

t1 = build_functions.random_omega_value_generator(5.8, 1, 5.4, total_len=1)
print("omega {:.2f} mHz".format(t1[0]))

t2 = build_functions.random_omega_value_generator(1.5, 0.2, 5.4, total_len=1)
print("omega {:.2f} mHz".format(t2[0]))

t3 = build_functions.random_omega_value_generator(0.2, 0.1, 5.4, total_len=1)
print("omega {:.2f} mHz".format(t3[0]))

t4 = build_functions.random_omega_value_generator(1.5, 0.8, 5.4, total_len=1)
print("omega {:.2f} mHz".format(t4[0]))

### Solutions ###
# test 1: omega > N_BV -- Expected omega value should be less than N_BV.
# test 2: omega is under the lamb line -- Expected omega value is below the lamb line and less than N_BV.
# test 3: omega < 0.5 -- Expected omega value is greater than 0.5 and below N_BV.
# test 4: random case -- Random draw. All of the above apply.

# Results might vary
# omega 3.46 mHz
# omega 1.28 mHz
# omega 0.54 mHz
# omega 1.80 mHz

## --------------------------------------------------------------------- ##

## TEST CASE: HORIZONTAL WAVENUMBER VALUES
## --------------------------------------------------------------------- ##


k1 = build_functions.random_kx_value_generator(0.4, 0.2, total_len=1)
print("kx {:.2f} 1/Mm".format(k1[0]))

k2 = build_functions.random_kx_value_generator(2, 1.2, total_len=1)
print("kx {:.2f} 1/Mm".format(k2[0]))

k3 = build_functions.random_kx_value_generator(0.4, 0.1, total_len=1)
print("kx {:.2f} 1/Mm".format(k3[0]))

k4 = build_functions.random_kx_value_generator(2, 1.2, total_len=1)
print("kx {:.2f} 1/Mm".format(k4[0]))

### Solutions ###
# tests might be out of order because of the random counter
# test 1: kx < 0.5 1/Mm -- Expected kx should be greater than 0.5 1/Mm.
# test 2: positive kx is under lamb line -- All right.
# test 3: kx > -0.5 1/Mm -- Expected kx should be less than -0.5 1/Mm.
# test 4: negative kx is under lamb line -- All right.


# Results might vary
# kx 0.63 1/Mm
# kx 2.89 1/Mm
# kx -0.54 1/Mm
# kx -0.55 1/Mm

