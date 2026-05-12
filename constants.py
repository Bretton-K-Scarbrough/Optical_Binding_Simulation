import numpy as np

use_circular_polarization = False
pol_angle = 45

# Units
microsecond = 1.0
microgram = 1.0
nanometer = 1.0
femtocoulomb = 1.0
kelvin = 1.0

second = microsecond * 1e6
kg = microgram * 1e9
meter = nanometer * 1e9
coulomb = femtocoulomb * 1e15
joule = kg * meter**2 * (second**-2)
volt = joule / coulomb
farad = coulomb / volt
ampere = coulomb / second
watt = joule / second
henry = second**2 / farad


# Fundamental Constants
pi = np.pi
epsilon_0 = 8.854187817e-12 * farad / meter
mu_0 = pi * (4e-7) * henry / meter
eta_0 = np.sqrt(mu_0 / epsilon_0)
c = 1.0 / (np.sqrt(epsilon_0 * mu_0))
kB = 1.38e-23 * kg * meter**2 / ((second**2) * kelvin)

# Colloid Properties
a = 75e-9 * meter
vol = pi * a**3 * (4 / 3)
rhoSIO2 = 2320 * kg / (meter**3)
mass = vol * rhoSIO2
mu = 1.6e-3 * kg / (meter * second)
gamma = 6.0 * pi * mu * a / mass
T = 300 * kelvin

# Laser Properties
lam = 600e-9 * meter
area = 100 * (1e-6 * meter) ** 2
power = 20 * watt  # Ludicrously high, lower for future simulations
I_0 = (
    power / area * 0.1
)  # I_0 = power / area  # Salandrino's original value, physically absurd

# Polarizability
# epsilon_p = -3 * (1.33**2)
# epsilon_p = -3
epsilon_p = 2.1 * (1.33) ** 2
mu_p = 1.0
epsilon_b = 1
mu_b = 1.0
eta_b = np.sqrt((mu_b / epsilon_b))
k0 = 2.0 * pi / lam
omega = k0 * c
k = k0 * np.sqrt(epsilon_b)
alpha_sr = complex(
    4.0
    * pi
    * epsilon_0
    * epsilon_b
    * a**3
    * ((epsilon_p - epsilon_b) / (epsilon_p + 2 * epsilon_b))
)
alpha_s = ((1 / alpha_sr) - 1j * (k**3 / (6 * pi * epsilon_0 * epsilon_b))) ** -1
alpha = alpha_s
alpha_real = alpha_s.real
alpha_imag = alpha_s.imag

sigma = k0 * alpha_imag / epsilon_0 * np.sqrt(epsilon_0 * mu_0)

# Physical Setup
L = 10 * lam  # Window size

r = lam / 2  # Particle distance arbitrarily chosen

init_pos_arr = np.asarray(
    [
        [0, r, 0],
        [r, 0, 0],
        [-r, 0, 0],
        [0, -r, 0],
    ]
)

# init_pos_arr = np.load(
#     "./data/Data for Thesis/Fixed Placements/plus and big square/position_data_1.npy"
# )[-1] # Grabs the final position from a saved data set.

np.random.seed(2002)
# additional_pos_arr = np.random.uniform(low=-L / 2, high=L / 2, size=(20, 3))
additional_pos_arr = np.asarray(
    [
        [-8 * r, 6 * r, 0],
        [-8 * r, 5 * r, 0],
    ]
)
init_pos_arr = np.vstack((init_pos_arr, additional_pos_arr))

num_of_particle = init_pos_arr.shape[0]

init_pos_arr[:, 2] = 0

w0 = 40 * lam
E0 = np.sqrt(2 * eta_0 * eta_b * I_0)
q0 = 3e-5

dt = 1
Γ = 2 * gamma * kB * T / mass
ΔB = Γ * dt
# maxstep = 100000 * 2
maxstep = 100000

if __name__ == "__main__":
    print(f"{I_0=}")
    print(f"{E0=}")
    print(f"{epsilon_0=}")
    print(f"{epsilon_p=}")
    print(f"{epsilon_b=}")
    print(f"{alpha_sr=}")
    print(f"{alpha_s=}")
    print(f"{sigma=}")
