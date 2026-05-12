import numpy as np
import matplotlib.pyplot as plt
from functions import gen_Hinc, gen_Einc_mi_gaussian_linear_polarization
from constants import L

N = 300  # N by N pixel grid

x = np.linspace(-L / 2, L / 2, N)  # Creates all x values
z = np.linspace(-L / 2, L / 2, N)

X, Z = np.meshgrid(x, z)

pos_arr = np.zeros(
    (N * N, 3)
)  # Creates array of shape (N^2, 3) i.e. N^2 x,y,z positions.
pos_arr[:, 0] = X.ravel()  # flattens the meshgrid i.e. (N,N) -> N^2
pos_arr[:, 2] = Z.ravel()

H_field = gen_Hinc(
    pos_arr
)  # Takes N x,y,z positions and returns the incident H_field at all N points
E_field = gen_Einc_mi_gaussian_linear_polarization(pos_arr)

S = 0.5 * np.real(
    np.cross(E_field, np.conjugate(H_field))
)  # Calculates the Poynting vector at all field locations

Sz = S[:, 2].reshape(
    N, N
)  # Reshapes the flattened image into a square, then takes the z component

plt.figure()
plt.imshow(
    Sz,
    extent=[x.min(), x.max(), z.min(), z.max()],  # type: ignore
    origin="lower",
    aspect="auto",
)
plt.xlabel("x")
plt.ylabel("z")
plt.title(r"$S_z$ on the $y=0$ plane")
plt.colorbar(label=r"$S_z$")
plt.show()
# plt.savefig("Poynting.png", dpi=300)
