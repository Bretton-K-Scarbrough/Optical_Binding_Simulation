"""
Run the main optical binding simulation.

This script initializes the particle positions, solves for the induced dipole
moments using the coupled dyadic Green's function system, computes the optical
and Coulomb-like forces acting on each particle, and numerically evolves the
particle positions through time. The resulting position history is saved to
disk for later analysis and visualization.
"""

import numpy as np
from tqdm.auto import trange
from constants import (
    init_pos_arr,
    num_of_particle,
    alpha,
    gamma,
    mass,
    dt,
    maxstep,
    ΔB,
    two_dimension_restriction,
)
from functions import (
    create_G_mnij,
    create_G_mnij_scatter,
    gen_Einc_mi,
    gen_F_grad,
    coulomb_force,
    radiation_Pressure,
    spin_Force,
)


if __name__ == "__main__":
    ## NOTE: Generate positions and calculate incident field
    polarizabilities = np.full((num_of_particle, 3), alpha)

    pos_arr = init_pos_arr.copy()

    Einc_mi = gen_Einc_mi(pos_arr)

    # NOTE: Calculate dyadic Green's function and solve for polarizabilities
    G_nmij = create_G_mnij(pos_arr, polarizabilities)

    G_flattened = G_nmij.transpose(0, 2, 1, 3).reshape(
        3 * num_of_particle, 3 * num_of_particle
    )  # Transpose due to strange default tensor ordering in numpy
    E_flattened = Einc_mi.reshape(3 * num_of_particle)

    inv = np.linalg.inv(alpha * G_flattened)
    p_i = ((inv @ E_flattened) / alpha).reshape(num_of_particle, 3)

    # NOTE: Calculate scattering dyadic Green's function and find Escatterd
    G_mnij_scatter = create_G_mnij_scatter(pos_arr)

    Escattered_ni = np.einsum("nmij,mj->ni", G_mnij_scatter, p_i)
    E_tot = Escattered_ni + Einc_mi

    F_grad = gen_F_grad(pos_arr, p_i)  # type: ignore

    ## NOTE: Create arrays to hold data
    velocity_arr = np.zeros((maxstep + 1, num_of_particle, 3))
    full_pos_arr = np.zeros((maxstep + 1, num_of_particle, 3))
    forces_arr = np.zeros((maxstep + 1, num_of_particle, 3))
    rad_forces_arr = np.zeros((maxstep + 1, num_of_particle, 3))
    spin_forces_arr = np.zeros((maxstep + 1, num_of_particle, 3))
    grad_forces_arr = np.zeros((maxstep + 1, num_of_particle, 3))
    p_i_arr = np.zeros((maxstep + 1, num_of_particle, 3))
    E_n_arr = np.zeros((maxstep + 1, num_of_particle, 3))
    det_G_arr = np.zeros((maxstep + 1))
    max_G_arr = np.zeros((maxstep + 1))

    eigenvalue_arr = []

    ## NOTE: initialize
    full_pos_arr[0] = init_pos_arr
    velocity_arr[0] = np.zeros((num_of_particle, 3))

    ## NOTE: Main loop
    for step in trange(1, maxstep + 1, desc="Simulating"):
        Einc_mi = gen_Einc_mi(full_pos_arr[step - 1])
        E_flattened = E_flattened = Einc_mi.reshape(3 * num_of_particle)
        G_nmij = create_G_mnij(full_pos_arr[step - 1], polarizabilities)

        G_flattened = G_nmij.transpose(0, 2, 1, 3).reshape(
            3 * num_of_particle, 3 * num_of_particle
        )

        det_G_arr[step - 1] = np.linalg.det(G_flattened)
        max_G_arr[step - 1] = np.max(G_flattened)

        inv = np.linalg.inv(alpha * G_flattened)
        p_i = ((inv @ E_flattened) / alpha).reshape(num_of_particle, 3)

        p_i_arr[step - 1] = p_i

        G_mnij_scatter = create_G_mnij_scatter(full_pos_arr[step - 1])
        Escattered_ni = np.einsum("nmij,mj->ni", G_mnij_scatter, p_i_arr[step - 1])
        E_n_arr[step - 1] = Einc_mi + Escattered_ni

        grad_force = gen_F_grad(full_pos_arr[step - 1], p_i)
        grad_forces_arr[step - 1] = grad_force

        rad_force = radiation_Pressure(full_pos_arr[step - 1], p_i)
        rad_forces_arr[step - 1] = rad_force

        spin_force = spin_Force(full_pos_arr[step - 1], p_i)
        spin_forces_arr[step - 1] = spin_force

        forces_arr[step - 1] = grad_force + coulomb_force(full_pos_arr[step - 1]) + (0.005 * rad_force) + spin_force  # type: ignore # Rad force dominates (likely due to insane indicent power), reduced rad_force a lot for 3D simulation.

        full_pos_arr[step] = full_pos_arr[step - 1] + velocity_arr[step - 1] * dt

        # Brownian kick goes here
        randomDeltaV = (
            np.sqrt(ΔB) * np.random.normal(0, 1, (num_of_particle, 3)) / np.sqrt(2)
        )

        velocity_arr[step] = (
            velocity_arr[step - 1] * np.exp(-gamma * dt)
            # + (0.001 * randomDeltaV) # Brownian motion term
            + forces_arr[step - 1] * (dt / mass)
        )

        ## NOTE: If you want to pin particles in place below stops them from moving
        # velocity_arr[step, 0:9] = np.asarray(
        #     [0, 0, 0]
        # )  # set velocity to zero on the first particle

        ## NOTE: Restricts to 2D or 3D
        if two_dimension_restriction:
            for i in range(num_of_particle):
                forces_arr[step, i, 2] = 0
                velocity_arr[step, i, 2] = 0
                full_pos_arr[step, i, 2] = 0

    ## NOTE: SAVE DATA
    np.save("./data/position_data.npy", np.asarray(full_pos_arr))
    # np.save("./data/velocity_data.npy", np.asarray(velocity_arr))
    # np.save("./data/forces_data.npy", np.asarray(forces_arr))
    # np.save("./data/grad_force_data.npy", np.asarray(grad_forces_arr))
    # np.save("./data/rad_force_data.npy", np.asarray(rad_forces_arr))
    # np.save("./data/spin_force_data.npy", np.asarray(spin_forces_arr))
    print("saved data")
