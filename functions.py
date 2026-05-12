"""
File filled with helper functions.
"""

import numpy as np
from numpy import float64, complex128
from numpy.typing import NDArray
from constants import (
    k,
    k0,
    epsilon_0,
    epsilon_b,
    E0,
    a,
    w0,
    use_circular_polarization,
    alpha_real,
    num_of_particle,
    pol_angle,
    q0,
    mu_0,
    sigma,
    omega,
)


def create_G_mnij(
    pos_arr: NDArray[float64], pol_arr: NDArray[complex128]
) -> NDArray[complex128]:
    """
    Computes the full (N, N, 3, 3) dyadic Green's tensor Gₘₙᵢⱼ for a system of N dipoles.

    For dipoles at positions `pos_arr[n]`, this function returns a 4D array where
    G[n, m, i, j] contains the dyadic Green's function tensor that describes the
    electromagnetic coupling between dipole `m` and dipole `n`. Self-interactions
    (where n == m) are set to -I / alphaₘ.

    Parameters
    ----------
    pos_arr : np.ndarray
        Array of particle positions with shape (N, 3).

    pol_arr : np.ndarray
        Array of particle polarizabilities with shape (N,).

    Returns
    -------
    np.ndarray
        The dyadic Green's tensor with shape (N, N, 3, 3), where the last two indices
        represent the tensor components and the first two index the dipole pair (n, m).
        Self-terms (n == m) are set to -1 / alpha * I₃.
    """
    N = pos_arr.shape[0]
    pos_diff_arr = pos_arr[:, None, :] - pos_arr[None, :, :]  # shape of (N, N, 3)

    R = np.linalg.norm(pos_diff_arr, axis=2)  # (N, N)
    kR = k * R  # (N, N)
    G = np.exp(1j * kR) / (4 * np.pi * R * epsilon_0 * epsilon_b)  # (N, N)
    outer_prod_arr = np.einsum(
        "...i,...j->...ij", pos_diff_arr, pos_diff_arr
    )  # (N, N, 3, 3)
    block_struct = np.eye(3)[None, None, :, :]  # (1, 1, 3, 3)

    main_diag_terms = (
        block_struct
        * G[:, :, None, None]
        * R[:, :, None, None] ** 2
        * (kR * (kR + 1j) - 1)[:, :, None, None]
    )  # (N, N, 3, 3)
    off_diag_terms = (
        G[:, :, None, None] * (-3 + kR * (kR + 3j))[:, :, None, None] * outer_prod_arr
    )  # (N, N, 3, 3)
    scaling_terms = 1 / ((R[:, :, None, None]) ** 4)
    dyadic = scaling_terms * (main_diag_terms - off_diag_terms)

    mask = np.eye(N, dtype=bool)  # (N,N)
    dyadic[mask] = (1 / pol_arr[:, None]) * np.eye(3)[None, :, :]
    return dyadic


def create_G_mnij_scatter(
    pos_arr: NDArray[float64],
) -> NDArray[complex128]:
    """
    Computes the full (N, N, 3, 3) dyadic Green's tensor Gₘₙᵢⱼ for a system of N dipoles.

    For dipoles at positions `pos_arr[n]`, this function returns a 4D array where
    G[n, m, i, j] contains the dyadic Green's function tensor that describes the
    electromagnetic coupling between dipole `m` and dipole `n`. Self-interactions
    (where n == m) are set to 0.

    Parameters
    ----------
    pos_arr : np.ndarray
        Array of particle positions with shape (N, 3).
    Returns
    -------
    np.ndarray
        The dyadic Green's tensor with shape (N, N, 3, 3), where the last two indices
        represent the tensor components and the first two index the dipole pair (n, m).
        Self-terms (n == m) are set to 0.
    """
    N = pos_arr.shape[0]
    pos_diff_arr = pos_arr[:, None, :] - pos_arr[None, :, :]  # shape of (N, N, 3)

    R = np.linalg.norm(pos_diff_arr, axis=2)  # (N, N)
    kR = k * (R)  # (N, N)
    G = np.exp(1j * kR) / (4 * np.pi * (R) * epsilon_0 * epsilon_b)  # (N, N)
    outer_prod_arr = np.einsum(
        "...i,...j->...ij", pos_diff_arr, pos_diff_arr
    )  # (N, N, 3, 3)
    block_struct = np.eye(3)[None, None, :, :]  # (1, 1, 3, 3)

    main_diag_terms = (
        block_struct
        * G[:, :, None, None]
        * R[:, :, None, None] ** 2
        * (kR * (kR + 1j) - 1)[:, :, None, None]
    )  # (N, N, 3, 3)
    off_diag_terms = (
        G[:, :, None, None] * (-3 + kR * (kR + 3j))[:, :, None, None] * outer_prod_arr
    )  # (N, N, 3, 3)
    scaling_terms = 1 / ((R[:, :, None, None]) ** 4)
    dyadic = scaling_terms * (main_diag_terms - off_diag_terms)

    mask = np.eye(N, dtype=bool)  # (N,N)
    dyadic[mask] = 0.0 * np.eye(3)[None, :, :]
    return dyadic


def create_G_field(
    src_pos_arr: NDArray[float64],  # (N, 3) source dipoles
    obs_pos_arr: NDArray[float64],  # (M, 3) observation grid points
) -> NDArray[complex128]:
    """
    Computes the dyadic Green's function Gₘₙᵢⱼ for fields at arbitrary observation points.

    Parameters
    ----------
    src_pos_arr : np.ndarray
        Positions of N source dipoles (N, 3).
    obs_pos_arr : np.ndarray
        Positions of M observation points (M, 3).

    Returns
    -------
    G_field : np.ndarray
        Dyadic Green's function array of shape (M, N, 3, 3), where
        G_field[m, n, i, j] gives the (i,j)-component of the Green’s tensor
        from dipole n to observation point m.
    """
    obs_pos_arr = np.asarray(obs_pos_arr)
    src_pos_arr = np.asarray(src_pos_arr)

    # (M, N, 3) difference vector: observation - source
    Rmn = obs_pos_arr[:, None, :] - src_pos_arr[None, :, :]

    R = np.linalg.norm(Rmn, axis=2)  # (M, N)
    kR = k * (R)

    # Free-space scalar Green’s function
    G_scalar = np.exp(1j * kR) / (4 * np.pi * (R) * epsilon_0 * epsilon_b)

    # Tensor structure
    I = np.eye(3)[None, None, :, :]  # (1,1,3,3)
    outer = np.einsum("mni,mnj->mnij", Rmn, Rmn)  # (M,N,3,3)

    # Dyadic terms
    main = (
        I
        * G_scalar[:, :, None, None]
        * (R[:, :, None, None] + a) ** 2
        * (kR * (kR + 1j) - 1)[:, :, None, None]
    )
    off = G_scalar[:, :, None, None] * (-3 + kR * (kR + 3j))[:, :, None, None] * outer

    G_field = (main - off) / ((R[:, :, None, None]) ** 4)

    return G_field


def gen_Einc_mi_gaussian_linear_polarization(
    pos_arr: NDArray[float64],
) -> NDArray[complex128]:
    """
    Generates the incident electric field.

    Parameters
    ----------
    pos_arr: np.ndarray
        Array of particle positons

    Returns
    np.ndarray
        Einc_mi with a shape of (N, 3), where N is the number of particles
    """

    N = pos_arr.shape[0]
    x = pos_arr[:, 0]
    y = pos_arr[:, 1]
    z = pos_arr[:, 2]

    Einc_mi = np.zeros((N, 3), dtype=complex128)
    Einc_mi[:, 0] = (
        np.cos(pol_angle)
        * E0
        * np.exp(1j * k * z)
        * np.exp(-(x**2 + y**2) / (2 * w0**2))
    )
    Einc_mi[:, 1] = (
        np.sin(pol_angle)
        * E0
        * np.exp(1j * k * z)
        * np.exp(-(x**2 + y**2) / (2 * w0**2))
    )

    return Einc_mi  # (N, 3)


def gen_Einc_mi_gaussian_circular_polarization(
    pos_arr: NDArray[float64],
) -> NDArray[complex128]:
    """
    Generates the incident electric field.

    Parameters
    ----------
    pos_arr: np.ndarray
        Array of particle positons

    Returns
    np.ndarray
        Einc_mi with a shape of (N, 3), where N is the number of particles
    """

    N = pos_arr.shape[0]
    x = pos_arr[:, 0]
    y = pos_arr[:, 1]
    z = pos_arr[:, 2]

    Einc_mi = np.zeros((N, 3), dtype=complex128)
    Einc_mi[:, 0] = E0 * np.exp(1j * k * z) * np.exp(-(x**2 + y**2) / (2 * w0**2))
    Einc_mi[:, 1] = 1j * E0 * np.exp(1j * k * z) * np.exp(-(x**2 + y**2) / (2 * w0**2))

    return Einc_mi  # (N, 3)


gen_Einc_mi_gaussian = (
    gen_Einc_mi_gaussian_circular_polarization
    if use_circular_polarization
    else gen_Einc_mi_gaussian_linear_polarization
)

gen_Einc_mi = gen_Einc_mi_gaussian


def gen_Hinc(pos_arr: NDArray[float64]) -> NDArray[complex128]:
    """
    Calculate the incident magnetic field for a linearly polarized Gaussian beam
    at each particle position.

    The incident magnetic field is computed from the curl of the incident
    electric field using the frequency-domain Maxwell relation

        H_inc = -(∇ × E_inc) / (i * omega * mu_0)

    assuming a time-dependence convention consistent with this sign. The
    incident electric field is modeled as a Gaussian beam propagating in the
    z-direction with transverse linear polarization set by ``pol_angle``.

    Parameters
    ----------
    pos_arr : NDArray[float64]
        Array of particle positions with shape ``(N, 3)``, where each row gives
        the ``(x, y, z)`` position of one particle.

    Returns
    -------
    NDArray[complex128]
        Complex-valued incident magnetic field with shape ``(N, 3)``. Each row
        contains the ``(Hx, Hy, Hz)`` components of the incident magnetic field
        at one particle position.

    Notes
    -----
    This function uses the global constants ``num_of_particle``, ``E0``, ``k``,
    ``w0``, ``pol_angle``, ``omega``, and ``mu_0``.

    The intermediate field components are proportional to the curl of the
    incident electric field and are divided by ``1j * omega * mu_0`` to obtain
    the magnetic field.
    """
    x = pos_arr[:, 0]
    y = pos_arr[:, 1]
    z = pos_arr[:, 2]

    constant = E0 * np.exp(1j * k * z) * np.exp(-(x**2 + y**2) / (2 * w0**2))

    Hinc_mi = np.zeros((num_of_particle, 3), dtype=complex128)
    Hinc_mi[:, 0] = -1j * k * constant * np.sin(pol_angle)
    Hinc_mi[:, 1] = 1j * k * constant * np.cos(pol_angle)
    Hinc_mi[:, 2] = (constant / w0**2) * (y * np.cos(pol_angle) - x * np.sin(pol_angle))

    Hinc_mi = Hinc_mi / (1j * omega * mu_0)

    return Hinc_mi  # (N, 3)


def gen_Hscat(
    pos_arr: NDArray[float64], dipole_arr: NDArray[complex128]
) -> NDArray[complex128]:
    """
    Calculate the scattered magnetic field at each particle position from the
    curl of the scattered electric field.

    The scattered electric field derivatives are computed using
    ``gen_dx_Escat_vec`` and then contracted with the Levi-Civita tensor to
    evaluate the curl. The magnetic field is calculated using the frequency
    domain Maxwell relation

        H_scat = -(∇ × E_scat) / (i * omega * mu_0)

    assuming a time dependence convention consistent with this sign.

    The derivative tensor follows the convention

        dx_Escat[m, j, k] = ∂E_j / ∂x_k

    where ``m`` indexes the observation particle, ``j`` indexes electric field
    components, and ``k`` indexes spatial derivative directions.

    Parameters
    ----------
    pos_arr : NDArray[float64]
        Array of particle positions with shape ``(N, 3)``, where each row gives
        the ``(x, y, z)`` position of one particle.

    dipole_arr : NDArray[complex128]
        Complex-valued induced dipole moments with shape ``(N, 3)``, where each
        row gives the dipole vector of one particle.

    Returns
    -------
    NDArray[complex128]
        Complex-valued scattered magnetic field with shape ``(N, 3)``. Each row
        gives the ``(Hx, Hy, Hz)`` components at one particle position.

    Notes
    -----
    This function constructs the Levi-Civita tensor explicitly and uses it to
    compute the curl through Einstein summation.

    It depends on ``gen_dx_Escat_vec`` and uses the global constants ``omega``
    and ``mu_0``.
    """
    eps_ijk = np.zeros((3, 3, 3), dtype=int)
    eps_ijk[0, 1, 2] = eps_ijk[1, 2, 0] = eps_ijk[2, 0, 1] = 1
    eps_ijk[0, 2, 1] = eps_ijk[2, 1, 0] = eps_ijk[1, 0, 2] = -1

    dx_Escat = gen_dx_Escat_vec(pos_arr, dipole_arr)

    Hscat = -np.einsum("mjk,ijk->mi", dx_Escat, eps_ijk) / (1j * omega * mu_0)
    return Hscat


def radiation_Pressure(
    pos_arr: NDArray[float64], dipole_arr: NDArray[complex128]
) -> NDArray[complex128]:
    """
    Calculate the radiation pressure force on each particle from the time-averaged
    Poynting vector of the total electromagnetic field.

    The total electric and magnetic fields are computed as the sum of the
    incident fields and the scattered fields produced by the induced dipoles.
    The radiation pressure force is then calculated from

        F = 0.5 * sigma * Re(E x H*)

    where ``sigma`` is an effective interaction cross section, ``E`` is the
    total electric field, and ``H*`` is the complex conjugate of the total
    magnetic field.

    Parameters
    ----------
    pos_arr : NDArray[float64]
        Array of particle positions with shape ``(N, 3)``, where each row gives
        the ``(x, y, z)`` position of one particle.

    dipole_arr : NDArray[complex128]
        Complex-valued induced dipole moments with shape ``(N, 3)``, where each
        row gives the dipole vector of one particle.

    Returns
    -------
    NDArray[complex128]
        Array of radiation pressure force vectors with shape ``(N, 3)``. Each
        row contains the force components acting on one particle.

    Notes
    -----
    This function depends on the helper functions ``gen_Escat``,
    ``gen_Einc_mi_gaussian_linear_polarization``, ``gen_Hscat``, and
    ``gen_Hinc``.

    It also uses the global constant ``sigma``.
    """
    E_field = gen_Escat(pos_arr, dipole_arr) + gen_Einc_mi_gaussian_linear_polarization(
        pos_arr
    )
    H_field = gen_Hscat(pos_arr, dipole_arr) + gen_Hinc(pos_arr)

    force = 0.5 * sigma * np.real(np.cross(E_field, np.conjugate(H_field)))

    return force


def spin_Force(
    pos_arr: NDArray[float64], dipole_arr: NDArray[complex128]
) -> NDArray[complex128]:
    """
    Calculate the spin-curl force on each particle from the total electric field.

    The total electric field is computed as the sum of the incident Gaussian
    linearly polarized field and the scattered field produced by the induced
    dipoles. The spatial derivative of the total field is then used to evaluate
    the spin-force contribution at each particle position.

    The derivative tensor follows the convention

        dx_Etot[n, i, j] = ∂E_i / ∂x_j

    where ``n`` indexes particles, ``i`` indexes electric field components, and
    ``j`` indexes spatial derivative directions.

    Parameters
    ----------
    pos_arr : NDArray[float64]
        Array of particle positions with shape ``(N, 3)``, where each row gives
        the ``(x, y, z)`` position of one particle.

    dipole_arr : NDArray[complex128]
        Complex-valued induced dipole moments with shape ``(N, 3)``, where each
        row gives the dipole vector of one particle.

    Returns
    -------
    NDArray[complex128]
        Array of spin-force vectors with shape ``(N, 3)``. Each row contains the
        force components acting on one particle.

    Notes
    -----
    This function depends on the helper functions
    ``gen_Einc_mi_gaussian_linear_polarization``, ``gen_Escat``,
    ``gen_dx_Einc_gaussian_linear_polarization``, and ``gen_dx_Escat_vec``.

    It also uses the global constants ``sigma``, ``epsilon_0``, and ``k0``.
    The final force is scaled by the empirical factor ``1.50003637e11``.
    """
    Einc = gen_Einc_mi_gaussian_linear_polarization(pos_arr)
    Escat = gen_Escat(pos_arr, dipole_arr)

    Etot = Einc + Escat

    dx_Einc = gen_dx_Einc_gaussian_linear_polarization(pos_arr)
    dx_Escat = gen_dx_Escat_vec(pos_arr, dipole_arr)
    dx_Etot = dx_Einc + dx_Escat

    # print(f"{Etot.shape=}")
    # print(f"{dx_Etot.shape=}")

    force = (
        sigma
        * 0.5
        * np.real(
            1j * epsilon_0 / k0 * np.einsum("nij,nj->ni", np.conjugate(dx_Etot), Etot)
        )
    )

    force = force * 1.50003637e11

    return force


def gen_Escat(
    pos_arr: NDArray[float64], dipole_arr: NDArray[complex128]
) -> NDArray[complex128]:
    """
    Generates the scattered E-field by taking the dot product between the
    position array and the polarizability array.

    Parameters
    ----------
    pos_arr: np.ndarray
        Array of particle positons
    dipole_arr: np.ndarray
        Array of particle polarizabilities

    Returns
    np.ndarray
        Escat_mi with a shape of (N, 3), where N is the number of particles
    """
    G_mnij = create_G_mnij_scatter(pos_arr)
    return np.einsum("nmij,mj->ni", G_mnij, dipole_arr)


def gen_dx_Escat_vec(
    pos_arr: NDArray[float64], dipole_arr: NDArray[complex128]
) -> NDArray[complex128]:
    """
    Compute the spatial derivative of the scattered electric field from a set of
    point dipoles using the free-space (background-medium) scalar Green's function.

    This routine evaluates the rank-3 tensor
        dx_Escat[m, i, l] = ∂E_scat,i(r_m) / ∂x_l
    where the scattered field at observation point r_m is produced by dipoles at
    source points r_n with dipole moments p_n (given by ``dipole_arr``). The
    computation uses analytic derivatives of the Helmholtz Green's function
        G(r) = exp(i k r) / (4π r)
    and contracts the resulting interaction tensor with the dipole moments.

    Parameters
    ----------
    pos_arr : NDArray[np.float64]
        Particle positions with shape (N, 3). ``pos_arr[m]`` is the observation
        location r_m and ``pos_arr[n]`` is the source location r_n.
        Units must be consistent with ``k`` (e.g., meters).
    dipole_arr : NDArray[np.complex128]
        Dipole moments / polarizations with shape (N, 3). ``dipole_arr[n, j]`` is
        the j-th component of the dipole moment p_n. Units must be consistent
        with your Green's-function convention and the scaling by
        ``epsilon_0 * epsilon_b``.

    Returns
    -------
    dx_Escat : NDArray[np.complex128]
        Array of shape (N, 3, 3) giving the field gradient at each particle:
        ``dx_Escat[m, i, l] = ∂E_scat,i(r_m) / ∂x_l``.
        The first index selects the observation particle, the second the field
        component, and the third the spatial-derivative component.

    Notes
    -----
    - Self-interactions (m == n) are excluded by masking the diagonal of the
      pairwise interaction tensor to zero.
    - This implementation assumes globally defined constants:
        ``num_of_particle`` (N), ``k`` (wavenumber), ``epsilon_0``,
        ``epsilon_b`` (background relative permittivity or dielectric factor),
      and a CuPy-compatible namespace ``np``.
    - The formula involves powers of 1/r; very small separations can lead to
      numerical blow-up. Enforcing a minimum separation or adding regularization
      is often necessary in dynamics.
    """
    π = np.pi
    N = num_of_particle

    xi = pos_arr[:, None, :] - pos_arr[None, :, :]  # shape of (N, N, 3) (m, n, i)
    r = np.linalg.norm(
        xi, axis=2
    )  # shape of (N, N) (m, n) # give 2 particle indices ill give you the distance
    kr = k * r  # (m, n)
    r_sq = r**2  # (m, n)

    G = np.exp(1j * kr) / (4 * π * r)  # (m, n)

    kron = np.eye(3)
    xiδjl = np.einsum("mni,jl->mnijl", xi, kron)
    xjδil = np.einsum("mnj,il->mnijl", xi, kron)
    xlδij = np.einsum("mnl,ij->mnijl", xi, kron)
    xixjxl = np.einsum("mni,mnj,mnl->mnijl", xi, xi, xi)

    # make everything a rank 5 tensor
    r = r[:, :, None, None, None]
    kr = kr[:, :, None, None, None]
    r_sq = r_sq[:, :, None, None, None]
    G = G[:, :, None, None, None]

    kron_terms = (r_sq * (kr**2 + 3j * kr - 3)) * (xiδjl + xjδil + xlδij)

    dxldxjdxiG = (-G / r**6) * (
        kron_terms + xixjxl * (1j * kr**3 - 6 * kr**2 - 15j * kr + 15)
    )

    full_terms = (G * (k**2) * (r**-2) * xlδij * (1j * kr - 1) + dxldxjdxiG) / (
        epsilon_0 * epsilon_b
    )

    mask = np.eye(N, N, dtype=bool)  # creates identity boolean mask
    mask = mask[:, :, None, None, None]  # resizes mask to (m,n,i,j,l)

    masked_full_terms = np.where(mask, 0, full_terms)
    dx_Escat = np.einsum("mnijl,nj->mil", masked_full_terms, dipole_arr)
    return dx_Escat  # (N, 3, 3)


def gen_dx_Einc_gaussian_linear_polarization(
    pos_arr: NDArray[float64],
) -> NDArray[complex128]:
    """
    Calculate the spatial derivatives of the incident Gaussian electric field
    with linear polarization at each particle position.

    The returned tensor stores the derivative of each electric field component
    with respect to each spatial coordinate. For particle ``n``,

        dx_Einc[n, i, j] = ∂E_i / ∂x_j

    where ``i`` indexes the electric field component ``(Ex, Ey, Ez)`` and
    ``j`` indexes the spatial derivative direction ``(x, y, z)``.

    The incident field is assumed to have a Gaussian transverse profile,
    propagation phase ``exp(1j * k * z)``, and linear polarization in the
    transverse ``x-y`` plane. The polarization direction is set by
    ``pol_angle``, where the field is weighted by ``cos(pol_angle)`` in the
    x-direction and ``sin(pol_angle)`` in the y-direction.

    Parameters
    ----------
    pos_arr : NDArray[float64]
        Array of particle positions with shape ``(N, 3)``, where each row is
        the ``(x, y, z)`` position of one particle.

    Returns
    -------
    NDArray[complex128]
        Complex-valued derivative tensor with shape ``(N, 3, 3)``. The first
        axis indexes particles, the second axis indexes electric field
        components, and the third axis indexes spatial derivative directions.

    Notes
    -----
    This function uses the global constants ``num_of_particle``, ``E0``, ``k``,
    ``w0``, and ``pol_angle``.
    """
    N = num_of_particle

    dx_Einc = np.zeros((N, 3, 3), dtype=complex128)

    for n in range(N):
        x, y, z = pos_arr[n]
        coeff = E0 * np.exp(1j * k * z) * np.exp(-(x**2 + y**2) / (2 * w0**2))

        matrix_term = np.zeros((3, 3), dtype=complex128)
        matrix_term[0, 0] = -np.cos(pol_angle) * (w0**-2) * x
        matrix_term[0, 1] = -np.cos(pol_angle) * (w0**-2) * y
        matrix_term[0, 2] = 1j * k

        matrix_term[1, 0] = -np.sin(pol_angle) * (w0**-2) * x
        matrix_term[1, 1] = -np.sin(pol_angle) * (w0**-2) * y
        matrix_term[1, 2] = 1j * k
        dx_Einc[n] = coeff * matrix_term

    return dx_Einc


def gen_dx_Einc_gaussian_circular_polarization(
    pos_arr: NDArray[float64],
) -> NDArray[complex128]:
    """
    Calculate the spatial derivatives of the incident Gaussian electric field
    with circular polarization at each particle position.

    The returned tensor stores the derivative of each electric field component
    with respect to each spatial coordinate. For particle ``n``,

        dx_Einc[n, i, j] = ∂E_i / ∂x_j

    where ``i`` indexes the electric field component ``(Ex, Ey, Ez)`` and
    ``j`` indexes the spatial derivative direction ``(x, y, z)``.

    The incident field is assumed to have a Gaussian transverse profile,
    propagation phase ``exp(1j * k * z)``, and circular polarization in the
    transverse ``x-y`` plane.

    Parameters
    ----------
    pos_arr : NDArray[float64]
        Array of particle positions with shape ``(N, 3)``, where each row is
        the ``(x, y, z)`` position of one particle.

    Returns
    -------
    NDArray[complex128]
        Complex-valued derivative tensor with shape ``(N, 3, 3)``. The first
        axis indexes particles, the second axis indexes electric field
        components, and the third axis indexes spatial derivative directions.

    Notes
    -----
    This function uses the global constants ``num_of_particle``, ``E0``, ``k``,
    and ``w0``.
    """
    N = num_of_particle

    dx_Einc = np.zeros((N, 3, 3), dtype=complex128)

    for n in range(N):
        x, y, z = pos_arr[n]
        coeff = E0 * np.exp(1j * k * z) * np.exp(-(x**2 + y**2) / (2 * w0**2))

        matrix_term = np.zeros((3, 3), dtype=complex128)
        matrix_term[0, 0] = -(w0**-2) * x
        matrix_term[0, 1] = -(w0**-2) * y
        matrix_term[0, 2] = 1j * k

        matrix_term[1, 0] = -1j * (w0**-2) * x
        matrix_term[1, 1] = -1j * (w0**-2) * y
        matrix_term[1, 2] = 1j * k

        dx_Einc[n] = coeff * matrix_term

    return dx_Einc


gen_dx_Einc_gaussian = (
    gen_dx_Einc_gaussian_circular_polarization
    if use_circular_polarization
    else gen_dx_Einc_gaussian_linear_polarization
)
gen_dx_Einc = gen_dx_Einc_gaussian


def gen_F_grad(
    pos_arr: NDArray[float64], dipole_arr: NDArray[complex128]
) -> NDArray[complex128]:
    """
    Compute the optical gradient force on each particle from the total electric field.

    This function forms the total field at each particle as the sum of an incident
    field and a scattered field,
        E_tot = E_inc + E_scat,
    along with their spatial derivatives (field Jacobians),
        ∂E_tot,i / ∂x_l = ∂E_inc,i / ∂x_l + ∂E_scat,i / ∂x_l,
    and evaluates the gradient-force expression (component-wise)
        F_grad,l ∝ Re{ E_tot · (∂E_tot/∂x_l)* }.

    In this implementation the force is computed as
        F_grad = (alpha_real / 4) * Re[ E_tot · (∇E_tot)* + E_tot* · (∇E_tot) ]
    which is an explicitly real form obtained by adding the complex-conjugate
    product so the result is real-valued up to numerical noise.

    Parameters
    ----------
    pos_arr : NDArray[np.float64]
        Particle positions with shape (N, 3).
    dipole_arr : NDArray[np.complex128]
        Dipole moments / polarizations with shape (N, 3), used to generate the
        scattered field and its spatial derivatives.

    Returns
    -------
    F_grad : NDArray[np.complex128]
        Gradient force array with shape (N, 3). In typical usage this is purely
        real (imaginary part ~ 0) because the expression takes ``np.real(...)``.
        Each row corresponds to the Cartesian force components on a particle.

    Notes
    -----
    - Depends on the following helper functions and globals:
        * ``gen_Einc_mi``: incident field E_inc(r_n), shape (N, 3)
        * ``gen_dx_Einc``: incident field gradient, shape (N, 3, 3)
        * ``gen_Escat``: scattered field from dipoles, shape (N, 3)
        * ``gen_dx_Escat_vec``: scattered field gradient, shape (N, 3, 3)
        * ``alpha_real``: real part of polarizability used for the gradient force
    - Uses Einstein summation via ``np.einsum("nj,njl->nl", ...)``:
        index ``j`` contracts field components, and ``l`` indexes the spatial
        derivative direction, producing a force component for each l.
    """
    Einc = gen_Einc_mi(pos_arr)  # (N, 3)
    dx_Einc = gen_dx_Einc(pos_arr)  # (N, 3, 3)

    Escat = gen_Escat(pos_arr, dipole_arr)  # (N, 3)
    dx_Escat = gen_dx_Escat_vec(pos_arr, dipole_arr)  # (N, 3, 3)

    prod_rule1 = np.einsum("nj,njl->nl", Escat, dx_Escat.conj()) + np.einsum(
        "nj,njl->nl", Escat.conj(), dx_Escat
    )

    prod_rule2 = np.einsum("nj,njl->nl", Escat, dx_Einc.conj()) + np.einsum(
        "nj,njl->nl", Einc.conj(), dx_Escat
    )

    prod_rule3 = np.einsum("nj,njl->nl", Einc, dx_Escat.conj()) + np.einsum(
        "nj,njl->nl", Escat.conj(), dx_Einc
    )

    prod_rule4 = np.einsum("nj,njl->nl", Einc, dx_Einc.conj()) + np.einsum(
        "nj,njl->nl", Einc.conj(), dx_Einc
    )

    F_grad = (alpha_real / 4) * np.real(
        prod_rule1 + prod_rule2 + prod_rule3 + prod_rule4
    )

    return F_grad


def coulomb_force(pos_arr: NDArray[float64]):
    """
    Calculate the net Coulomb-like force on each particle from pairwise
    interactions with all other particles.

    The pairwise displacement tensor is computed as

        r_nm = r_n - r_m

    where ``n`` indexes the particle receiving the force and ``m`` indexes the
    source particle. The force contribution is then calculated using an inverse
    square-law form written as

        F_nm = q0 * r_nm / |r_nm|^3

    and summed over all source particles.

    Parameters
    ----------
    pos_arr : NDArray[float64]
        Array of particle positions with shape ``(N, 3)``, where each row gives
        the ``(x, y, z)`` position of one particle.

    Returns
    -------
    NDArray[float64]
        Array of net Coulomb-like force vectors with shape ``(N, 3)``. Each row
        contains the total force acting on one particle.

    Notes
    -----
    The diagonal terms, corresponding to self-interactions, are regularized by
    adding an identity matrix to the pairwise distance matrix before division.

    This function uses the global constants ``num_of_particle`` and ``q0``.
    """

    pos_diff_arr = pos_arr[:, None, :] - pos_arr[None, :, :]  # (N, N, 3)
    R = np.linalg.norm(pos_diff_arr, axis=2) + np.eye(num_of_particle)  # (N, N)

    gradient_array = q0 * pos_diff_arr / (R[:, :, None]) ** 3

    coul_force = np.sum(gradient_array, axis=1)  # (num_of_particle, 3)

    return coul_force


if __name__ == "__main__":
    print("Hello World")
