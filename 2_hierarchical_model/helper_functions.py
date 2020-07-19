"""
Utils for plots_for_paper.ipynb 

@author Francesca Capel
@date Jan 2020
"""

import numpy as np
from scipy.integrate import quad, dblquad

Om = 0.3
Ol = 0.7
H0 = 70  # km s^-1 Mpc^-1
c = 3e5  # km s^-1
DH = c / H0  # Mpc

Mpc_to_cm = 3.086e24
m_to_cm = 100
yr_to_s = 3.154e7


def xx(z):

    return ((1 - Om) / Om) / pow(1 + z, 3)


def phi(x):

    x2 = np.power(x, 2)
    x3 = pow(x, 3)
    numerator = 1.0 + (1.320 * x) + (0.4415 * x2) + (0.02656 * x3)
    denominator = 1.0 + (1.392 * x) + (0.5121 * x2) + (0.03944 * x3)

    return numerator / denominator


def luminosity_distance(z):
    """
    Luminosity distance based on approximation used in Adachi & Kasai 2012.
    
    Units of [Mpc].
    """

    x = xx(z)
    zp = 1 + z

    A = 2 * DH * zp / np.sqrt(Om)
    B = phi(xx(0)) - ((1 / np.sqrt(zp)) * phi(x))

    return A * B


def comoving_distance(z):

    return luminosity_distance(z) / (1 + z)


def E_fac(z):

    Omp = Om * (1 + z) ** 3

    return np.sqrt(Omp + Ol)


def differential_comoving_volume(z):

    dc = comoving_distance(z)

    return (DH * dc ** 2) / E_fac(z)


def comoving_volume(z):

    return 4 * np.pi * differential_comoving_volume(z)


def n_alpha(alpha):
    """
    Factor coming from the definition of power law normalisation
    through luminosity = \int_Emin^20Emin dE dN/dEdt.
    """

    if alpha == 2:

        return 1 / np.log(20)

    else:

        return ((alpha - 2) / (alpha - 1)) / (1 - np.power(20, 2 - alpha))


def dN_dEdtdA(E, L, alpha, Emin, z):
    """
    Differential flux of a source at redshift z.
    
    Units [TeV^-1 s^-1 cm^-2]
    """

    A = L * n_alpha(alpha) * (alpha - 1) * np.power(Emin, -2)
    B = 1 / (4 * np.pi * np.power(luminosity_distance(z) * Mpc_to_cm, 2))
    return A * B * np.power(E / Emin, -alpha)


def dN_dtdA(L, alpha, Emin, z):
    """
    Flux integrated over energy from a source at redshift z.
    
    Units [s^-1 cm^-2]
    """

    dl = luminosity_distance(z) * Mpc_to_cm

    A = L * n_alpha(alpha) * np.power(1 + z, 1 - alpha) / Emin
    B = 1 / (4 * np.pi * dl ** 2)

    return A * B


def dN_dEdtdA_total(E, L, alpha, Emin, zmin, zmax, source_evolution):
    """
    The above integrated over redshift and convoluted with the source distribution.
    """

    integrand = (
        lambda z: dN_dEdtdA(E, L, alpha, Emin, z)
        * source_evolution(z)
        * differential_comoving_volume(z)
        / (1 + z)
    )

    integral, err = quad(integrand, zmin, zmax)

    return integral


def N_nu_integrand(E, cosz, L, alpha, Emin, z, t, effective_area):
    """
    Integrand for expected number of neutrinos from a single source.
    """
    A = effective_area(E * 1e3, cosz) * m_to_cm ** 2
    T = t * yr_to_s
    return A * dN_dEdtdA((1 + z) * E, L, alpha, Emin, z) * T


def P_det(z, L, alpha, Emin, T, effective_area):
    """
    Probability to detect a source at redshift z.
    """

    Nnu, err = dblquad(
        N_nu_integrand,
        -1,
        1,
        lambda x: Emin,
        lambda x: 10 * Emin,
        args=(L, alpha, Emin, z, T, effective_area),
    )
    return 1 - (np.exp(-Nnu) * (1 + Nnu))


def P_det_new(z, L, alpha, Emin, T, Aeff_integral):
    """
    Fast version withn pre-calculated Aeff_integral
    """
    Nnu = expected_neutrinos(z, L, alpha, Emin, T, Aeff_integral)
    return 1 - (np.exp(-Nnu) * (1 + Nnu))


def reduced_integrand(E, cosz, effective_area, alpha, Emin):
    """
    Minimal version of integrand with most terms factored out.
    
    :param E: energy in [TeV]
    :param cosz: cosine of the zenith angle
    :param alpha: spectral index
    """

    E_GeV = E * 1e3  # TeV -> GeV
    A = effective_area(E_GeV, cosz) * m_to_cm ** 2  # cm^2

    return A * np.power(E, -alpha)


def expected_neutrinos(z, L, alpha, Emin, T, Aeff_integral):
    """
    To match expected implementation in Stan.
    """

    time = T * yr_to_s  # s
    na = n_alpha(alpha)
    dl = luminosity_distance(z)  # Mpc

    A = L * na * (alpha - 1) / np.power(Emin, 2 - alpha)
    # TeV^-(1-alpha) s^-1
    B = np.power(1 + z, -alpha) / (4 * np.pi * np.power(dl * Mpc_to_cm, 2))
    # cm^-2
    # Aeff_integral cm^2 TeV^(1-alpha)

    return A * B * Aeff_integral * time


def zth_minimizer(z, L, phi_th):
    dl = luminosity_distance(z) * Mpc_to_cm
    rhs = L / (phi_th * 4 * np.pi)
    return dl ** 2 - rhs


def N_src_integrand(z, rise, decay, peak):

    return (
        source_evolution(z, rise, decay, peak)
        * differential_comoving_volume(z)
        / (1 + z)
    )


def expected_sources(n0, zmin, zmax, rise, decay, peak):

    integral, err = quad(N_src_integrand, zmin, zmax, args=(rise, decay, peak))

    return n0 * integral


def SFR_density(z, n0):
    """
    SFR - using the parameteristion and results from Madau (2014)
    """

    num = np.power(1 + z, 2.7)
    denom = 1 + np.power((1 + z) / 2.9, 5.6)

    return n0 * (num / denom)


def SFR_density_old(z, N0):
    """
    SFR - using the parameteristion of Cole et al. (2001) and the results of 
    Yüksel et al. (2008).
    
    :param N0: Local number density [Mpc^-3]
    """
    a = 3.4
    b = -0.3
    c = -3.5
    eta = -10
    B = 5000
    C = 9
    return N0 * (
        (1 + z) ** (a * eta) + ((1 + z) / B) ** (b * eta) + ((1 + z) / C) ** (c * eta)
    ) ** (1 / eta)


def agn_density(z, N0, model="PLE"):
    """
    From Aird et al. (2010.)
    See Equation 21 and Figure 10.
    
    :param z: Redshift
    :param N0: Local number density [Mpc^-3]
    """

    if model == "LADE":
        d = -0.19
        p1 = 6.36
        p2 = -0.24
        zc = 0.75

        N = np.exp(d * (1 + z))
        norm = np.exp(d)

    elif model == "PLE":
        N = 1.0
        norm = 1.0
        p1 = 5.55
        p2 = -1.38
        zc = 0.84

    z_a = ((1 + zc) / (1 + z)) ** p1
    z_b = ((1 + zc) / (1 + z)) ** p2

    z_a_0 = np.power(1 + zc, p1)
    z_b_0 = np.power(1 + zc, p2)

    normalisation = norm * (z_a_0 + z_b_0) ** (-1)

    return N0 * (z_a + z_b) ** (-1) / normalisation


def source_evolution(z, p1, p2, zc):

    return np.power(1 + z, p1) / np.power(1 + z / zc, p2)


def source_flux(L, gamma, z):

    dl = luminosity_distance(z) * Mpc_to_cm
    A = L / (4 * np.pi * np.power(dl, 2))
    B = np.power((1 + z), 2 - gamma)

    return A * B


def total_flux(z, L, gamma, n0, p1, p2, zc):

    Vc = comoving_volume(z)

    n = n0 * source_evolution(z, p1, p2, zc)

    phi = source_flux(L, gamma, z)

    return Vc * n * phi


def total_flux_agn(z, L, gamma, n0, model="PLE"):

    Vc = comoving_volume(z)

    n = agn_density(z, n0, model=model)

    phi = source_flux(L, gamma, z)

    return Vc * n * phi


def total_flux_sfr(z, L, gamma, n0):

    Vc = comoving_volume(z)

    n = SFR_density(z, n0)

    phi = source_flux(L, gamma, z)

    return Vc * n * phi


def total_flux_flat(z, L, gamma, n0):

    Vc = comoving_volume(z)

    phi = source_flux(L, gamma, z)

    return Vc * n0 * phi


def total_flux_neg(z, L, gamma, n0, power):

    Vc = comoving_volume(z)

    n = n0 * np.power(1 + z, -power)

    phi = source_flux(L, gamma, z)

    return Vc * n * phi


def k_gamma(gamma, Emin, Emax):

    if gamma == 2:

        output = 1 / np.log(Emax / Emin)

    else:

        output = (
            (gamma - 2)
            * np.power(Emin, gamma - 2)
            / (1 - np.power(Emin / Emax, gamma - 2))
        )

    return output


def density_from_L_phi(L, phi_norm, gamma, p1, p2, zc, Emin, Emax, Enorm):
    """
    Everything in TeV. 
    """

    kg = k_gamma(gamma, Emin, Emax)

    phi_norm = phi_norm * np.power(1 / Mpc_to_cm, -2)
    # TeV^-1 Mpc^-2 s^-1 sr^-1

    fac = phi_norm * 4 * np.pi / (DH * np.power(Enorm, -gamma) * kg * L)

    def integrand(z, p1, p2, zc, gamma):
        return source_evolution(z, p1, p2, zc) * np.power(1 + z, -gamma) / E_fac(z)

    integ, _ = quad(integrand, 0, 10, args=(p1, p2, zc, gamma))

    return fac * (1 / integ)  # Mpc^-3
