import numpy as np
from numpy.typing import ArrayLike


# correct
def e_sat_murphy_koop(T: ArrayLike) -> ArrayLike:
    """Determines the saturation vapor pressure of water as a function of temperature using the Murphy-Koop equation

    Args
    ----
        T: ArrayLike
            Temperature (K)
    
    Returns
    -------
        ArrayLike:
            Saturation vapor pressure of water as a function of temperature (Pa)

    See Also
    --------
        Murphy, D. M., & Koop, T. (2005)
    """
    return (np.exp(54.842763 - 6763.22/T - 4.210 * np.log(T) + 0.000367 * T + np.tanh(0.0415 * (T - 218.8)) * 
        (53.878 - 1331.22/T - 9.44523 * np.log(T) + 0.014025 * T)))


# correct
def p_sat_ice_murphy_koop(T: ArrayLike) -> ArrayLike:
    """Determines the saturation vapor pressure of ice as a function of temperature using the Murphy-Koop equation

    Args
    ----
        T: ArrayLike
            Temperature (K)

    Returns
    -------
        ArrayLike:
            Saturation vapor pressure of ice as a function of temperature (Pa)
    """
    return (np.exp(9.550426 - 5723.265/T + 3.53068 * np.log(T) -  0.00728332 * T ))


# correct
def number_concentration_sat_water(T: ArrayLike) -> ArrayLike:
    """Determines the number concentration of H20 molecules at water saturation as a 
       function of temperature

    Args
    ----
        T: ArrayLike
            Temperature (K)

    Returns
    -------
        ArrayLike:
            Number concentration of saturation water as a function of temperature (m^-3)
    """
    psat = e_sat_murphy_koop(T) # Saturation vapor pressure of water (Pa)
    kB = 1.38064852e-23 # Boltzmann constant (J/K)
    return psat / (kB * T)


# correct
def mean_thermal_speed(T: ArrayLike) -> ArrayLike:
    """Determines the mean thermal speed of water molecules as a function of temperature

    Args
    ----
        T: ArrayLike
            Temperature (K)

    Returns
    -------
        ArrayLike:
            Mean thermal speed of water molecules as a function of temperature (m/s)
    """
    m = 2.9915 * 10**-26 # Mass of water molecule (kg)
    kB = 1.38064852 * 10**-23 # Boltzmann constant (J/K)
    return np.sqrt((8 * kB * T) / (np.pi * m))


# correct 
# IMPORTANT -> the below function may need to change
def latent_heat_vap_water(T: ArrayLike) -> ArrayLike:
    """Determines the latent heat of vaporization of water as a function of temperature

    Args
    ----
        T: ArrayLike
            Temperature (K)
    
    Returns
    -------
        ArrayLike:
            Latent heat of vaporization of water as a function of temperature (J/molecule)
    
    Notes:
    Watson Correlation is used and a better formula may be needed
    Note the denominator converts the latent heat of vaporization from J/kg to J/molecule
    """
    Tref = 373.15 # Reference temperature (K)
    Lref = 2256.4 * 10**3# Reference latent heat of vaporization (J/kg)
    w = 0.38 # Watson correlation parameter
    Tc = 647.096 # Critical temperature of water (K)

    return (Lref * ((Tc - T) / (Tc - Tref))**w )/(3.3428 * 10 ** 25)



def number_concentration_sat_ice(T: ArrayLike) -> ArrayLike:
    """Determines the number concentration of H20 molecules at ice saturation as a 
       function of temperature

    Args
    ----
        T: ArrayLike
            Temperature (K)

    Returns
    -------
        ArrayLike:
            Number concentration of saturation ice as a function of temperature (m^-3)
    """
    psat = p_sat_ice_murphy_koop(T) # Saturation vapor pressure of ice (Pa)
    kB = 1.38064852e-23 # Boltzmann constant (J/K)
    return psat / (kB * T)
