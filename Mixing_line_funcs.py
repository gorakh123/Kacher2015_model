import numpy as np
from Thermodynamic_funcs import e_sat_murphy_koop, latent_heat_vap_water, p_sat_ice_murphy_koop
from numpy.typing import ArrayLike
from scipy.optimize import fsolve


# should be correct
def slope_mixing_line(cp: float, Mw: float, P: float, Q: float, eta: float)-> float:
    """Determines the slope of the mixing line

    Args
    ----
        cp: float
            Specific heat capacity of the plume (J/kgK)
        Mw: float
            Molecular weight of water (kg/mol)
        P: float
            Pressure of the plume (Pa)
        Q: float
            Heat release rate of the plume (W)
        eta: float
            Plume entrainment coefficient
    
    Returns
    -------
        float:
            Slope of the mixing line
    """
    return (cp * Mw * P) / (0.622 * Q * (1 - eta))


# correct
def Theta_RH(pa: float, RH: float, Ta: float, G: float)->float:
    """Determines the Theta parameter of the mixing line 

    Args
    ----
        pa: float
            Pressure of the plume (Pa)
        RH: float
            Relative humidity of the plume (fraction)
        Ta: float
            Ambient temperature (K)
        G: float
            Mixing ratio of the plume (Pa/K)
    
    Returns
    -------
        float:
            Theta parameter of the mixing line
    """

    """Determines the Theta G parameter of the mixing line iteratively"""
    def thetaG_func(thetaG: float) -> float:
        return G - (2.6e6 / (461 * thetaG**2)) * e_sat_murphy_koop(thetaG)
    
    thetaG = fsolve(thetaG_func, 225)[0]

    def theta_RH_func(thetaRH: float) -> float:
        lhs = RH * e_sat_murphy_koop(Ta) + G * (thetaG - thetaRH)
        rhs = e_sat_murphy_koop(thetaG)
        return lhs - rhs
    
    thetaRH = fsolve(theta_RH_func, thetaG - 5)[0]
    return thetaRH
 

# correct
def dilution_param_function_of_temp(T: ArrayLike , T0: float, Ta: float) -> ArrayLike:
    """Determines the value of the Dilution parameter D as a function of plume temperature

    Args
    ----
        T: ArrayLike 
            Plume temperature along the plume mixing line (K)
        T0: float
            Initial plume temperature (K)
        Ta: float
            Ambient temperature (K)
    
    Returns
    -------
        ArrayLike: 
            Dilution parameter D as a function of plume temperature
    
    See Also
    --------
        The Dilution parameter is extensively defined in Kacher 2015
    """
    return (T - Ta) / (T0 - Ta)


# correct
def dilution_param_function_of_time(t: ArrayLike , tau: float) -> ArrayLike:
    """Determines the value of the Dilution parameter D as a function of time

    Args
    ----
        t: ArrayLike 
            time starting from tau (s)
        tau: float
            The characteristic mixing timescale.
            It is the time it takes for the centerline of the plume to 
            begin mixing. (s)
    
    Returns
    -------
        ArrayLike: 
            Dilution parameter D as a function of time

    Notes
    -----
        t must start from tau since D cannot be less than 1 for t < tau.

    See Also
    --------
        The Dilution parameter is extensively defined in Kacher 2015
    """
    beta = 0.9 # beta parameter as defined in Kacher 2015
    return (tau / t)**beta


# correct
def cooling_rate(D: ArrayLike, T0: float, Ta: float, tau: float) -> ArrayLike:
    """Determines the cooling rate (dT/dt) of the plume as a function of the Dilution parameter D
    
    Args
    ----
        D: ArrayLike
            Dilution parameter D as a function of time
        T0: float
            Initial plume temperature (K)
        Ta: float
            Ambient temperature (K)
        tau: float
            The characteristic mixing timescale.
            It is the time it takes for the centerline of the plume to 
            begin mixing (s)

    Returns
    -------
        ArrayLike:
            Cooling rate (dT/dt) of the plume as a function of the Dilution parameter D (K/s)
    
    See Also
    --------
        Kacher 2015 for the derivation
    """
    beta = 0.9 # beta parameter as defined in Kacher 2015
    return -beta * ((T0 - Ta) / tau) * D**(1 + 1 / beta)


# correct
def saturation_ratio_mixing_line(
        T: ArrayLike,
        Ta: float,
        G: float,
        pwa: float
) -> ArrayLike:
    """Determines the saturation ratio of the plume along the mixing line (with respect to water)

    Args
    ----
        T: ArrayLike
            Plume temperature along the mixing line (K)
        Ta: float
            Ambient temperature (K)
        G: float
            Mixing ratio of the plume (Pa/K)
        pwa: float
            Ambient water vapor pressure

    Returns
    -------
        ArrayLike:
            Saturation ratio of the plume along the mixing line (with respect to water)

    See Also
    --------
        Kacher 2015 for the derivation
    """
    return (pwa + G * (T - Ta)) / e_sat_murphy_koop(T)



def saturation_ratio_mixing_line_ice(
        T: ArrayLike,
        Ta: float,
        G: float,
        pwa: float
) -> ArrayLike:
    """Determines the saturation ratio of the plume (with respect to ice) along the mixing line

    Args
    ----
        T: ArrayLike
            Plume temperature along the mixing line (K)
        Ta: float
            Ambient temperature (K)
        G: float
            Mixing ratio of the plume (Pa/K)
        pwa: float
            Ambient water vapor pressure

    Returns
    -------
        ArrayLike:
            Saturation ratio of the plume along the mixing line with respect to ice

    See Also
    --------
        Kacher 2015 for the derivation
    """
    return (pwa + G * (T - Ta)) / p_sat_ice_murphy_koop(T)




# irrelevant for now
def time_function_of_temp(tau: float, T: ArrayLike, T0: float, Ta: float) -> ArrayLike:
    """Determines the time as a function of temperature

    Args
    ----
        tau: float
            The characteristic mixing timescale.
        T: ArrayLike
            Plume temperature along the mixing line (K)
        T0: float
            Initial plume temperature (K)
        Ta: float
            Ambient temperature (K)
    
    Returns
    -------
        ArrayLike:
            Time as a function of temperature
    """
    beta = 0.9 # beta parameter as defined in Kacher 2015
    return tau * ((T0- Ta) / (T - Ta))**(1 / 0.9)


# correct
def supersaturation_forcing(
        G: float,
        ew: ArrayLike,
        Smw: ArrayLike,
        T: ArrayLike,
        dTdt: ArrayLike
) -> ArrayLike:
    """Determines the supersaturation forcing along the mixing line-> P_W aka dSmw/dT 

    Args
    ----
        G: float
            Mixing ratio of the plume (Pa/K)
        ew: ArrayLike
            Saturated water vapor pressure of the plume (Pa)
        Smw: ArrayLike
            Saturation ratio of water
        T: ArrayLike
            Temperature of the plume (K)
        dTdt: ArrayLike
            Cooling rate of the plume (K/s)

    Returns
    -------
        ArrayLike:
            Supersaturation forcing term in the microphysical model

    See Also
    --------
        Kacher 2015 for the derivation
    """
    L = latent_heat_vap_water(T) # Latent heat of vaporization of water (J/molecule)
    kb = 1.38064852e-23 # Boltzmann constant (J/K)
    return (G / ew - Smw * (L / (kb * T**2))) * dTdt



def supersaturation_forcing2(T: ArrayLike, dTdt: ArrayLike, Smw: ArrayLike) -> ArrayLike:
    """Determines the supersaturation forcing term in the microphysical model

    Args
    ----
        T: ArrayLike
            Temperature of the plume (K)
        dTdt: ArrayLike
            Cooling rate of the plume (K/s)
        Smw: ArrayLike
            Saturation ratio of water

    Returns
    -------
        ArrayLike:
            Supersaturation forcing term in the microphysical model

    See Also
    --------
        Kacher 2015 for the derivation
    """
    dSdT = np.gradient(Smw, T)
    return dSdT * dTdt