import numpy as np
from numpy.typing import ArrayLike


# correct
def zeta_param(sigma: float):
    """Determines zeta, the size dsitribution slope parameter

    Args
    ----
        sigma: float
            The standard deviation of the size distribution of soot and ambient aerosols
    
    Returns
    -------
        float:
            The size distribution slope parameter zeta
    
    See Also
    --------
        Kacher 2015 for the derivation
    """
    return 4 / (np.sqrt(2 * np.pi) * np.log(sigma))


# correct
def activation_radius(rk: float, k: float, smw: float) -> ArrayLike:
    """Determines the radius required for water droplets to activate onto the aerosol

    Args
    ----
        rk: float
            The kelvin radius (m)
        k: float
            The hygroscopicity parameter of the aerosol
        smw: float
            The supersaturation ratio of water
    
    Returns
    -------
        float:
            The activation radius of the aerosol (m)
    
    See Also
    --------
        Kacher 2015 for the derivation
    """
    return (rk * ((smw**(-2))**(1 / 3))) / ((54 * k)**(1 / 3))


# correct
def mean_activation_radius(
        psiSoot: ArrayLike,
        nwSoot: ArrayLike,
        rActSoot: ArrayLike,
        psiAmbient: ArrayLike,
        nwAmbient: ArrayLike,
        rActAmbient: ArrayLike
) -> ArrayLike:
    """Determines the average activation radius of the plume
    
    Args
    ----
        psiSoot: ArrayLike
            The proportion of activated soot compared to the emitted non activated soot
        nwSoot: ArrayLike
            The number concentration of activated soot particles (m^-3)
        rActSoot: ArrayLike
            The activation radius for soot (m)
        psiAmbient: ArrayLike
            The proportion of activated aerosols compared to the emitted non activated aerosols
        nwAmbient: ArrayLike
            The number concentration of activated aerosols(m^-3)
        rActAmbient: ArrayLike
            The activation radius of the ambient aerosols (m)"""
    return ((psiSoot * nwSoot * rActSoot + psiAmbient  * nwAmbient * rActAmbient) / 
            (psiSoot * nwSoot + psiAmbient * nwAmbient))
    

# correct
def mean_zeta(
        psiSoot: ArrayLike,
        nwSoot: ArrayLike,
        zetaSoot: float,
        psiAmbient: ArrayLike,
        nwAmbient: ArrayLike,
        zetaAmbient: float
)-> ArrayLike:
    """Determines the average size distribution slope parameter of the plume
    
    Args
    ----
        psiSoot: ArrayLike
            The proportion of activated soot compared to the emitted non activated soot
        nwSoot: ArrayLike
            The number concentration of activated soot particles (m^-3)
        zetaSoot: float
            The size distribution slope parameter of the soot
        psiAmbient: ArrayLike
            The proportion of activated aerosols compared to the emitted non activated aerosols
        nwAmbient: ArrayLike
            The number concentration of activated aerosols(m^-3)
        zetaAmbient: float
            The size distribution slope parameter of the ambient aerosols"""
    return ((psiSoot * nwSoot * zetaSoot + psiAmbient  * nwAmbient * zetaAmbient) / 
            (psiSoot * nwSoot + psiAmbient * nwAmbient))

# correct
def psi_function(rAct: ArrayLike, rMeandry: float, zeta: float) -> ArrayLike:
    """Determines the proportion of activated aerosols compored to the 
       emitteed non actived aerosols, psi

    Args
    ----
        rAct: ArrayLike
            The radius required for water droplets to activate onto the aerosol (m)
        rm_dry: float
            The dry radius of the aerosol (m)
        zeta: float
            The size distribution slope parameter
    
    Returns
    -------
        ArrayLike:
            The proportion of activated aerosols compared to the emitted non activated aerosols
    
    See Also
    --------
        Kacher 2015 for the derivation
       
    """
    return 1 / (1 + (rAct / rMeandry)**zeta)



def AEIfinal_func(N0: float, no: float, Dilution: float, rho: float):
    """Determines the time invarient apparent emission index of isce particles (/kg of fuel burned)
    
    Args
    ----
        N0: float
            The air to fuel ratio
        no: float
            The number concentration of activated ice aerosols at to (m^-3)
        Dilution: float
            The dilution parameter of the plume at to
        rho: float
            The density of the plume (kg/m^3) at to
            
    Returns
    -------
        float:
            The time invarient apparent emission index of ice particles (/kg of fuel burned)
        """
    return (N0 * no) / (Dilution * rho)
    