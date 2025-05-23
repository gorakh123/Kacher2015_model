import numpy as np
import Thermodynamic_funcs as Tf
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from numpy.typing import ArrayLike


# correct
def number_conc_soot_activated1(
        psi: ArrayLike, 
        EIsoot: float, 
        rho: ArrayLike, 
        Dilution: ArrayLike, 
        N0: float
) -> ArrayLike:
    """Determines the number concentration of activated soot particles in the plume

    Args
    ----
        psi: ArrayLike
            The proportion of activated soot compared to the emitted non activated soot
        EIsoot: float
            The emission index of soot per fuel burned (/kg)
        rho: ArrayLike
            The density of the aerosol (kg/m^3)
        Dilution: ArrayLike
            The dilution parameter of the plume
        N0: float
            air to fuel ratio
    
    Returns
    -------
        ArrayLike:
            The number concentration of activated soot particles in the plume
    
    Notes
    -----
        The Dilution parameter used typically is a function of temperature
    """
    return (psi * Dilution * EIsoot * rho) / N0


# correct
def number_conc_ambient_activated1(
        psi: ArrayLike,
        Ta: float,
        T: ArrayLike,
        Dilution: ArrayLike,
        na: float
) -> ArrayLike:
    """Determines the number concentration of activated ambient aerosols in the plume

    Args
    ----
        psi: ArrayLike
            The proportion of activated aerosols compared to the emitted non activated aerosols
        Ta: float
            The ambient temperature (K)
        T: ArrayLike
            The temperature of the plume (K)
        Dilution: ArrayLike
            The dilution parameter of the plume
        na: float
            The number concentration of ambient aerosols (m^-3) already in the atmosphere
    
    Returns
    -------
        ArrayLike:
            The number concentration of activated ambient aerosols in the plume
    """
    return (psi * (Ta / T) * (1 - Dilution) * na)


# correct
def b1_param(
        T: ArrayLike, 
        nwSat: ArrayLike, 
        meanSpeed: ArrayLike,
        smw: ArrayLike
        ) -> ArrayLike:
    """Determines the b1 parameter of the microphysical model

    Args
    ----
        T: ArrayLike
            Temperature (K)
        nwSat: ArrayLike
            Number concentration of H20 at saturation water as a function of temperature (m^-3)
        meanSpeed: ArrayLike
            Mean thermal speed of water molecules as a function of temperature (m/s)
        smw: float
            supersaturation ratio of water

    Returns
    -------
        ArrayLike:
            b1 parameter of the microphysical model
    """
    vol = 2.99 * 10**-29 # Volume of water molecule (m^3) May need improvement
    return (nwSat * vol * meanSpeed * smw)/4


# correct
def tau_activation(avgZeta: float, sw: ArrayLike, dSdT: ArrayLike) -> ArrayLike:
    """Determines the activation timescale of the aerosol

    Args
    ----
        avgZeta: float
            The average size distribution slope parameter of the plume
        sw: ArrayLike
            The supersaturation ratio of the plume along the mixing line
        dSdT: ArrayLike
            The supersaturation forcing term in the microphysical model
    
    Returns
    -------
        ArrayLike:
            The activation timescale of the aerosol
    """
    return (3 * sw) / (2 * avgZeta * dSdT)


# correct
def tau_growth(b1: ArrayLike, rActivationAvg: ArrayLike):
    """Determines the growth timescale of the aerosol

    Args
    ----
        b1: ArrayLike
            The b1 parameter of the microphysical model
        rActivationAvg: ArrayLike
            The average activation radius of the plume particles
    
    Returns
    -------
        ArrayLike:
            The growth timescale of the aerosol
    """
    return rActivationAvg / b1


# correct
def kw_parameter(tauActivation: ArrayLike, tauGrowth: ArrayLike) -> ArrayLike:
    """Determines the kw parameter of the microphysical model

    Args
    ----
        tauActivation: ArrayLike
            The activation timescale of the aerosol
        tauGrowth: ArrayLike
            The growth timescale of the aerosol
    
    Returns
    -------
        ArrayLike:
            The kw parameter of the microphysical model

    See Also
    --------
        Kacher 2015 for the derivation
    """
    return tauActivation / tauGrowth


# correct
def condensation_sink(b1: ArrayLike, rActivation: ArrayLike, nwSaturation: ArrayLike, kw: ArrayLike) -> ArrayLike:
    """Determines the condensation sink of the plume R_w

    Args
    ----
        b1: ArrayLike
            The b1 parameter of the microphysical model
        rActivation: ArrayLike
            The activation radius of the plume particles
        nwSaturation: ArrayLike
            The number concentration of H2O in water saturation
        kw: ArrayLike
            The kw parameter of the microphysical model
    
    Returns
    -------
        ArrayLike:
            The condensation sink of the plume
    
    See Also
    --------
        Kacher 2015 for the derivation
    """
    volH2O = 2.99 * 10**-29 # Volume of water molecule (m^3)
    return ((b1 * 4 * np.pi * rActivation**2) * 
            (1 + 2 * kw + 2 * kw**2))/(volH2O * nwSaturation)


# correct
def number_conc_all_activated2(dSdT: ArrayLike, condSink: ArrayLike):
    """Determines the number concentration required to quench all the excess water vapour
    
    Args
    ----
        dSdT: ArrayLike
            The supersaturation forcing term in the microphysical model
        condSink: ArrayLike
            The condensation sink of the plume
    
    Returns
    -------
        ArrayLike:
            The number concentration required to quench all the excess water vapour
    """
    return dSdT / condSink


# correct
def number_conc_all_activated1(nwSoot: ArrayLike, nwAmbient: ArrayLike) -> ArrayLike:
    """Determines the total number concentration of activated water droplets

    Args
    ----
        nwSoot: ArrayLike
            The number concentration of activated soot particles in the plume (m^-3)
        nwAmbient: ArrayLike
            The number concentration of activated ambient aerosols in the plume (m^-3)
    
    Returns
    -------
        ArrayLike:
            The total number concentration of activated water droplets (m^-3)
    """
    return nwSoot + nwAmbient

# correct
def find_ro(rAct: ArrayLike, kW: ArrayLike)-> ArrayLike:
    """Determines the radius of the activated water droplets

    Args
    ----
        rAct: ArrayLike
            The average activation radius of the plume particles
        kW: ArrayLike
            The kw parameter of the microphysical model
    
    Returns
    -------
        ArrayLike:
            The radius of the activated water droplets
    """
    return rAct * (1 + kW)


# correct
def find_activated_nw(sw: ArrayLike, nw1: ArrayLike, nw2: ArrayLike)-> ArrayLike:
    """Determines the number concentration of activated water droplets by finding the intercept
       of number_conc_all_activated2 and number_conc_activated

    Args
    ----
        sw: ArrayLike
            The supersaturation ratio of the plume along the mixing line
        nw1: ArrayLike
            The number concentration of activated soot particles in the plume
        nw2: ArrayLike
            The number concentration of activated ambient aerosols in the plume
    
    Returns
    -------
        ArrayLike:
            [1 x 3] array containing the supersaturation ratio, the number concentration of activated water droplets
            and the index of the intercept
    
    Notes
    -----
        Ensure arrays are cleaned, ie no NaNs or Infs and also sw >= 0
    """
    diff = np.array(nw1) - np.array(nw2) 
    
    sign_changes = np.where(np.diff(np.sign(diff)))[0] # Finds the index just before sign of diff changes
    return (sw[sign_changes] + sw[sign_changes+1])/2, (nw1[sign_changes] + nw1[sign_changes+1] + nw2[sign_changes] + nw2[sign_changes+1])/4 , sign_changes


# Double check with paper
def Phi_function(nwfinal: ArrayLike, nwSoot: ArrayLike, psiSoot: ArrayLike, nwAmbient: ArrayLike, psiAmbient: ArrayLike)-> ArrayLike:
    """Determines the fraction of activated water droplets that are soot

    Args
    ----
        nwfinal: ArrayLike
            The number concentration of activated water droplets
        nwSoot: ArrayLike
            The number concentration of activated soot particles in the plume
        psiSoot: ArrayLike
            The proportion of activated soot compared to the emitted non activated soot
        nwAmbient: ArrayLike
            The number concentration of activated ambient aerosols in the plume
        psiAmbient: ArrayLike
            The proportion of activated aerosols compared to the emitted non activated aerosols
    
    Returns
    -------
        ArrayLike:
            The fraction of activated water droplets that are soot
    """
    nSoot = nwSoot / psiSoot
    nAmbient = nwAmbient / psiAmbient
    return nwfinal / (nSoot + nAmbient)



def freezing_time_scale(dTdt: float) -> float:
    """Determines the freezing time scale of the plume tau_frz

    Args
    ----
        dTdt: float
            The cooling rate of the plume (K/s)
    
    Returns
    -------
        tau_frz: float:
            The freezing time scale of the plume (s)
    """
    return 1 / (-3.5714 * dTdt)



def Diffusion_coefficient(T: ArrayLike, pa: ArrayLike) ->ArrayLike:
    """Determines the diffusion coefficient of the plume
    
    Args
    ----
        T: ArrayLike
            The temperature of the plume (K)
        pa: ArrayLike
            The ambient pressure (Pa)
    
    Returns
    -------
        ArrayLike:
            The diffusion coefficient of water vapor in air (m^2/s)
    """
    return 0.211 * (T / 273.15)**1.94 * (101325 / pa) 

def growth_time_scale_ice(r_i: float, T_i: ArrayLike, pa: float) -> ArrayLike:
    """Determines the growth time scale of the plume tau_g

    Args
    ----
        r_i: float
            The radius of the activated water droplets (m)
        T_i: ArrayLike
            The temperature of the plume (K)
        pa: float
            The ambient pressure (Pa)

    Returns
    -------
        tau_g: float:
            The growth time scale for ice particle growth(s)
    """ 
    alpha = 1 # accomodation coefficient
    h2oVol = 2.99 * 10**-29 # Volume of water molecule (m^3)
    thermalSpeed = Tf.mean_thermal_speed(T_i) # Mean thermal speed of water molecules
    numberConcSat = Tf.number_concentration_sat_water(T_i) # Number concentration of H2O at water saturation

    D = Diffusion_coefficient(T_i, pa) 
    b1 = (alpha * h2oVol * thermalSpeed * numberConcSat) / 4

    b2 = thermalSpeed / (4 * D)

    return   (1 + b2 * r_i) / (b1 / r_i) # Growth time scale of the plume (s)



def ki_parameter(tau_frz: ArrayLike, tau_g: ArrayLike) -> ArrayLike:
    """Determines the ki parameter of the microphysical model

    Args
    ----
        tau_frz: ArrayLike
            The freezing time scale of the plume (s)
        tau_g: ArrayLike
            The growth time scale of the plume (s)
    
    Returns
    -------
        ArrayLike:
            The ki parameter of the microphysical model 
    
    See Also
    --------
        Kacher 2015 for the derivation
    """
    return tau_frz / tau_g
# INCOMPLETE



def optical_depth_function(nwfinal: float, rfinal: float, d0: float, Do: float)-> float:
    """Determines the optical depth of the plume

    Args
    ----
        nwfinal: float
            The number concentration of activated water droplets
        rfinal: float
            The radius of the activated water droplets
        d0: float
            engine diameter
        Do: float
            The dilution parameter at activation relaxation
    
    Returns
    -------
        float:
            The optical depth of the plume
    """
    d = d0 / np.sqrt(Do) # Plume diamter at activation relaxation
    Q = 1 # to be determined
    return np.pi * rfinal**2 * Q * nwfinal * d