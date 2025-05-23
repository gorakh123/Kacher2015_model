from numpy.typing import ArrayLike
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt

import Thermodynamic_funcs as Tf
import Mixing_line_funcs as Mf
import Microphysical_funcs as Mpf
import Particle_distribution_funcs as PDf


"""
Input Parameters for the model
------------------------------
"""

# Ambient & initial plume conditions
Ta = 220            # Ambient temperature (K)
pa = 25000          # Ambient total pressure (Pa)
Ti = 600            # Initial temperature of the plume (K)
pwa = Tf.p_sat_ice_murphy_koop(Ta) # Ambient water vapor pressure (Pa)
nAmbient = 600 * 10**6 # Ambient number concentration of aerosols (/m^3)

# Engine (and Mixing Line) Parameters 
N0 = 60             # Air to fuel ratio
EIsoot = 5 * 10**14 # Emission index of soot per fuel burned (/kg)
G = 1.64            # Mixing line slope (Pa/K)
tau = 0.01          # Characteristic mixing timescale (s)

# Aerosol properties
sigmaSoot = 1.6     # Standard deviation of a log-normal distribution for soot
sigmaAmbient = 2.2  # Standard deviation of a log-normal distribution for ambient aerosols
kSoot = 0.005       # Solubility parameter for soot
kAmbient = 0.5      # Solubility parameter for ambient aerosols
rK = 1 * 10**-9     # Kelvin radius (m) Note: Held Constant in this parametrisation
rMeanSoot = 12.5 * 10**-9 # Mean dry soot radius (m)
rMeanAmbient = 15 * 10**-9 # Mean dry Ambient aerosol radius (m)
# Model resolution 
n = 100000          # Number of points in the model

"""
Model Calculations
------------------------------
"""
# Temperature and time
T = np.geomspace(Ta, Ti, n)                      # Temperature of the plume along the mixing line (K)
rho = pa/(287 * T)                              # Density of the plume as a function of temperature (kg/m^3)
                                                # Note we assume isobaric expansion
time = Mf.time_function_of_temp(tau, T, Ti, Ta) # Time as a function of temperature

# Thermodynamic parameters
pwSat = Tf.e_sat_murphy_koop(T)                 # Saturation vapor pressure of water (Pa)
nwSat = Tf.number_concentration_sat_water(T)    # Number concentration of H20 at water saturation as a function of temperature (m^-3)
vThermal = Tf.mean_thermal_speed(T)             # Mean thermal speed of water molecules as a function of temperature (m/s)

# Mixing line parameters
Dilution = Mf.dilution_param_function_of_temp(T, Ti, Ta)# Dilution parameter of the plume
Smw = Mf.saturation_ratio_mixing_line(T, Ta, G, pwa)    # Saturation ratio of the plume along the mixing line
smw = Smw - 1                                           # Supersaturation ratio of the plume along the mixing line
dTdt = Mf.cooling_rate(Dilution, Ti, Ta, tau)           # Cooling rate of the plume (K/s)      

# Particle distribution parameters
zetaSoot = PDf.zeta_param(sigmaSoot)                    # Zeta parameter for soot
zetaAmbient = PDf.zeta_param(sigmaAmbient)              # Zeta parameter for ambient aerosols

rActivationSoot = PDf.activation_radius(rK, kSoot, smw) # Activation radius for soot (m)
rActivationAmbient = PDf.activation_radius(rK, kAmbient, smw) # Activation radius for ambient aerosols (m)

psiSoot = PDf.psi_function(rActivationSoot, rMeanSoot, zetaSoot) # Psi function for soot
psiAmbient = PDf.psi_function(rActivationAmbient, rMeanAmbient, zetaAmbient) # Psi function for ambient aerosols

nwSoot = Mpf.number_conc_soot_activated1(psiSoot, EIsoot, rho, Dilution, N0) # Number concentration of activated soot in the plume (m^-3)
nwAmbient = Mpf.number_conc_ambient_activated1(psiAmbient, Ta, T, Dilution, nAmbient) # Number concentration of activated ambient aerosols in the plume (m^-3) 

rActivationAvg = PDf.mean_activation_radius(psiSoot, nwSoot, rActivationSoot, psiAmbient, nwAmbient, rActivationAmbient) # Average activation radius of the plume (m)
zetaAvg = PDf.mean_zeta(psiSoot, nwSoot, zetaSoot, psiAmbient, nwAmbient, zetaAmbient) # Average zeta parameter of the plume

# Microphysical parameters
nw1 = Mpf.number_conc_all_activated1(nwSoot, nwAmbient) # Number concentration of all activated aerosols in the plume (m^-3)

b1 = Mpf.b1_param(T, nwSat, vThermal, smw) # b1 parameter of the microphysical model

dSdT = Mf.supersaturation_forcing(G, pwSat, Smw, T, dTdt) # Supersaturation forcing term in the microphysical model

tauActivation = Mpf.tau_activation(zetaAvg, smw, dSdT) # Activation timescale of the aerosol
tauGrowth = Mpf.tau_growth(b1, rActivationAvg) # Growth timescale of the aerosol
kW = Mpf.kw_parameter(tauActivation, tauGrowth) # kW parameter of the microphysical model

Rw = Mpf.condensation_sink(b1, rActivationAvg, nwSat, kW) # Condensation sink parameter of the plume (s^-1)
nw2 = Mpf.number_conc_all_activated2(dSdT, Rw) # Number concentration of all activated aerosols in the plume (m^-3)

ro = Mpf.find_ro(rActivationAvg, kW) # mean radius of activated ice particles

# Determine Final Results:
# - Pressure    - Float
# - RH(ambient) - Float     - pwa / pwSat(Ta)
# - RHi(ambient)- Float     - pwa / picesat(Ta)
# - Ta          - Float
# - Theta - T   - Float     - Need to find Phi
# - Phi         - Float     - Need to find Phi
# - ni          - Float     - intersection of nw1 and nw2
# - to          - Float     - time of intersection
# - ri          - Float     - ro at intersection
# - OD          - Float     - Optical depth at intersection -> Not calculated atm


# Clean up the data 
valid_indices = np.logical_not(np.isnan(nw1)) & np.logical_not(np.isnan(nw2)) & (smw >= 0)

time = time[valid_indices]
T = T[valid_indices]

smw = smw[valid_indices]
nw1 = nw1[valid_indices]
nw2 = nw2[valid_indices]
ro = ro[valid_indices]

nwSoot = nwSoot[valid_indices]
nwAmbient = nwAmbient[valid_indices]
psiSoot = psiSoot[valid_indices]
psiAmbient = psiAmbient[valid_indices]
Dilution = Dilution[valid_indices]
rho = rho[valid_indices]

#print(f"nw1: {nw1}")
#print(f"nw2: {nw2}")

# Find the intersection of the two functions for nw1 and nw2 of smw
sw_intercept, nw_intercept, index = Mpf.find_activated_nw(smw, nw1, nw2)
#print(f"sw_intercept: {sw_intercept}")
#print(f"nw_intercept: {nw_intercept}")
#print(f"index: {index}")
# Find the values at the intersection
ro_intercept = (ro[index]+ro[index+1])/2

time_intercept = (time[index] + time[index + 1]) / 2
T_intercept = (T[index] + T[index + 1]) / 2

nwSoot_intercpet = (nwSoot[index] + nwSoot[index + 1]) / 2
nwAmbient_intercept = (nwAmbient[index] + nwAmbient[index + 1]) / 2
psiSoot_intercept = (psiSoot[index] + psiSoot[index + 1]) / 2
psiAmbient_intercept = (psiAmbient[index] + psiAmbient[index + 1]) / 2

Dilution_intercept = (Dilution[index] + Dilution[index + 1]) / 2
rho_intercept = (rho[index] + rho[index + 1]) / 2

Phi = Mpf.Phi_function(nw_intercept, nwSoot_intercpet, psiSoot_intercept, nwAmbient_intercept, psiAmbient_intercept)

RH = pwa / Tf.e_sat_murphy_koop(Ta) # Relative humidity of the ambient air over water
RHi = pwa / Tf.p_sat_ice_murphy_koop(Ta) # Relative humidity of the ambient air over ice

thetaRH = Mf.Theta_RH(pa, RH, Ta, G)
AEIice = PDf.AEIfinal_func(N0, nw_intercept, Dilution_intercept, rho_intercept)

print(f"pa: {pa}")
print(f"RH: {RH}")
print(f"RHi: {RHi}")
print(f"Ta: {Ta}")
print(f"thetaRH - Ta: {thetaRH - Ta}")
print(f"Phi: {Phi}")
print(f"ni (/cm^3): {nw_intercept * 10**-6}")
print(f"to: {time_intercept}")
print(f"ri (micrometres): {ro_intercept * 10**6}")

# Figure 11 

EIsoot = np.geomspace(10 * 10**11, 10 * 10**16, 1000)
n = 10000

AEIiceList = []
PhiList = []
toList = [] 
noList = []
riList = []



for EIsooti in EIsoot:
    """
    Model Calculations
    ------------------------------
    """
    # Temperature and time
    T = np.geomspace(Ta, Ti, n)                      # Temperature of the plume along the mixing line (K)
    rho = pa/(287 * T)                              # Density of the plume as a function of temperature (kg/m^3)
                                                    # Note we assume isobaric expansion
    time = Mf.time_function_of_temp(tau, T, Ti, Ta) # Time as a function of temperature

    # Thermodynamic parameters
    pwSat = Tf.e_sat_murphy_koop(T)                 # Saturation vapor pressure of water (Pa)
    nwSat = Tf.number_concentration_sat_water(T)    # Number concentration of H20 at water saturation as a function of temperature (m^-3)
    vThermal = Tf.mean_thermal_speed(T)             # Mean thermal speed of water molecules as a function of temperature (m/s)

    # Mixing line parameters
    Dilution = Mf.dilution_param_function_of_temp(T, Ti, Ta)# Dilution parameter of the plume
    Smw = Mf.saturation_ratio_mixing_line(T, Ta, G, pwa)    # Saturation ratio of the plume along the mixing line
    smw = Smw - 1                                           # Supersaturation ratio of the plume along the mixing line
    dTdt = Mf.cooling_rate(Dilution, Ti, Ta, tau)           # Cooling rate of the plume (K/s)      

    # Particle distribution parameters
    zetaSoot = PDf.zeta_param(sigmaSoot)                    # Zeta parameter for soot
    zetaAmbient = PDf.zeta_param(sigmaAmbient)              # Zeta parameter for ambient aerosols

    rActivationSoot = PDf.activation_radius(rK, kSoot, smw) # Activation radius for soot (m)
    rActivationAmbient = PDf.activation_radius(rK, kAmbient, smw) # Activation radius for ambient aerosols (m)

    psiSoot = PDf.psi_function(rActivationSoot, rMeanSoot, zetaSoot) # Psi function for soot
    psiAmbient = PDf.psi_function(rActivationAmbient, rMeanAmbient, zetaAmbient) # Psi function for ambient aerosols

    nwSoot = Mpf.number_conc_soot_activated1(psiSoot, EIsooti, rho, Dilution, N0) # Number concentration of activated soot in the plume (m^-3)
    nwAmbient = Mpf.number_conc_ambient_activated1(psiAmbient, Ta, T, Dilution, nAmbient) # Number concentration of activated ambient aerosols in the plume (m^-3) 

    rActivationAvg = PDf.mean_activation_radius(psiSoot, nwSoot, rActivationSoot, psiAmbient, nwAmbient, rActivationAmbient) # Average activation radius of the plume (m)
    zetaAvg = PDf.mean_zeta(psiSoot, nwSoot, zetaSoot, psiAmbient, nwAmbient, zetaAmbient) # Average zeta parameter of the plume

    # Microphysical parameters
    nw1 = Mpf.number_conc_all_activated1(nwSoot, nwAmbient) # Number concentration of all activated aerosols in the plume (m^-3)

    b1 = Mpf.b1_param(T, nwSat, vThermal, pwSat, smw) # b1 parameter of the microphysical model

    dSdT = Mf.supersaturation_forcing(G, pwSat, Smw, T, dTdt) # Supersaturation forcing term in the microphysical model

    tauActivation = Mpf.tau_activation(zetaAvg, smw, dSdT) # Activation timescale of the aerosol
    tauGrowth = Mpf.tau_growth(b1, rActivationAvg) # Growth timescale of the aerosol
    kW = Mpf.kw_parameter(tauActivation, tauGrowth) # kW parameter of the microphysical model

    Rw = Mpf.condensation_sink(b1, rActivationAvg, nwSat, kW) # Condensation sink parameter of the plume (s^-1)
    nw2 = Mpf.number_conc_all_activated2(dSdT, Rw) # Number concentration of all activated aerosols in the plume (m^-3)

    ro = Mpf.find_ro(rActivationAvg, kW) # mean radius of activated ice particles

    # Determine Final Results:
    # - Pressure    - Float
    # - RH(ambient) - Float     - pwa / pwSat(Ta)
    # - RHi(ambient)- Float     - pwa / picesat(Ta)
    # - Ta          - Float
    # - Theta - T   - Float     - Need to find Phi
    # - Phi         - Float     - Need to find Phi
    # - ni          - Float     - intersection of nw1 and nw2
    # - to          - Float     - time of intersection
    # - ri          - Float     - ro at intersection
    # - OD          - Float     - Optical depth at intersection -> Not calculated atm


    # Clean up the data 
    valid_indices = np.logical_not(np.isnan(nw1)) & np.logical_not(np.isnan(nw2)) & (smw >= 0)

    time = time[valid_indices]
    T = T[valid_indices]

    smw = smw[valid_indices]
    nw1 = nw1[valid_indices]
    nw2 = nw2[valid_indices]
    ro = ro[valid_indices]

    nwSoot = nwSoot[valid_indices]
    nwAmbient = nwAmbient[valid_indices]
    psiSoot = psiSoot[valid_indices]
    psiAmbient = psiAmbient[valid_indices]
    Dilution = Dilution[valid_indices]
    rho = rho[valid_indices]

    #print(f"nw1: {nw1}")
    #print(f"nw2: {nw2}")

    # Find the intersection of the two functions for nw1 and nw2 of smw
    sw_intercept, nw_intercept, index = Mpf.find_activated_nw(smw, nw1, nw2)
    #print(f"sw_intercept: {sw_intercept}")
    #print(f"nw_intercept: {nw_intercept}")
    #print(f"index: {index}")
    # Find the values at the intersection
    ro_intercept = (ro[index]+ro[index+1])/2

    time_intercept = (time[index] + time[index + 1]) / 2
    T_intercept = (T[index] + T[index + 1]) / 2

    nwSoot_intercpet = (nwSoot[index] + nwSoot[index + 1]) / 2
    nwAmbient_intercept = (nwAmbient[index] + nwAmbient[index + 1]) / 2
    psiSoot_intercept = (psiSoot[index] + psiSoot[index + 1]) / 2
    psiAmbient_intercept = (psiAmbient[index] + psiAmbient[index + 1]) / 2

    Dilution_intercept = (Dilution[index] + Dilution[index + 1]) / 2
    rho_intercept = (rho[index] + rho[index + 1]) / 2

    Phi = Mpf.Phi_function(nw_intercept, nwSoot_intercpet, psiSoot_intercept, nwAmbient_intercept, psiAmbient_intercept)

    RH = pwa / Tf.e_sat_murphy_koop(Ta) # Relative humidity of the ambient air over water
    RHi = pwa / Tf.p_sat_ice_murphy_koop(Ta) # Relative humidity of the ambient air over ice

    thetaRH = Mf.Theta_RH(pa, RH, Ta, G)
    AEIice = PDf.AEIfinal_func(N0, nw_intercept, Dilution_intercept, rho_intercept)

    AEIiceList.append(AEIice)
    PhiList.append(Phi)
    toList.append(time_intercept)
    noList.append(nw_intercept * 10**-6)
    riList.append(ro_intercept * 10**6)

fig1, ax = plt.subplots(2, 3, figsize=(10, 10), constrained_layout=True)
ax[0, 0].plot(EIsoot, AEIiceList)
ax[0, 0].set_xlabel("EIsoot (/kg fuel burnt)")
ax[0, 0].set_ylabel("AEIice (/kg fuel burnt)")
ax[0, 0].set_xscale("log")
ax[0, 0].set_yscale("log")
ax[0, 0].set_xlim([10**11, 10**16])
ax[0, 0].set_ylim([10**12, 10**16])

ax[0, 1].plot(EIsoot, PhiList)
ax[0, 1].set_xlabel("EIsoot (/kg fuel burnt)")
ax[0, 1].set_ylabel("Phi")
ax[0, 1].set_xscale("log")
ax[0, 1].set_xlim([10**11, 10**16])
ax[0, 1].set_ylim([0, 1])

ax[0, 2].plot(EIsoot, toList)
ax[0, 2].set_xlabel("EIsoot (/kg fuel burnt)")
ax[0, 2].set_ylabel("to (s)")
ax[0, 2].set_xscale("log")
ax[0, 2].set_xlim([10**11, 10**16])
ax[0, 2].set_ylim([0, 1])

ax[1, 0].plot(EIsoot, noList)
ax[1, 0].set_xlabel("EIsoot (/kg fuel burnt)")
ax[1, 0].set_ylabel("ni (cm^-3)")
ax[1, 0].set_xscale("log")
ax[1, 0].set_yscale("log")
ax[1, 0].set_xlim([10**11, 10**16])
ax[1, 0].set_ylim([10**2, 10**7])

ax[1, 1].plot(EIsoot, riList)
ax[1, 1].set_xlabel("EIsoot (/kg fuel burnt)")
ax[1, 1].set_ylabel("ri (micrometres)")
ax[1, 1].set_xscale("log")
ax[1, 1].set_xlim([10**11, 10**16])
ax[1, 1].set_ylim([0, 2])

plt.show()


