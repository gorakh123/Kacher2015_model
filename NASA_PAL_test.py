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
Ta = 226.15            # Ambient temperature (K)
pa = 18754         # Ambient total pressure (Pa)
Ti = 363.15            # Initial temperature of the plume (K)
pwa = Tf.p_sat_ice_murphy_koop(Ta) # Ambient water vapor pressure (Pa)
nAmbient = 0 # Ambient number concentration of aerosols (/m^3)

# Engine (and Mixing Line) Parameters 
N0 = 61.7             # Air to fuel 
nsoot = 4.4736 * 10**12 # Number concentration of soot in the plume at the exhaust (/m^3)
G = 3.39                # Mixing line slope (Pa/K)
tau = 0.02          # Characteristic mixing timescale (s)
# 1 - 0.033625
# Aerosol properties
sigmaSoot = 1.7     # Standard deviation of a log-normal distribution for soot
sigmaAmbient = 2.2  # Standard deviation of a log-normal distribution for ambient aerosols
kSoot = 0.005       # Solubility parameter for soot
kAmbient = 0.5      # Solubility parameter for ambient aerosols
rK = 1 * 10**-9     # Kelvin radius (m) Note: Held Constant in this parametrisation
rMeanSoot = 2.359 * 10**-9 # Mean dry soot radius (m)
rMeanAmbient = 14 * 10**-9 # Mean dry Ambient aerosol radius (m)
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
pwSat = Tf.e_sat_murphy_koop(T)                 # Saturation (at liquid saturation) vapor pressure of water (Pa)
piSat = Tf.p_sat_ice_murphy_koop(T)            # Saturation (at ice saturation) vapor pressure of water (Pa)
niSat = Tf.number_concentration_sat_ice(T)     # Number concentration of H20 at ice saturation as a function of temperature (m^-3)
nwSat = Tf.number_concentration_sat_water(T)    # Number concentration of H20 at water saturation as a function of temperature (m^-3)
vThermal = Tf.mean_thermal_speed(T)             # Mean thermal speed of water molecules as a function of temperature (m/s)

# Mixing line parameters
Dilution = Mf.dilution_param_function_of_temp(T, Ti, Ta)# Dilution parameter of the plume
Smw = Mf.saturation_ratio_mixing_line(T, Ta, G, pwa)    # Saturation ratio of the plume along the mixing line
smw = Smw - 1                                           # Supersaturation ratio of the plume along the mixing line

Smi = Mf.saturation_ratio_mixing_line_ice(T, Ta, G, pwa) # Saturation ratio of the plume along the mixing line with respect to ice
smi = Smi - 1  
                                        # Supersaturation ratio of the plume along the mixing line with respect to ice
dTdt = Mf.cooling_rate(Dilution, Ti, Ta, tau)           # Cooling rate of the plume (K/s)

# Particle distribution parameters
EIsoot = (N0 * nsoot) / (rho * Dilution) # Apparent emission index of soot particles (/kg of fuel burned)
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

# Microphysical parameters - determines radius and number concentration at water saturation
nw1 = Mpf.number_conc_all_activated1(nwSoot, nwAmbient) # Number concentration of all activated aerosols in the plume (m^-3)

b1 = Mpf.b1_param(T, nwSat, vThermal, smw) # b1 parameter of the microphysical model

dSdT = Mf.supersaturation_forcing(G, pwSat, Smw, T, dTdt) # Supersaturation forcing term in the microphysical model

tauActivation = Mpf.tau_activation(zetaAvg, smw, dSdT) # Activation timescale of the aerosol
tauGrowth = Mpf.tau_growth(b1, rActivationAvg) # Growth timescale of the aerosol
kW = Mpf.kw_parameter(tauActivation, tauGrowth) # kW parameter of the microphysical model

Rw = Mpf.condensation_sink(b1, rActivationAvg, nwSat, kW) # Condensation sink parameter of the plume (s^-1)
nw2 = Mpf.number_conc_all_activated2(dSdT, Rw) # Number concentration of all activated aerosols in the plume (m^-3)

ro = Mpf.find_ro(rActivationAvg, kW) # mean radius of activated ice particles

# Determine Final Results at Water Saturation:
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

timeV = time[valid_indices]
TV = T[valid_indices]


smwV = smw[valid_indices]
nw1V = nw1[valid_indices]
nw2V = nw2[valid_indices]
roV = ro[valid_indices]
rhoV = rho[valid_indices]
DilutionV = Dilution[valid_indices]

nwSootV = nwSoot[valid_indices]
nwAmbientV = nwAmbient[valid_indices]
psiSootV = psiSoot[valid_indices]
psiAmbientV = psiAmbient[valid_indices]

#print(f"nw1: {nw1}")
#print(f"nw2: {nw2}")

# Find the intersection of the two functions for nw1 and nw2 of smw
sw_intercept, nw_intercept, index = Mpf.find_activated_nw(smwV, nw1V, nw2V)

# Find the values at the intersection
ro_intercept = (roV[index]+roV[index+1])/2

time_intercept = (timeV[index] + timeV[index + 1]) / 2
T_intercept = (TV[index] + TV[index + 1]) / 2

nwSoot_intercept = (nwSootV[index] + nwSootV[index + 1]) / 2
nwAmbient_intercept = (nwAmbientV[index] + nwAmbientV[index + 1]) / 2
psiSoot_intercept = (psiSootV[index] + psiSootV[index + 1]) / 2
psiAmbient_intercept = (psiAmbientV[index] + psiAmbientV[index + 1]) / 2

Dilution_intercept = (DilutionV[index] + DilutionV[index + 1]) / 2
rho_intercept = (rhoV[index] + rhoV[index + 1]) / 2

index1 = np.where(T == TV[index])[0][0] # Find the index of the intersection point in T


# Find results at Ice saturation

Dilution = Dilution[0:index1+1]
time = time[0:index1+1]
rho = rho[0:index1+1]
T = T[0:index1+1]
smi = smi[0:index1+1]
Smi = Smi[0:index1+1]
dTdt = dTdt[0:index1+1]
niSat = niSat[0:index1+1]
vThermal = vThermal[0:index1+1]
piSat = piSat[0:index1+1]

#ractSootI = PDf.activation_radius(rK, kSoot, smi) # Activation radius for soot (m)
ractSootI = ro_intercept # Activation radius for soot (m)

b1i = Mpf.b1_param(T, niSat, vThermal, smi)
dSidT = Mf.supersaturation_forcing(G, piSat, Smi, T, dTdt)

ni1 = (nwSoot_intercept * rho * Dilution) / (Dilution_intercept * rho_intercept) 
tauFreeze = Mpf.freezing_time_scale(dTdt)
tauGrowthIce = Mpf.growth_time_scale_ice(ro_intercept, T, pwa)
ki = Mpf.ki_parameter(tauFreeze, tauGrowthIce)
Ri = Mpf.condensation_sink(b1i, ractSootI, niSat, ki)
ni2 = Mpf.number_conc_all_activated2(dSidT, Ri)


ri = Mpf.find_ro(ro_intercept, ki) # mean radius of activated ice particles

si_intercept, ni_intercept, index2 = Mpf.find_activated_nw(smi, ni1, ni2)

index2 = index1 # since the number concentration required to quench the ice is lower than the 
                # number concentration along the mixing line
ri_intercept = ri[index2]
ni_intercept = ni1[index2]
timei_intercept = time[index2]
Ti_intercept = T[index2]
Dilutioni_intercept = Dilution[index2]
rhoi_intercept = rho[index2]


Phi = Mpf.Phi_function(nw_intercept, nwSoot_intercept, psiSoot_intercept, nwAmbient_intercept, psiAmbient_intercept)

RH = pwa / Tf.e_sat_murphy_koop(Ta) # Relative humidity of the ambient air over water
RHi = pwa / Tf.p_sat_ice_murphy_koop(Ta) # Relative humidity of the ambient air over ice

thetaRH = Mf.Theta_RH(pa, RH, Ta, G)
AEIice = PDf.AEIfinal_func(N0, ni_intercept, Dilutioni_intercept, rhoi_intercept)


print(f"pa: {pa}")
print(f"RH: {RH}")
print(f"RHi: {RHi}")
print(f"Ta: {Ta}")
print(f"thetaRH - Ta: {thetaRH - Ta}")
print(f"Phi: {Phi}")
print(f"no (/m^3): {nw_intercept }")
print(f"ni (/m^3): {ni_intercept }")
print(f"to: {time_intercept}")
print(f"ti (s): {timei_intercept}") 
print(f"To (K): {T_intercept}")
print(f"Ti (K): {Ti_intercept}")
print(f"ro (micrometres): {ro_intercept * 10**6}")
print(f"ri (micrometres): {ri_intercept * 10**6}")


