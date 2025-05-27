import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import Microphysical_funcs as Mpf
import Particle_distribution_funcs as PDf
import Thermodynamic_funcs as Tf
import Mixing_line_funcs as Mf

plt.rcParams['font.size'] = 9
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.serif'] = 'Times New Roman'

# Figure 1 of Report:
# Mixing line diagram

T = np.linspace(200, 300, 10000)
pwsat = Tf.e_sat_murphy_koop(T)
piSat = Tf.p_sat_ice_murphy_koop(T)

G = 1.64

TP = np.linspace(200, 300, 10000)
pvP = Tf.p_sat_ice_murphy_koop(220) + G * (TP - 220)

Sw = Mf.saturation_ratio_mixing_line(TP, 220, G, Tf.e_sat_murphy_koop(220))
Si = Mf.saturation_ratio_mixing_line_ice(TP, 220, G, Tf.p_sat_ice_murphy_koop(220))

maxw = np.argmax(Sw)
maxi = np.argmax(Si)



fig, axes = plt.subplots( 1, 2,figsize=(6, 3), constrained_layout=True)

axes[0].scatter(220, Tf.p_sat_ice_murphy_koop(220), color='black', marker='o', zorder=4)
axes[0].scatter(TP[maxw], pvP[maxw], color='black', marker='x', zorder=4)
axes[0].plot(T, pwsat, label='$p_{w,sat}$', color='blue')
axes[0].plot(T, piSat, label='$p_{i,sat}$', color='cyan')
axes[0].plot(TP[TP >= 220], pvP[pvP >= Tf.p_sat_ice_murphy_koop(220)], label='$p_{v}$ / Mixing line', color='black', linestyle='--')

axes[0].set_xlabel(' Temperature (K)')
axes[0].set_xlim(200, 260)
axes[0].set_ylim(0, 60)
axes[0].set_ylabel('Water vapour partial pressure (Pa)')
axes[0].legend()
axes[0].grid()

axes[1].plot(T, Sw, color='blue', label= '$S_{mw}$')
axes[1].plot(T, Si, color='cyan', label= '$S_{mi}$')
axes[1].scatter(T[maxw], Sw[maxw], color='black', marker='x', zorder=4)
axes[1].scatter(T[maxw], Si[maxw], color='black', marker='x', zorder=4)
axes[1].set_xlabel(' Temperature (K)')
axes[1].set_xlim(220, 260)
axes[1].set_ylim(0, 2.5)
axes[1].set_ylabel('Saturation ratio')
axes[1].legend()
axes[1].grid()
plt.show()

# Save the figure
fig.savefig('mixing_line_diagram.png', dpi=400, bbox_inches='tight')