import matplotlib.pyplot as plt
import numpy as np
import Thermodynamic_funcs as Tf
import Mixing_line_funcs as Mf

plt.rcParams['font.size']   = 8
#plt.rcParams['font.family'] = 'serif'
#plt.rcParams['mathtext.fontset'] = 'cm'
#plt.rcParams['font.serif']  = 'Times New Roman'

# Generate data
T     = np.linspace(200, 300, 10000)
pwsat = Tf.e_sat_murphy_koop(T)
piSat = Tf.p_sat_ice_murphy_koop(T)

G   = 1.64
TP  = np.linspace(200, 300, 10000)
pvP = Tf.p_sat_ice_murphy_koop(220) + G * (TP - 220)

Sw   = Mf.saturation_ratio_mixing_line(TP, 220, G, Tf.e_sat_murphy_koop(220))
Si   = Mf.saturation_ratio_mixing_line_ice(TP, 220, G, Tf.p_sat_ice_murphy_koop(220))
maxw = np.argmax(Sw)

# Create a 6×3.5 inch figure without constrained_layout
fig, axes = plt.subplots(1, 2, figsize=(6, 3.5), constrained_layout=False)

# --- Left panel: partial pressures ---
ax = axes[0]
ax.scatter(220, Tf.p_sat_ice_murphy_koop(220), color='black', marker='o', zorder=4)
ax.scatter(TP[maxw], pvP[maxw], color='black', marker='x', zorder=4)
ax.plot(T, pwsat, color='blue', label=r'$p_{w,sat}$')
ax.plot(T, piSat, color='cyan', label=r'$p_{i,sat}$')
ax.plot(
    TP[TP >= 220],
    pvP[pvP >= Tf.p_sat_ice_murphy_koop(220)],
    label=r'$p_v$ (mixing line)',
    linestyle='--',
    color='black'
)
ax.set_xlabel('Temperature (K)')
ax.set_xlim(200, 260)
ax.set_ylim(0, 60)
ax.set_ylabel('Water vapour partial pressure (Pa)')
ax.legend()
ax.grid()

# --- Right panel: saturation ratios ---
ax = axes[1]
ax.plot(T, Sw, color='blue', label=r'$S_{mw}$')
ax.plot(T, Si, color='cyan', label=r'$S_{mi}$')
ax.scatter(T[maxw], Sw[maxw], color='black', marker='x', zorder=4)
ax.scatter(T[maxw], Si[maxw], color='black', marker='x', zorder=4)
ax.set_xlabel('Temperature (K)')
ax.set_xlim(220, 260)
ax.set_ylim(0, 2.5)
ax.set_ylabel('Saturation ratio')
ax.legend()
ax.grid()

fig.subplots_adjust(
    bottom=0.12,  # ↑ increased from 0.05 to 0.12
    top=0.9071,
    left=0.075,
    right=0.98,
    wspace=0.3,  # ↑ increased from 0.2 to 0.3
)

# Add panel labels just above the top-left of each axes
labels = ['(a)', '(b)']
for ax, lab in zip(axes, labels):
    bbox = ax.get_position()
    x = bbox.x0
    y = bbox.y1
    dx = 0.01
    dy = 0.02
    fig.text(
        x + dx, y + dy, lab,
        ha='left', va='bottom',
        fontsize=9, fontweight='bold'
    )

plt.show()
# Save the figure
fig.savefig('mixing_line_diagram.png', dpi=400, bbox_inches='tight')