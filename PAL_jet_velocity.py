import numpy as np
import matplotlib.pyplot as plt

# Given conditions
rhoj = 18753.74 / (297 * (273.14 + 90))
rhoe = 18753.74 / (297 * (273.14 - 48))
p_e = rhoe / rhoj

r = 0.018  # nozzle radius (m)
uExit = 13.43  # exit velocity (m/s)

# Thermodynamic constants
gamma = 1.4
R = 297
Tj = 273.14 + 90  # stagnation temperature at exit in K

# Exit Mach number
a_exit = np.sqrt(gamma * R * Tj)
Mj = uExit / a_exit

# Eddy-viscosity constant for subsonic
K = 0.08 * (1 - 0.16 * Mj) * p_e**(-0.22)
kappa = K

# Non-dimensional potential-core length
Xc = 0.70

# Axial distances
x = np.linspace(0, 1.15, 1000)
bar_x = x / r

# Threshold for potential core (bar_x threshold)
bar_x_thresh = Xc / (kappa * np.sqrt(p_e))

# Compute normalized centerline velocity
u_bar = np.empty_like(bar_x)
# Inside potential core: velocity equals exit velocity
u_bar[bar_x <= bar_x_thresh] = 1.0
# Beyond potential core: use the exact decay law
mask = bar_x > bar_x_thresh
denominator = kappa * bar_x[mask] * np.sqrt(p_e) - Xc
u_bar[mask] = 1 - np.exp(-1 / denominator)

# Compute dimensional centerline velocity
u_c = uExit * u_bar

# Plot
plt.figure()
plt.plot(x, u_c)
plt.xlabel('Axial distance x (m)')
plt.ylabel('Centerline velocity $u_c(x)$ (m/s)')
plt.title('Exact Centerline Velocity Decay (Witze 1974)')
plt.grid(True)
plt.show()

tau = np.trapezoid(u_c**(-1), x)
print(f"Time constant tau: {tau:.4f} s")
