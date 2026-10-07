"""
Physical constants and default astrophysical parameters.

All quantities are expressed in the units indicated
in the variable names.
"""

# Particle masses (MeV)
PROTON_MASS_MeV = 938.27208943

# Fundamental constants
C_kmps = 299792458e-3         # c sim 3e5 km/s
HBAR_C_MeV_fm = 197.3269804   # hbarc sim 197 MeV fm
ELECTRON_CHARGE_coulomb = -1.602176634e-19

# Dark Matter default parameters
LOCAL_DM_DENSITY_GeV_cm3 = 0.3  # rho sim 0.3 Gev c^-2 cm^-3
SHM_V0_kmps = 220               # Standard Halo Model v0
SHM_VESC_kmps = 544             # Galactic escape speed
VEARTH_kmps = 238               # relative Earth speed

# Conversion factors
CM_TO_M = 1e-2
KM_TO_M = 1e3
DAY_TO_SECONDS = 24 * 3600