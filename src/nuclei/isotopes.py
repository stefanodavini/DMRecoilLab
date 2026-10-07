"""
Frequently used target isotopes for
dark-matter direct-detection studies.

The nuclei defined in this module are
pre-configured Nucleus instances with
approximate nuclear masses and spins.

All masses are expressed in eV (c=1).
"""

from src.nuclei.nucleus import Nucleus


#
# Sodium
#

Na23 = Nucleus(
    symbol="Na23",
    A=23,
    Z=11,
    mass=22.5e9,
    spin=3/2,
)


#
# Silicon
#

Si28 = Nucleus(
    symbol="Si28",
    A=28,
    Z=14,
    mass=26.1e9,
    spin=0,
)

Si29 = Nucleus(
    symbol="Si29",
    A=29,
    Z=14,
    mass=27.0e9,
    spin=0.5,
)

Si30 = Nucleus(
    symbol="Si30",
    A=30,
    Z=14,
    mass=27.9e9,
    spin=0,
)

#
# Argon
#

Ar40 = Nucleus(
    symbol="Ar40",
    A=40,
    Z=18,
    mass=37.5e9,
    spin=0,
)

#
# Germanium
#

Ge70 = Nucleus(
    symbol="Ge70",
    A=70,
    Z=32,
    mass=65.2e9,
    spin=0,
)

Ge72 = Nucleus(
    symbol="Ge72",
    A=72,
    Z=32,
    mass=67.0e9,
    spin=0,
)

Ge73 = Nucleus(
    symbol="Ge73",
    A=73,
    Z=32,
    mass=68.0e9,
    spin=9/2,
)

Ge74 = Nucleus(
    symbol="Ge74",
    A=74,
    Z=32,
    mass=68.9e9,
    spin=0,
)

Ge76 = Nucleus(
    symbol="Ge76",
    A=76,
    Z=32,
    mass=70.8e9,
    spin=0,
)

#
# Xenon
#

Xe128 = Nucleus(
    symbol="Xe128",
    A=128,
    Z=54,
    mass=119.2e9,
    spin=0,
)

Xe129 = Nucleus(
    symbol="Xe129",
    A=129,
    Z=54,
    mass=120.1e9,
    spin=1/2,
)

Xe130 = Nucleus(
    symbol="Xe130",
    A=130,
    Z=54,
    mass=121.1e9,
    spin=0,
)

Xe131 = Nucleus(
    symbol="Xe131",
    A=131,
    Z=54,
    mass=122.0e9,
    spin=3/2,
)

Xe132 = Nucleus(
    symbol="Xe132",
    A=132,
    Z=54,
    mass=122.9e9,
    spin=0,
)

Xe134 = Nucleus(
    symbol="Xe134",
    A=134,
    Z=54,
    mass=124.8e9,
    spin=0,
)

Xe136 = Nucleus(
    symbol="Xe136",
    A=136,
    Z=54,
    mass=126.7e9,
    spin=0.0,
)

"""
Dictionary mapping mass number A
to the corresponding xenon isotope.
"""
XENON = {
    128: Xe128,
    129: Xe129,
    130: Xe130,
    131: Xe131,
    132: Xe132,
    134: Xe134,
    136: Xe136,
}