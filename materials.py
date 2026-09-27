# -*- coding: utf-8 -*-
"""Nominal bulk properties and legacy numerical-model settings.

Bulk units: density is kg/m^3, youngs_modulus is Pa, and poisson_ratio is
dimensionless. These are representative values, not a matched set of measured
properties for a specified grade, temperature, direction, and processing history.
See docs/material_properties.md for the source audit and unresolved values.

The strength field mixes strength definitions between entries. Strain thresholds,
compression limits, spring_min_ratio, stiffness_scale, and shear_boost are legacy
solver settings; shear_boost is a dimensionless multiplier, not G/E. Their inline
comments describe historical tuning rationale, not independently verified data.
Restitution and friction depend on the contacting pair and test conditions.
The unused hardness field is a legacy ranking without a consistent hardness
scale. Poisson's ratio is also currently metadata, not an input to the solver.
"""

import numpy as np

MATERIALS = {
    'tungsten': {
        'name': 'Tungsten (pure)',
        'role': 'penetrator',
        'density': 19300,
        'strength': 980e6,
        'youngs_modulus': 411e9,
        'restitution': 0.016,
        'hardness': 7.5,
        'poisson_ratio': 0.28,
        'friction_coeff': 0.4,
        'breaking_strain': 0.008,  # dynamic: slight decrease from static 1-2%; fracture mode shifts transgranular
        'yield_strain': 0.002,     # ~750 MPa yield / 411 GPa = 0.0018; onset of micro-plasticity ~0.002
        'compression_limit': 0.40,
        'spring_min_ratio': 0.78,  # floor = 0.532 + 0.0425×ln(K); K=318 GPa (Haglund et al.)
        'shear_boost': 2.3,        # G = E/(2(1+ν)) = 411/(2×1.28) = 161 GPa → G/E = 0.391
        'stiffness_scale': 6.0e-3, # HV ~3430; ref: RHA HV 345 → 6e-4; 3430×(6e-4/345) = 6.0e-3
        'color': np.array([1.0, 0.7, 0.25])
    },
    'tungsten_carbide': {
        'name': 'Tungsten Carbide',
        'role': 'penetrator',
        'density': 15000,          # nominal cemented WC; exact binder/grade not verified
        'strength': 2700e6,        # WC-Co compressive strength ~2.7 GPa; TRS ~2.0-2.5 GPa
        'youngs_modulus': 630e9,   # nominal value; a matched WC-Co datasheet is still needed
        'restitution': 0.012,
        'hardness': 9.5,
        'poisson_ratio': 0.22,
        'friction_coeff': 0.35,
        'breaking_strain': 0.003,  # dynamic: minimal change; WC-Co stays brittle at all rates
        'yield_strain': 0.003,     # onset of micro-cracking before fracture
        'compression_limit': 0.55,
        'spring_min_ratio': 0.78,  # floor = 0.532 + 0.0425×ln(K); K≈360 GPa for WC-6Co cermet
        'shear_boost': 2.4,        # G = E/(2(1+ν)) = 630/(2×1.22) = 258 GPa → G/E = 0.410 (highest of the group)
        'stiffness_scale': 3.8e-3, # HV ~2200 (WC-6Co); 2200×(6e-4/345) = 3.8e-3
        'color': np.array([0.85, 0.78, 0.65])
    },
    'depleted_uranium': {
        'name': 'Depleted Uranium',
        'role': 'penetrator',
        'density': 19050,          # U-0.75Ti: 19.0-19.05 g/cm³ (OSTI reports)
        'strength': 1200e6,        # U-0.75Ti aged condition UTS ~1200 MPa (OSTI 6998138)
        'youngs_modulus': 175e9,   # Kessler 1987 J.Nucl.Mater: E = 175-193 GPa aged alpha-prime
        'restitution': 0.015,
        'hardness': 6.5,           # HB ~270-350; self-sharpening adiabatic shear behaviour
        'poisson_ratio': 0.23,     # Kessler 1987: ~0.23 for standard aged microstructure
        'friction_coeff': 0.45,
        'breaking_strain': 0.04,   # dynamic: adiabatic shear bands form above ~2891 s⁻¹, halving effective ductility
        'yield_strain': 0.006,     # ~1000 MPa yield / 175 GPa = 0.00571
        'compression_limit': 0.45,
        'spring_min_ratio': 0.74,  # floor = 0.532 + 0.0425×ln(K); K=131 GPa (AIP J.Appl.Phys 2016)
        'shear_boost': 2.4,        # G = E/(2(1+ν)) = 175/(2×1.23) = 71 GPa → G/E = 0.406; ductile but shear ratio similar to W
        'stiffness_scale': 4.7e-4, # HV ~270 (U-0.75Ti aged, HB 270-350); 270×(6e-4/345) = 4.7e-4
        'color': np.array([0.6, 0.62, 0.5])
    },
    'steel_core': {
        'name': 'Steel (AP core)',
        'role': 'penetrator',
        'density': 7850,
        'strength': 2100e6,        # 60 HRC SAE 52100: UTS ~2100-2150 MPa
        'youngs_modulus': 210e9,
        'restitution': 0.026,
        'hardness': 7.0,           # 60 HRC ≈ HV~746; well above RHA (~360 HB)
        'poisson_ratio': 0.29,
        'friction_coeff': 0.55,
        'breaking_strain': 0.02,   # dynamic: essentially brittle at 60 HRC; near-zero elongation at ballistic rates
        'yield_strain': 0.0086,    # ~1800 MPa yield / 210 GPa = 0.00857
        'compression_limit': 0.48,
        'spring_min_ratio': 0.75,  # floor = 0.532 + 0.0425×ln(K); K=160 GPa (hardened, E=210 GPa ν=0.29)
        'shear_boost': 2.2,        # G = E/(2(1+ν)) = 210/(2×1.29) = 81 GPa → G/E = 0.387
        'stiffness_scale': 1.3e-3, # HV ~746 (60 HRC); 746×(6e-4/345) = 1.3e-3
        'color': np.array([0.55, 0.52, 0.58])
    },
    'copper_jet': {
        'name': 'Copper (HEAT jet)',
        'role': 'penetrator',
        'density': 8960,           # OFHC copper: 8.96 g/cm³
        'strength': 190e6,         # OFHC-Cu jet flow stress: 193 MPa theory / 144 MPa from fragment hardness (Appl.Sci. 2023 13:5128)
        'youngs_modulus': 117e9,   # annealed OFHC: 110-128 GPa
        'restitution': 0.032,      # very plastic; a jet does not rebound, it flows
        'hardness': 2.3,           # HV ~50 annealed OFHC (Mohs 3); softer than Al 5083
        'poisson_ratio': 0.34,
        'friction_coeff': 0.45,    # Cu on steel, dry: 0.36-0.53
        'breaking_strain': 0.55,   # OFE copper jet elongation at failure ~0.55 before particulation (Petit et al. J.Appl.Phys 2005)
        'yield_strain': 0.0016,    # 190 MPa dynamic flow / 117 GPa; yields almost immediately = hydrodynamic stretching
        'compression_limit': 0.42,
        'spring_min_ratio': 0.74,  # floor = 0.532 + 0.0425×ln(K); K = 122 GPa (E=117, ν=0.34)
        'shear_boost': 2.1,        # G = E/(2(1+ν)) = 117/(2×1.34) = 43.7 GPa → G/E = 0.373
        'stiffness_scale': 8.7e-5, # HV ~50; 50×(6e-4/345) = 8.7e-5
        'color': np.array([0.85, 0.45, 0.20])
    },
    'steel': {
        'name': 'RHA Steel',
        'role': 'both',
        'density': 7850,
        'strength': 1600e6,        # dynamic flow stress at ballistic strain rates ~1400-1700 MPa (Cowper-Symonds RHA)
        'youngs_modulus': 207e9,   # Q&T low-alloy steel; modulus barely changes with heat treatment
        'restitution': 0.030,
        'hardness': 5.5,           # HB 302-360 for Class 3 RHA
        'poisson_ratio': 0.29,
        'friction_coeff': 0.6,
        'breaking_strain': 0.09,   # dynamic: significantly reduced from static 13-20%; void density increase at ballistic rates
        'yield_strain': 0.0053,    # ~1090 MPa yield / 207 GPa = 0.00527
        'compression_limit': 0.50,
        'spring_min_ratio': 0.75,  # floor = 0.532 + 0.0425×ln(K); K=165 GPa (anchor: formula calibrated here)
        'shear_boost': 2.2,        # G = E/(2(1+ν)) = 207/(2×1.29) = 80 GPa → G/E = 0.387
        'stiffness_scale': 6.0e-4, # HV ~345 (HB 302-360 → HV 310-380); anchor reference point
        'color': np.array([0.5, 0.55, 0.65])
    },
    'high_hardness_steel': {
        'name': 'High Hardness Steel',
        'role': 'armor',
        'density': 7850,
        'strength': 1850e6,        # dynamic UTS ~1850 MPa (Armox 500T min 1700; Cai et al. 2010 ballistic tests)
        'youngs_modulus': 210e9,   # nominal low-alloy-steel value; not universal to all steels
        'restitution': 0.024,      # harder than RHA → less elastic rebound
        'hardness': 6.3,           # BHN ~500 (HRC ~51); between RHA 5.5 and steel_core 7.0
        'poisson_ratio': 0.29,
        'friction_coeff': 0.58,
        'breaking_strain': 0.04,   # dynamic: ~3-6% at ballistic rates; ductile-to-brittle transition begins (Cai 2010)
        'yield_strain': 0.0074,    # ~1550 MPa dynamic yield / 210 GPa
        'compression_limit': 0.53,
        'spring_min_ratio': 0.77,  # harder than RHA → engages compression floor sooner
        'shear_boost': 2.2,        # G/E essentially same as RHA (same E, same ν)
        'stiffness_scale': 8.7e-4, # HV ~500; 500×(6e-4/345) = 8.7e-4
        'color': np.array([0.55, 0.57, 0.62])
    },
    'ultra_high_hardness_steel': {
        'name': 'Ultra-High Hardness Steel',
        'role': 'armor',
        'density': 7850,
        'strength': 2250e6,        # dynamic UTS ~2250 MPa (Armox 600T min 2200; quasi-brittle at ballistic rates)
        'youngs_modulus': 210e9,
        'restitution': 0.018,      # nearly inelastic — UHHS tends to shatter rather than rebound
        'hardness': 6.8,           # BHN ~600 (HRC ~58); below steel_core 60 HRC but harder than HHS
        'poisson_ratio': 0.29,
        'friction_coeff': 0.52,
        'breaking_strain': 0.018,  # dynamic: ~1.5-2.5% at ballistic rates; near-brittle at high strain rate
        'yield_strain': 0.0090,    # ~1900 MPa dynamic yield / 210 GPa
        'compression_limit': 0.56,
        'spring_min_ratio': 0.79,  # very little compression tolerated before hardening
        'shear_boost': 2.2,
        'stiffness_scale': 1.06e-3,# HV ~610; 610×(6e-4/345) = 1.06e-3
        'color': np.array([0.42, 0.44, 0.50])
    },
    'ceramic': {
        'name': 'Alumina (Al2O3)',
        'role': 'armor',
        'density': 3960,           # nominal high-purity alumina; supplier grade is unspecified
        'strength': 2500e6,        # contact/compressive strength ~2.5 GPa (MOR 380 MPa was flexural only)
        'youngs_modulus': 370e9,   # supported for some high-purity grades; see source audit
        'restitution': 0.013,
        'hardness': 9.0,           # HV 1400-1600; anchor for ceramics
        'poisson_ratio': 0.22,
        'friction_coeff': 0.5,
        'breaking_strain': 0.0010, # dynamic: ~0.1-0.3% strain at ballistic rates; increased transgranular fracture
        'yield_strain': 0.001,     # no plastic yield zone — fracture = yield for ceramics
        'compression_breaking_strain': 0.007,  # ~2500 MPa / 370 GPa = 0.0068; brittle ceramics fracture in compression too
        'compression_limit': 0.65,
        'spring_min_ratio': 0.77,  # floor = 0.532 + 0.0425×ln(K); K=257 GPa (NIST Al2O3 99.5%)
        'shear_boost': 2.4,        # G = E/(2(1+ν)) = 370/(2×1.22) = 151 GPa → G/E = 0.410
        'stiffness_scale': 2.6e-3, # HV ~1500 (Al2O3 99%); 1500×(6e-4/345) = 2.6e-3
        'color': np.array([0.95, 0.9, 0.82])
    },
    'boron_carbide': {
        'name': 'Boron Carbide (B4C)',
        'role': 'armor',
        'density': 2520,
        'strength': 3500e6,        # compressive strength ~2.5-3.5 GPa; B4C is the hardest armor ceramic
        'youngs_modulus': 460e9,   # RUS measurement 458.7 GPa; AZoM/MakeItFrom confirm ~460 GPa
        'restitution': 0.010,
        'hardness': 9.8,           # HV ~38 GPa; 3rd hardest material (after diamond, cBN)
        'poisson_ratio': 0.17,     # retained; exact value not verified by the selected reference
        'friction_coeff': 0.45,
        'breaking_strain': 0.0012, # dynamic: ~0.001-0.005; stress-induced amorphization possible at extreme rates
        'yield_strain': 0.00127,   # no plastic zone; fracture = yield
        'compression_breaking_strain': 0.008,  # ~3500 MPa / 460 GPa = 0.0076; B4C fractures violently in compression
        'compression_limit': 0.70,
        'spring_min_ratio': 0.76,  # floor = 0.532 + 0.0425×ln(K); K=231 GPa avg (MDPI 2024: 221-241 GPa)
        'shear_boost': 2.5,        # G = E/(2(1+ν)) = 460/(2×1.17) = 197 GPa → G/E = 0.427
        'stiffness_scale': 3.0e-3, # reduced from HV-based 6.7e-3; HV-formula gave 25× RHA stiffness (should be ~11× from E ratio); 3.0e-3 gives k≈11× RHA and stays numerically stable with progressive hardening
        'color': np.array([0.72, 0.72, 0.72])
    },
    'silicon_carbide': {
        'name': 'Silicon Carbide (SiC)',
        'role': 'armor',
        'density': 3210,
        'strength': 3000e6,        # compressive strength ~3 GPa; SiC often exceeds Al2O3 in contact resistance
        'youngs_modulus': 410e9,
        'restitution': 0.012,
        'hardness': 9.3,           # HV ~2500; between Al2O3 and B4C
        'poisson_ratio': 0.19,
        'friction_coeff': 0.48,
        'breaking_strain': 0.0010, # dynamic: spall-based fracture at ~450-606 MPa; very brittle at all rates
        'yield_strain': 0.00088,   # no plastic zone
        'compression_breaking_strain': 0.008,  # ~3000 MPa / 410 GPa = 0.0073; SiC shatters in compression similarly to B4C
        'compression_limit': 0.68,
        'spring_min_ratio': 0.77,  # floor = 0.532 + 0.0425×ln(K); K=250 GPa (3C-SiC, MakeItFrom)
        'shear_boost': 2.5,        # G = E/(2(1+ν)) = 410/(2×1.19) = 172 GPa → G/E = 0.420
        'stiffness_scale': 2.5e-3, # reduced from HV-based 4.3e-3 for same stability reason as B4C
        'color': np.array([0.65, 0.68, 0.66])
    },
    'glass': {
        'name': 'Glass (soda-lime)',
        'role': 'armor',
        'density': 2500,           # soda-lime float glass: 2500 kg/m³ (Pilkington ATS-129, AGC)
        'strength': 1700e6,        # dynamic compressive strength 1.7 GPa (ARL-TR-9120); quasi-static 1.3-1.5 GPa
        'youngs_modulus': 72e9,    # 71.7-72 GPa (glass.org FM05-12, Pilkington)
        'restitution': 0.020,
        'hardness': 5.5,           # Mohs 5-6; Knoop 470-605 kg/mm², HV ~470
        'poisson_ratio': 0.22,     # 0.20-0.23 across datasheets
        'friction_coeff': 0.55,    # glass on steel, dry: ~0.5-0.6
        'breaking_strain': 0.0006, # annealed MOR 41 MPa / 72 GPa; glass fails in tension at flaw sites
        'yield_strain': 0.0006,    # no plastic zone — fracture = yield
        'compression_breaking_strain': 0.024,  # 1700 MPa / 72 GPa; comminutes into rubble under contact load
        'compression_limit': 0.60,
        'spring_min_ratio': 0.69,  # floor = 0.532 + 0.0425×ln(K); K = 42.9 GPa (E=72, ν=0.22)
        'shear_boost': 2.4,        # G = 29.6 GPa (glass.org) → G/E = 0.413
        'stiffness_scale': 8.2e-4, # HV ~470; 470×(6e-4/345) = 8.2e-4
        'color': np.array([0.72, 0.86, 0.88])
    },
    'titanium': {
        'name': 'Titanium (Ti-6Al-4V)',
        'role': 'armor',
        'density': 4430,
        'strength': 1450e6,        # dynamic YS at high strain rates ~1200-1500 MPa (Johnson-Cook Ti-6Al-4V)
        'youngs_modulus': 114e9,   # nominal Ti-6Al-4V value; sources vary with condition
        'restitution': 0.028,
        'hardness': 4.8,           # HB 334-379; clearly below RHA (5.5) — fixes titanium < RHA
        'poisson_ratio': 0.34,
        'friction_coeff': 0.52,    # Ti-6Al-4V has notably high friction (~0.5-0.6) due to galling
        'breaking_strain': 0.09,   # dynamic: engineering fracture strain 4-6% → true strain ~0.09 at ballistic rates
        'yield_strain': 0.0096,    # ~1100 MPa yield / 114 GPa = 0.00965
        'compression_limit': 0.48,
        'spring_min_ratio': 0.73,  # floor = 0.532 + 0.0425×ln(K); K=101 GPa (E=114 GPa, ν=0.34)
        'shear_boost': 2.1,        # G = E/(2(1+ν)) = 114/(2×1.34) = 42.5 GPa → G/E = 0.373
        'stiffness_scale': 6.3e-4, # HV ~360 (HB 334-379 → HV ~345-390); 360×(6e-4/345) = 6.3e-4
        'color': np.array([0.7, 0.7, 0.72])
    },
    'aluminum_armor': {
        'name': 'Aluminum 5083',
        'role': 'armor',
        'density': 2660,           # 5083: 2.66 g/cm³ ✓
        'strength': 420e6,         # dynamic YS 5083 at high strain rate ~350-450 MPa (Lesuer 2000)
        'youngs_modulus': 70.3e9,  # 5083: 70.3 GPa
        'restitution': 0.030,
        'hardness': 2.8,           # HB ~80 (5083); below mild steel
        'poisson_ratio': 0.33,
        'friction_coeff': 0.55,
        'breaking_strain': 0.13,   # dynamic: ~10-14% fracture strain at ballistic rates (Clausen et al. 2004 5083 split-Hopkinson); static 15-20%
        'yield_strain': 0.003,     # ~215 MPa yield / 70.3 GPa = 0.00306
        'compression_limit': 0.50,
        'spring_min_ratio': 0.70,  # floor = 0.532 + 0.0425×ln(K); K=58 GPa (E=70.3 GPa, ν=0.33)
        'shear_boost': 2.1,        # G = E/(2(1+ν)) = 70.3/(2×1.33) = 26.4 GPa → G/E = 0.376
        'stiffness_scale': 1.4e-4, # HV ~80 (HB ~80 for 5083); 80×(6e-4/345) = 1.4e-4
        'color': np.array([0.82, 0.84, 0.86])
    },
    'uhmwpe': {
        'name': 'UHMWPE (Dyneema)',
        'role': 'armor',
        'density': 970,            # nominal fibre value; no matched composite grade verified
        'strength': 3.4e9,         # legacy coefficient, not a verified bulk-composite property
        'youngs_modulus': 100e9,   # longitudinal fibre-scale estimate, not isotropic plate modulus
        'restitution': 0.020,
        'hardness': 1.8,           # Shore D ~60-65; very soft vs metals
        'poisson_ratio': 0.42,     # legacy scalar; no matched fibre/composite value verified
        'friction_coeff': 0.25,    # UHMWPE: one of the lowest dry friction coefficients of any solid
        'breaking_strain': 0.025,  # dynamic: ductile-to-brittle transition above ~100 s⁻¹; yarns fail at ~2-3% at ballistic rates
        'yield_strain': 0.015,     # ~1.5% onset of fibre/matrix damage
        'compression_limit': 0.35,
        'spring_min_ratio': 0.60,  # floor = 0.532 + 0.0425×ln(K); K≈3 GPa through-thickness (polymer matrix dominates)
        'shear_boost': 2.0,        # G = E/(2(1+ν)) = 100/(2×1.42) = 35 GPa → G/E = 0.352
        'stiffness_scale': 9e-6,   # Shore D ~62 → HV ~5 (very rough); 5×(6e-4/345) ≈ 9e-6; extremely compliant
        'color': np.array([0.92, 0.93, 0.95])
    },
    'era_filler': {
        'name': 'ERA Filler (Comp B)',
        'role': 'armor',
        'density': 1720,           # Composition B (60/40 RDX/TNT), the usual ERA fill (Wikipedia Gurney table)
        'strength': 59e6,          # dynamic peak compressive flow stress ~60 MPa (cast Comp B / PBX class)
        'youngs_modulus': 1.1e9,
        'restitution': 0.035,
        'hardness': 1.0,
        'poisson_ratio': 0.48,
        'friction_coeff': 0.40,
        'explosive': True,
        'detonation_velocity': 7920,   # Comp B D_CJ = 7.92 km/s
        'gurney_velocity': 2700,       # √(2E) = 2.70 km/s; chemical energy per mass is E = √(2E)² / 2
        'product_gamma': 3.0,          # polytropic γ of Comp B detonation products
        'initiation_velocity': 400,    # shock initiation; ~2-3 GPa into the charge
        'breaking_strain': 0.007,  # tensile response far weaker than compressive; tears below 1% strain
        'yield_strain': 0.005,     # crushes plastically well before failure in compression
        'compression_limit': 0.30,
        'spring_min_ratio': 0.64,  # floor = 0.532 + 0.0425×ln(K); K = 11.4 GPa
        'shear_boost': 1.9,        # G = E/(2(1+ν)) = 1.1/(2×1.48) = 0.37 GPa → G/E = 0.338
        'stiffness_scale': 2.6e-5, # HV ~15; 15×(6e-4/345) = 2.6e-5
        'color': np.array([0.80, 0.62, 0.35])
    },
    'adamantium': {
        'name': 'Adamantium',
        'role': 'both',
        'density': 5e8,
        'strength': 2e12,
        'youngs_modulus': 2000e9,
        'restitution': 0.0,
        'hardness': 10.0,
        'poisson_ratio': 0.20,
        'friction_coeff': 0.15,
        'breaking_strain': 1.0,
        'yield_strain': 0.02,
        'compression_limit': 0.995,
        'spring_min_ratio': 0.92,  # fictional; effectively incompressible
        'shear_boost': 5.0,        # effectively indestructible; no lateral deformation
        'stiffness_scale': 5e-2,   # fictional; far stiffer than any real material
        'color': np.array([0.75, 0.78, 0.85])
    },
}
