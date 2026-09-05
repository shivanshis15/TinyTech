import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import miepython as mp

from scipy.interpolate import interp1d
from scipy.signal import find_peaks


# ============================================================
# SETTINGS
# ============================================================

WAVELENGTH_MIN_NM = 300
WAVELENGTH_MAX_NM = 800

wavelengths_nm = np.arange(
    WAVELENGTH_MIN_NM,
    WAVELENGTH_MAX_NM + 1,
    1
)

# Factor 1: nanoparticle radius
radii_nm = [20, 40, 60, 80, 100]

# Factor 2: host-medium refractive index
host_indices = [1.00, 1.33, 1.45, 1.52, 1.60]

# Nanoparticle materials
materials = {
    "Au": "au_nk.csv",
    "Ag": "ag_nk.csv"
}


# ============================================================
# LOAD JOHNSON & CHRISTY DATA
# ============================================================

def load_material_data(filename):

    df = pd.read_csv(filename)

    # Johnson & Christy wavelength is in micrometres
    wl_nm = df["wl"].values * 1000

    n = df["n"].values
    k = df["k"].values

    # Complex refractive index
    m_complex = n + 1j * k

    # Interpolation
    interpolation = interp1d(
        wl_nm,
        m_complex,
        kind="linear",
        bounds_error=True
    )

    return interpolation, wl_nm


# ============================================================
# MIE CALCULATION
# ============================================================

def calculate_mie(
    radius_nm,
    n_medium,
    m_interpolated
):

    qabs_values = []
    qsca_values = []
    qext_values = []

    for wavelength_nm in wavelengths_nm:

        # Complex refractive index of nanoparticle
        m_particle = m_interpolated(wavelength_nm)

        # Relative refractive index
        m_relative = m_particle / n_medium

        # Size parameter
        x = (
            2
            * np.pi
            * n_medium
            * radius_nm
            / wavelength_nm
        )

        # Mie theory
        qext, qsca, qback, g = mp.single_sphere(
            m_relative,
            x,
            0,
            1
        )

        # Absorption
        qabs = qext - qsca

        # Remove tiny numerical negative values
        qabs = max(qabs, 0)

        qabs_values.append(qabs)
        qsca_values.append(qsca)
        qext_values.append(qext)

    return (
        np.array(qabs_values),
        np.array(qsca_values),
        np.array(qext_values)
    )


# ============================================================
# FIND DOMINANT EXTINCTION RESONANCE
# ============================================================

def find_resonance(material, qext):

    if material == "Au":

        search_min = 450
        search_max = 750

    else:

        search_min = 330
        search_max = 600

    mask = (
        (wavelengths_nm >= search_min)
        &
        (wavelengths_nm <= search_max)
    )

    wl_region = wavelengths_nm[mask]
    qext_region = qext[mask]

    # Find local maxima
    peaks, properties = find_peaks(
        qext_region,
        prominence=np.max(qext_region) * 0.01,
        distance=10
    )

    if len(peaks) > 0:

        # Strongest extinction resonance
        strongest = peaks[
            np.argmax(qext_region[peaks])
        ]

        peak_wavelength = wl_region[strongest]
        peak_value = qext_region[strongest]

    else:

        # Fallback
        strongest = np.argmax(qext_region)

        peak_wavelength = wl_region[strongest]
        peak_value = qext_region[strongest]

    return int(peak_wavelength), peak_value


# ============================================================
# LOAD DATA
# ============================================================

material_data = {}

for material, filename in materials.items():

    print(f"\nLoading {material} data...")

    m_interpolated, available_wavelengths = (
        load_material_data(filename)
    )

    # Check wavelength coverage
    if wavelengths_nm.min() < available_wavelengths.min():

        raise ValueError(
            f"{filename} does not contain data down to "
            f"{wavelengths_nm.min()} nm."
        )

    if wavelengths_nm.max() > available_wavelengths.max():

        raise ValueError(
            f"{filename} does not contain data up to "
            f"{wavelengths_nm.max()} nm."
        )

    material_data[material] = m_interpolated


# ============================================================
# ============================================================
# STUDY 1: EFFECT OF NANOPARTICLE RADIUS
# ============================================================
# ============================================================

radius_results = {}

radius_summary = []


for material in materials:

    print(
        f"\nCalculating radius dependence for {material}..."
    )

    radius_results[material] = {}

    # Keep host medium fixed at air
    n_medium = 1.00

    for radius_nm in radii_nm:

        print(
            f"  Radius = {radius_nm} nm"
        )

        qabs, qsca, qext = calculate_mie(
            radius_nm,
            n_medium,
            material_data[material]
        )

        peak_wavelength, peak_qext = (
            find_resonance(
                material,
                qext
            )
        )

        radius_results[material][radius_nm] = {
            "qabs": qabs,
            "qsca": qsca,
            "qext": qext,
            "peak_wavelength": peak_wavelength,
            "peak_qext": peak_qext
        }

        radius_summary.append({
            "Material": material,
            "Radius (nm)": radius_nm,
            "Host n": n_medium,
            "Resonance (nm)": peak_wavelength,
            "Qext at resonance": round(
                peak_qext,
                4
            )
        })


# ============================================================
# PLOT: GOLD - RADIUS EFFECT
# ============================================================

plt.figure(figsize=(10, 6))

for radius_nm in radii_nm:

    data = radius_results["Au"][radius_nm]

    plt.plot(
        wavelengths_nm,
        data["qext"],
        label=f"{radius_nm} nm"
    )

    plt.scatter(
        data["peak_wavelength"],
        data["peak_qext"],
        s=50,
        zorder=5
    )

    plt.annotate(
        f"{data['peak_wavelength']} nm",
        (
            data["peak_wavelength"],
            data["peak_qext"]
        ),
        xytext=(5, 8),
        textcoords="offset points",
        fontsize=9
    )


plt.title(
    "Effect of Nanoparticle Radius on Au Optical Response"
)

plt.xlabel("Wavelength (nm)")

plt.ylabel(
    "Extinction Efficiency, $Q_{ext}$"
)

plt.xlim(
    WAVELENGTH_MIN_NM,
    WAVELENGTH_MAX_NM
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend(
    title="Radius"
)

plt.tight_layout()

plt.show()


# ============================================================
# PLOT: SILVER - RADIUS EFFECT
# ============================================================

plt.figure(figsize=(10, 6))

for radius_nm in radii_nm:

    data = radius_results["Ag"][radius_nm]

    plt.plot(
        wavelengths_nm,
        data["qext"],
        label=f"{radius_nm} nm"
    )

    plt.scatter(
        data["peak_wavelength"],
        data["peak_qext"],
        s=50,
        zorder=5
    )

    plt.annotate(
        f"{data['peak_wavelength']} nm",
        (
            data["peak_wavelength"],
            data["peak_qext"]
        ),
        xytext=(5, 8),
        textcoords="offset points",
        fontsize=9
    )


plt.title(
    "Effect of Nanoparticle Radius on Ag Optical Response"
)

plt.xlabel("Wavelength (nm)")

plt.ylabel(
    "Extinction Efficiency, $Q_{ext}$"
)

plt.xlim(
    WAVELENGTH_MIN_NM,
    WAVELENGTH_MAX_NM
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend(
    title="Radius"
)

plt.tight_layout()

plt.show()


# ============================================================
# ============================================================
# STUDY 2: EFFECT OF HOST-MEDIUM REFRACTIVE INDEX
# ============================================================
# ============================================================

medium_results = {}

medium_summary = []


# For this study, keep particle radius fixed
FIXED_RADIUS_NM = 40


for material in materials:

    print(
        f"\nCalculating host-medium dependence for {material}..."
    )

    medium_results[material] = {}

    for n_medium in host_indices:

        print(
            f"  Host refractive index = {n_medium}"
        )

        qabs, qsca, qext = calculate_mie(
            FIXED_RADIUS_NM,
            n_medium,
            material_data[material]
        )

        peak_wavelength, peak_qext = (
            find_resonance(
                material,
                qext
            )
        )

        medium_results[material][n_medium] = {
            "qabs": qabs,
            "qsca": qsca,
            "qext": qext,
            "peak_wavelength": peak_wavelength,
            "peak_qext": peak_qext
        }

        medium_summary.append({
            "Material": material,
            "Radius (nm)": FIXED_RADIUS_NM,
            "Host n": n_medium,
            "Resonance (nm)": peak_wavelength,
            "Qext at resonance": round(
                peak_qext,
                4
            )
        })


# ============================================================
# PLOT: GOLD - HOST MEDIUM EFFECT
# ============================================================

plt.figure(figsize=(10, 6))

for n_medium in host_indices:

    data = medium_results["Au"][n_medium]

    plt.plot(
        wavelengths_nm,
        data["qext"],
        label=f"n = {n_medium}"
    )

    plt.scatter(
        data["peak_wavelength"],
        data["peak_qext"],
        s=50,
        zorder=5
    )

    plt.annotate(
        f"{data['peak_wavelength']} nm",
        (
            data["peak_wavelength"],
            data["peak_qext"]
        ),
        xytext=(5, 8),
        textcoords="offset points",
        fontsize=9
    )


plt.title(
    f"Effect of Host Medium on Au Optical Response "
    f"(r = {FIXED_RADIUS_NM} nm)"
)

plt.xlabel("Wavelength (nm)")

plt.ylabel(
    "Extinction Efficiency, $Q_{ext}$"
)

plt.xlim(
    WAVELENGTH_MIN_NM,
    WAVELENGTH_MAX_NM
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend(
    title="Host refractive index"
)

plt.tight_layout()

plt.show()


# ============================================================
# PLOT: SILVER - HOST MEDIUM EFFECT
# ============================================================

plt.figure(figsize=(10, 6))

for n_medium in host_indices:

    data = medium_results["Ag"][n_medium]

    plt.plot(
        wavelengths_nm,
        data["qext"],
        label=f"n = {n_medium}"
    )

    plt.scatter(
        data["peak_wavelength"],
        data["peak_qext"],
        s=50,
        zorder=5
    )

    plt.annotate(
        f"{data['peak_wavelength']} nm",
        (
            data["peak_wavelength"],
            data["peak_qext"]
        ),
        xytext=(5, 8),
        textcoords="offset points",
        fontsize=9
    )


plt.title(
    f"Effect of Host Medium on Ag Optical Response "
    f"(r = {FIXED_RADIUS_NM} nm)"
)

plt.xlabel("Wavelength (nm)")

plt.ylabel(
    "Extinction Efficiency, $Q_{ext}$"
)

plt.xlim(
    WAVELENGTH_MIN_NM,
    WAVELENGTH_MAX_NM
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend(
    title="Host refractive index"
)

plt.tight_layout()

plt.show()


# ============================================================
# PRINT SUMMARY TABLES
# ============================================================

radius_summary_df = pd.DataFrame(radius_summary)

medium_summary_df = pd.DataFrame(medium_summary)


print("\n")
print("=" * 65)
print("STUDY 1: EFFECT OF NANOPARTICLE RADIUS")
print("=" * 65)

print(
    radius_summary_df.to_string(index=False)
)


print("\n")
print("=" * 65)
print("STUDY 2: EFFECT OF HOST-MEDIUM REFRACTIVE INDEX")
print("=" * 65)

print(
    medium_summary_df.to_string(index=False)
)